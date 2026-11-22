"""Spark Structured Streaming job: Kafka frame records -> validation -> lake / dead-letter -> fight scoring -> alerts.

    docker compose --profile streaming up -d        # runs this job in the "spark" container (UI: http://localhost:4040)

Two streaming queries read topic rover.pose-frames:

1. "quality" (every 10 s): each record is checked (valid JSON with the right types, schema version, required fields, 17 keypoints of 3 values,
   boxes, confidences and coordinates in range, event time not too old / not in the future); duplicates (same
   session and frame) within a micro-batch are written once.
     valid   -> s3://validated/stream/pose/date=.../camera_id=.../batch-<id>.parquet   (training / drift data)
     invalid -> topic rover.pose-frames.dlq + s3://rejected/stream/pose/...json        (with the reason)
2. "fight-scoring" (every 2 s): per camera session, a stateful function (applyInPandasWithState) rebuilds the
   tracks and judges every close pair with the served @champion model, exactly like the edge (streaming/scoring.py).
   Repeated or out-of-order frames of a session are skipped by that state.
     every judgement -> PostgreSQL table predictions   (monitoring and drift, Phase 6)
     alerts          -> topic rover.alerts             (the alert bridge forwards them to the backend)

Delivery: Kafka offsets and the scoring state are checkpointed, so a restart continues where it stopped. The sinks
are idempotent (files named by batch id, predictions unique per frame and pair, alert ids), so a replayed
micro-batch doesn't create duplicates.
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

log = logging.getLogger("stream")

KAFKA = os.environ.get("KAFKA_BOOTSTRAP", "kafka:9092")
TOPIC = os.environ.get("FRAMES_TOPIC", "rover.pose-frames")
DLQ_TOPIC = os.environ.get("DLQ_TOPIC", "rover.pose-frames.dlq")
ALERTS_TOPIC = os.environ.get("ALERTS_TOPIC", "rover.alerts")
SERVING_URL = os.environ.get("SERVING_URL", "http://serving:8000/v1/models/fight")
CHECKPOINTS = os.environ.get("CHECKPOINT_DIR", "/checkpoints")
MAX_LAG = int(os.environ.get("MAX_LAG_SECONDS", "300"))        # older events are "too_late"
MAX_SKEW = 60                                                     # events this far in the future: clock problem
STATE_TIMEOUT_MS = 120_000                                        # a camera session silent this long is forgotten

PREDICTIONS_DDL = """
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    activity TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    session TEXT NOT NULL,
    frame INTEGER NOT NULL,
    t DOUBLE PRECISION NOT NULL,
    event_time TIMESTAMPTZ NOT NULL,
    person_a INTEGER NOT NULL,
    person_b INTEGER NOT NULL,
    probability REAL NOT NULL,
    threshold REAL NOT NULL,
    above BOOLEAN NOT NULL,
    alert BOOLEAN NOT NULL,
    model_version TEXT,
    scored_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (activity, camera_id, session, frame, person_a, person_b)
);
CREATE INDEX IF NOT EXISTS predictions_event_time ON predictions (event_time);
"""


# ----------------------------------------------------------------------------------------------- schema + rules
def frame_schema():
    from pyspark.sql.types import ArrayType, DoubleType, IntegerType, LongType, StringType, StructField, StructType
    person = StructType([StructField("id", IntegerType()), StructField("box", ArrayType(DoubleType())),
                         StructField("conf", DoubleType()), StructField("kpts", ArrayType(ArrayType(DoubleType())))])
    obj = StructType([StructField("id", IntegerType()), StructField("name", StringType()),
                      StructField("box", ArrayType(DoubleType())), StructField("conf", DoubleType())])
    return StructType([StructField("schema", IntegerType()), StructField("camera_id", StringType()),
                       StructField("session", StringType()), StructField("frame", LongType()),
                       StructField("t", DoubleType()), StructField("ts", DoubleType()),
                       StructField("width", IntegerType()), StructField("height", IntegerType()),
                       StructField("people", ArrayType(person)), StructField("objects", ArrayType(obj)),
                       StructField("_corrupt_record", StringType())])    # filled by Spark: not JSON / wrong types


# (reason, SQL condition that makes a record invalid), checked in this order; first match wins
RULES = [
    ("malformed", "f IS NULL OR f._corrupt_record IS NOT NULL"),
    ("unsupported_schema", "f.schema IS NULL OR f.schema != 1"),
    ("missing_camera", "f.camera_id IS NULL OR length(f.camera_id) = 0 OR f.session IS NULL"),
    ("missing_time", "f.frame IS NULL OR f.t IS NULL OR f.ts IS NULL"),
    ("bad_frame_size", "f.width IS NULL OR f.height IS NULL OR f.width <= 0 OR f.height <= 0"),
    ("missing_people", "f.people IS NULL"),
    ("bad_person_shape", "exists(f.people, p -> p.id IS NULL OR p.box IS NULL OR size(p.box) != 4 OR p.kpts IS NULL"
                         " OR size(p.kpts) != 17 OR exists(p.kpts, k -> k IS NULL OR size(k) != 3))"),
    ("bad_box", "exists(f.people, p -> p.box[2] <= p.box[0] OR p.box[3] <= p.box[1] OR p.conf < 0 OR p.conf > 1)"),
    ("keypoint_out_of_range", "exists(f.people, p -> exists(p.kpts, k -> k[2] < 0 OR k[2] > 1"
                              " OR k[0] < -0.1 * f.width OR k[0] > 1.1 * f.width"
                              " OR k[1] < -0.1 * f.height OR k[1] > 1.1 * f.height))"),
    ("too_late", f"f.ts < unix_timestamp() - {MAX_LAG}"),
    ("clock_skew", f"f.ts > unix_timestamp() + {MAX_SKEW}"),
]


def reason_expr():
    from pyspark.sql import functions as F
    expr = None
    for reason, cond in RULES:
        expr = (F.when(F.expr(cond), F.lit(reason)) if expr is None else expr.when(F.expr(cond), F.lit(reason)))
    return expr.otherwise(F.lit(None))


def read_frames(spark):
    from pyspark.sql import functions as F
    raw = (spark.readStream.format("kafka").option("kafka.bootstrap.servers", KAFKA).option("subscribe", TOPIC)
           .option("startingOffsets", "latest").option("maxOffsetsPerTrigger", 5000)
           .option("failOnDataLoss", "false").load())
    return (raw.select(F.col("value").cast("string").alias("json"), F.col("partition"), F.col("offset"),
                       F.col("timestamp").alias("kafka_time"))
            .withColumn("f", F.from_json("json", frame_schema(), {"mode": "PERMISSIVE"}))
            .withColumn("reason", reason_expr()))


# ----------------------------------------------------------------------------------------------- metrics
class Metrics:
    """Pipeline metrics of this job, served on METRICS_PORT (9101) for Prometheus. Spark's own streaming metrics
    (input rate, batch duration, state rows) are on the UI port: /metrics/prometheus/."""

    def __init__(self):
        from prometheus_client import Counter, Gauge, Histogram
        self.frames = Counter("rover_stream_frames_total", "Frame records checked", ["result"])
        self.rejected = Counter("rover_stream_rejected_total", "Frame records sent to the dead-letter queue",
                                ["reason"])
        self.judgements = Counter("rover_stream_judgements_total", "Pair judgements by the stream")
        self.alerts = Counter("rover_stream_alerts_total", "Alerts raised by the stream")
        self.batch = Histogram("rover_stream_batch_seconds", "Micro-batch sink duration", ["query"],
                               buckets=(.1, .25, .5, 1, 2, 5, 10, 30))
        self.last = Gauge("rover_stream_last_batch_timestamp_seconds", "When a micro-batch with data finished",
                          ["query"])
        self.lag = Gauge("rover_stream_event_lag_seconds", "Now minus the newest event time of the last batch",
                         ["query"])


METRICS = None


def metrics():
    global METRICS
    if METRICS is None:
        from prometheus_client import start_http_server
        METRICS = Metrics()
        start_http_server(int(os.environ.get("METRICS_PORT", "9101")))
    return METRICS


# ----------------------------------------------------------------------------------------------- query 1: quality
def lake():
    from mlops import lake as lk
    return lk


def quality_batch(df, batch_id):
    """Split one micro-batch into the lake's validated zone and the dead-letter topic / rejected zone."""
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyspark.sql import functions as F
    t0 = time.time()
    df = df.persist()
    bad = df.filter(F.col("reason").isNotNull())
    good = df.filter(F.col("reason").isNull())
    bad_reasons = {r["reason"]: r["count"] for r in bad.groupBy("reason").count().collect()}
    n_bad = sum(bad_reasons.values())
    if n_bad:
        dead = bad.select(F.col("f.camera_id").alias("key"), F.to_json(F.struct(
            F.col("reason"), F.col("json").alias("record"), F.col("partition"), F.col("offset"),
            F.col("kafka_time"), F.current_timestamp().alias("rejected_at"))).alias("value"))
        dead.write.format("kafka").option("kafka.bootstrap.servers", KAFKA).option("topic", DLQ_TOPIC).save()
        rows = [json.loads(r.value) for r in dead.collect()]
        day = time.strftime("%Y-%m-%d")
        lake().client().put_object(Bucket="rejected", Key=f"stream/pose/date={day}/batch-{batch_id:08d}.json",
                                   Body="\n".join(json.dumps(r) for r in rows).encode(),
                                   ContentType="application/json")
    flat = (good.select("f.camera_id", "f.session", "f.frame", "f.t", "f.ts", "f.width", "f.height",
                        F.to_json("f.people").alias("people"), F.to_json("f.objects").alias("objects"),
                        F.size("f.people").alias("n_people"), "partition", "offset")
            .dropDuplicates(["session", "frame"]))
    pdf = flat.toPandas()
    if len(pdf):
        pdf["date"] = [time.strftime("%Y-%m-%d", time.gmtime(ts)) for ts in pdf["ts"]]
        for (day, cam), part in pdf.groupby(["date", "camera_id"]):
            buf = io.BytesIO()
            pq.write_table(pa.Table.from_pandas(part.drop(columns=["date", "camera_id"]), preserve_index=False), buf)
            lake().client().put_object(Bucket="validated", Body=buf.getvalue(),
                                       Key=f"stream/pose/date={day}/camera_id={cam}/batch-{batch_id:08d}.parquet")
    df.unpersist()
    m = metrics()
    m.frames.labels("valid").inc(len(pdf))
    m.frames.labels("invalid").inc(n_bad)
    for reason, n in bad_reasons.items():
        m.rejected.labels(reason).inc(n)
    if len(pdf) or n_bad:
        m.batch.labels("quality").observe(time.time() - t0)
        m.last.labels("quality").set(time.time())
        if len(pdf):
            m.lag.labels("quality").set(time.time() - float(pdf["ts"].max()))
    log.info("quality batch %d: %d valid -> validated/, %d invalid -> %s", batch_id, len(pdf), n_bad, DLQ_TOPIC)


# ----------------------------------------------------------------------------------------------- query 2: scoring
_DETECTOR = None


def detector():
    """One fight detector per Python worker, its model behind the serving API (always the @champion)."""
    global _DETECTOR
    if _DETECTOR is None:
        from rover_ai.learned import LearnedFightDetector
        cfg = {"model": SERVING_URL, "model_threshold": None, "model_consecutive": None, "proximity": 3.5,
               "window_seconds": 0.6}
        _DETECTOR = LearnedFightDetector(cfg, ROOT)
    return _DETECTOR


def score_group(key, pdfs, state):
    """applyInPandasWithState: all new frames of one camera session -> predictions (incl. alert flag)."""
    import pandas as pd

    from streaming import scoring
    if state.hasTimedOut:
        state.remove()
        return
    records = [json.loads(j) for pdf in pdfs for j in pdf["json"]]
    try:
        det = detector()
    except OSError as e:                         # serving down: keep the frames as track context, judge none
        log.warning("serving API unreachable (%s); frames of %s not scored", e, key)
        st = scoring.loads(state.get[0] if state.exists else "")
        st["frames"] = (st["frames"] + sorted(records, key=lambda r: r["t"]))[-300:]
        state.update((scoring.dumps(st),))
        state.setTimeoutDuration(STATE_TIMEOUT_MS)
        return
    st = scoring.loads(state.get[0] if state.exists else "")
    preds, _, st, _ = scoring.score_frames(det, st, records)
    state.update((scoring.dumps(st),))
    state.setTimeoutDuration(STATE_TIMEOUT_MS)
    if preds:
        out = pd.DataFrame(preds)
        out["message"] = [f"FIGHT: ID {a} and ID {b} ({p:.0%})" for a, b, p in
                          zip(out.person_a, out.person_b, out.probability)]
        yield out[SCORE_COLUMNS]


SCORE_COLUMNS = ["camera_id", "session", "frame", "t", "ts", "person_a", "person_b", "probability", "threshold",
                 "above", "alert", "model_version", "message"]


def score_schema():
    from pyspark.sql.types import BooleanType, DoubleType, IntegerType, LongType, StringType, StructField, StructType
    types = dict(camera_id=StringType(), session=StringType(), frame=LongType(), t=DoubleType(), ts=DoubleType(),
                 person_a=IntegerType(), person_b=IntegerType(), probability=DoubleType(), threshold=DoubleType(),
                 above=BooleanType(), alert=BooleanType(), model_version=StringType(), message=StringType())
    return StructType([StructField(c, types[c]) for c in SCORE_COLUMNS])


def pg_connect():
    import psycopg
    return psycopg.connect(os.environ["DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://"),
                           autocommit=True)


def scores_batch(df, batch_id):
    """Predictions -> PostgreSQL (idempotent), alerts -> Kafka (id = camera/session/frame/pair, deduplicated
    by the bridge)."""
    from pyspark.sql import functions as F
    t0 = time.time()
    df = df.persist()                    # used twice below: without this the stateful scoring would run twice
    pdf = df.toPandas()
    if not len(pdf):
        df.unpersist()
        return
    with pg_connect() as conn, conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO predictions (activity, camera_id, session, frame, t, event_time, person_a, person_b,"
            " probability, threshold, above, alert, model_version) VALUES"
            " ('fight', %s, %s, %s, %s, to_timestamp(%s), %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [(r.camera_id, r.session, int(r.frame), float(r.t), float(r.ts), int(r.person_a), int(r.person_b),
              float(r.probability), float(r.threshold), bool(r.above), bool(r.alert), r.model_version)
             for r in pdf.itertuples()])
    alerts = df.filter(F.col("alert"))
    n_alerts = int(pdf["alert"].sum())
    m = metrics()
    m.judgements.inc(len(pdf))
    m.alerts.inc(n_alerts)
    m.last.labels("fight-scoring").set(time.time())
    m.lag.labels("fight-scoring").set(time.time() - float(pdf["ts"].max()))
    if n_alerts:
        (alerts.select(F.col("camera_id").alias("key"), F.to_json(F.struct(
            F.concat_ws("-", "camera_id", "session", "frame", "person_a", "person_b").alias("id"),
            F.date_format(F.from_unixtime("ts"), "yyyy-MM-dd'T'HH:mm:ss").alias("time"),
            F.lit("fight").alias("type"), F.col("message"), F.array("person_a", "person_b").alias("track_ids"),
            F.round("probability", 4).alias("score"),
            F.concat(F.lit("fight-detector v"), F.col("model_version")).alias("model_version"),
            F.col("camera_id").alias("rover_id"), F.col("session"), F.col("frame"), F.col("t"),
            F.lit("stream").alias("source"))).alias("value"))
         .write.format("kafka").option("kafka.bootstrap.servers", KAFKA).option("topic", ALERTS_TOPIC).save())
    df.unpersist()
    m.batch.labels("fight-scoring").observe(time.time() - t0)
    log.info("scoring batch %d: %d judgements -> predictions, %d alerts -> %s", batch_id, len(pdf), n_alerts,
             ALERTS_TOPIC)


# ----------------------------------------------------------------------------------------------- main
def main():
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.streaming.state import GroupStateTimeout
    from pyspark.sql.types import StringType, StructField, StructType

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("py4j").setLevel(logging.WARNING)
    with pg_connect() as conn:
        conn.execute(PREDICTIONS_DDL)
    metrics()                                    # start the metrics endpoint now, not at the first batch

    spark = (SparkSession.builder.appName("rover-stream")
             .config("spark.sql.shuffle.partitions", os.environ.get("SHUFFLE_PARTITIONS", "2"))   # = state partitions
             .config("spark.sql.streaming.stateStore.stateSchemaCheck", "true")
             # Spark's own metrics for Prometheus: http://spark:4040/metrics/prometheus/ (stable names: namespace)
             .config("spark.ui.prometheus.enabled", "true")
             .config("spark.sql.streaming.metricsEnabled", "true")
             .config("spark.metrics.namespace", "rover")
             .config("spark.metrics.conf.*.sink.prometheusServlet.class",
                     "org.apache.spark.metrics.sink.PrometheusServlet")
             .config("spark.metrics.conf.*.sink.prometheusServlet.path", "/metrics/prometheus")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    frames = read_frames(spark)

    quality = (frames.writeStream.queryName("quality").foreachBatch(quality_batch)
               .option("checkpointLocation", f"{CHECKPOINTS}/quality").trigger(processingTime="10 seconds").start())

    valid = (frames.filter(F.col("reason").isNull())
             .select("f.camera_id", "f.session", "json"))
    scored = valid.groupBy("camera_id", "session").applyInPandasWithState(
        score_group, outputStructType=score_schema(),
        stateStructType=StructType([StructField("state", StringType())]),
        outputMode="append", timeoutConf=GroupStateTimeout.ProcessingTimeTimeout)
    scoring_q = (scored.writeStream.queryName("fight-scoring").foreachBatch(scores_batch)
                 .option("checkpointLocation", f"{CHECKPOINTS}/fight-scoring").trigger(processingTime="2 seconds")
                 .start())
    log.info("streaming queries started: %s, %s (Spark UI on port 4040)", quality.name, scoring_q.name)
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
