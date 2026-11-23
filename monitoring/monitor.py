"""Drift monitor service: data drift, prediction drift and concept drift as Prometheus metrics, and the Alertmanager
webhook that turns a drift alert into a retraining request.

    python -m monitoring.monitor        # or the "drift-monitor" service: docker compose --profile monitoring up -d

Every DRIFT_EVERY_SECONDS (default 300) it computes:
  data drift        per camera: PSI of each feature (monitoring/drift.py) of the frames the stream archived in the
                    lake during the last DRIFT_WINDOW_MINUTES, against the training reference (stage drift_reference)
  prediction drift  PSI of the fight probabilities of the last window against the 7 days before it
                    (PostgreSQL table predictions); a change in what the model outputs, even without labels
  concept drift     share of labelled alerts marked "false_alarm" by people in the app/dashboard (last 7 days)

HTTP (port 9200):
  GET  /metrics        Prometheus metrics (rover_drift_*, rover_prediction_*, rover_false_alarm_rate, ...)
  GET  /report         the latest drift report (JSON; also saved to s3://processed/monitoring/drift/)
  POST /alertmanager   Alertmanager webhook: every alert is stored in table ops_events; a firing alert labelled
                       action="retrain" creates a retraining request (table retrain_requests) for every model
                       activity with retraining.on_drift in activities.yaml and starts its Airflow DAG (e.g.
                       fight_retraining, AIRFLOW_URL); if Airflow is down, the DAG's daily run picks the request up
  GET  /health
"""
import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, generate_latest

from monitoring.drift import FEATURES, PSI_DRIFT, frames_from_records, level, psi
from monitoring.drift import collect as collect_features

log = logging.getLogger("drift-monitor")
REFERENCE_KEY = "monitoring/drift_reference.json"
PROB_BINS = {"edges": [i / 10 for i in range(1, 10)]}

DDL = """
CREATE TABLE IF NOT EXISTS ops_events (
    id BIGSERIAL PRIMARY KEY,
    received TIMESTAMPTZ NOT NULL DEFAULT now(),
    status TEXT NOT NULL,              -- firing | resolved
    alertname TEXT NOT NULL,
    severity TEXT,
    summary TEXT,
    labels JSONB NOT NULL,
    starts_at TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS retrain_requests (
    id BIGSERIAL PRIMARY KEY,
    created TIMESTAMPTZ NOT NULL DEFAULT now(),
    reason TEXT NOT NULL,              -- the alert that asked for it, e.g. DataDriftDetected
    details JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',   -- pending | running | done | failed (set by the retraining DAG)
    handled_at TIMESTAMPTZ,
    run_id TEXT
);
ALTER TABLE retrain_requests ADD COLUMN IF NOT EXISTS activity TEXT NOT NULL DEFAULT 'fight';
"""


def drift_retrain_targets():
    """activity -> retraining DAG id, for every activity that retrains on drift (activities.yaml)."""
    try:
        from rover_ai import activities
        return {a.name: a.retraining.dag for a in activities.retrained() if a.retraining.on_drift}
    except Exception as e:                                         # noqa: BLE001 - keep monitoring
        log.warning("activities.yaml not readable (%s): drift retrains the fight model only", e)
        return {"fight": "fight_retraining"}


class Metrics:
    def __init__(self):
        self.registry = r = CollectorRegistry()
        self.psi = Gauge("rover_drift_psi", "PSI of a data feature: a camera's live window vs training reference",
                         ["camera", "feature"], registry=r)
        self.mean = Gauge("rover_drift_feature_mean", "Feature mean (set = live | reference)",
                          ["camera", "feature", "set"], registry=r)
        self.samples = Gauge("rover_drift_samples", "Values in the live window", ["camera", "feature"], registry=r)
        self.frames = Gauge("rover_drift_window_frames", "Stream frames in the live window", ["camera"], registry=r)
        self.detected = Gauge("rover_drift_detected", "1 if one of the camera's features has PSI above the drift "
                              "threshold (with enough samples)", ["camera"], registry=r)
        self.pred_psi = Gauge("rover_prediction_psi", "PSI of fight probabilities: last window vs the 7 days before",
                              registry=r)
        self.pred_mean = Gauge("rover_prediction_mean", "Mean fight probability", ["window"], registry=r)
        self.pred_n = Gauge("rover_prediction_samples", "Judgements", ["window"], registry=r)
        self.labelled = Gauge("rover_alerts_labelled", "Alerts labelled by people in the last 7 days", ["label"],
                              registry=r)
        self.false_alarm = Gauge("rover_false_alarm_rate", "Share of labelled alerts that were false alarms (7 d)",
                                 registry=r)
        self.last_run = Gauge("rover_drift_last_run_timestamp_seconds", "When the drift check last finished",
                              registry=r)
        self.run_seconds = Gauge("rover_drift_run_seconds", "Duration of the last drift check", registry=r)
        self.errors = Counter("rover_drift_errors_total", "Failed drift checks", registry=r)
        self.webhooks = Counter("rover_ops_alerts_received_total", "Alertmanager notifications", ["alertname",
                                "status"], registry=r)
        self.retrain = Counter("rover_retrain_requests_total", "Retraining requests created", ["reason"], registry=r)


class Monitor:
    def __init__(self, lake=None, db=None, window_minutes=60, min_samples=200, min_frames=1000, trigger=None,
                 retrain_targets=None):
        self.lake = lake                # mlops.lake module (S3 client), injectable for tests
        self.db = db                    # callable -> psycopg connection, injectable for tests
        self.window = timedelta(minutes=window_minutes)
        self.min_samples = min_samples          # values per feature, and ...
        self.min_frames = min_frames            # ... frames per camera (~2 min of video): a few seconds of one
                                                # unusual scene is not a camera that drifted
        self.trigger = trigger                  # callable(request_id, dag): starts a retraining DAG (None = don't)
        self.retrain_targets = retrain_targets if retrain_targets is not None else drift_retrain_targets()
        self.metrics = Metrics()
        self.reference = None
        self.report = {}

    # ------------------------------------------------------------------ data drift
    def load_reference(self):
        body = self.lake.client().get_object(Bucket="processed", Key=REFERENCE_KEY)["Body"].read()
        self.reference = json.loads(body)
        log.info("reference: %s (%d clips, created %s)", self.reference["source"], self.reference["clips"],
                 self.reference["created"])

    def live_records(self, now):
        """Frame rows the stream archived in the lake during the window (validated/stream/pose/date=.../...)."""
        import io

        import pyarrow.parquet as pq
        since = now - self.window
        days = {since.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")}
        rows = {}
        for day in sorted(days):
            for obj in self.lake.list_keys("validated", f"stream/pose/date={day}/"):
                if obj["LastModified"] < since:
                    continue
                camera = obj["Key"].split("camera_id=")[1].split("/")[0]
                body = self.lake.client().get_object(Bucket="validated", Key=obj["Key"])["Body"].read()
                t = pq.read_table(io.BytesIO(body), columns=["ts", "height", "people"]).to_pylist()
                rows.setdefault(camera, []).extend(r for r in t if r["ts"] >= since.timestamp())
        return rows                                       # camera -> frame rows

    def data_drift(self, records):
        values = collect_features(frames_from_records(records))
        out = {}
        for f in FEATURES:
            ref, v = self.reference["features"][f], values[f]
            enough = len(v) >= self.min_samples and len(records) >= self.min_frames
            value = psi(ref, v) if enough else None
            out[f] = {"psi": value, "level": level(value) if enough else "insufficient_data", "samples": int(len(v)),
                      "live_mean": float(v.mean()) if len(v) else None, "reference_mean": ref["mean"]}
        return out

    # ------------------------------------------------------------------ prediction + concept drift
    def prediction_drift(self, conn, now):
        recent = np.array([r[0] for r in conn.execute(
            "SELECT probability FROM predictions WHERE event_time > %s", (now - self.window,)).fetchall()])
        base = np.array([r[0] for r in conn.execute(
            "SELECT probability FROM predictions WHERE event_time > %s AND event_time <= %s",
            (now - self.window - timedelta(days=7), now - self.window)).fetchall()])
        out = {"recent_samples": len(recent), "baseline_samples": len(base),
               "recent_mean": float(recent.mean()) if len(recent) else None,
               "baseline_mean": float(base.mean()) if len(base) else None, "psi": None}
        if len(recent) >= self.min_samples and len(base) >= self.min_samples:
            from monitoring.drift import proportions
            ref = {"edges": PROB_BINS["edges"], "proportions": proportions(base, PROB_BINS["edges"]).tolist()}
            out["psi"] = psi(ref, recent)
        return out

    def concept_drift(self, conn, now):
        rows = dict(conn.execute("SELECT label, count(*) FROM alerts WHERE label IS NOT NULL AND received > %s "
                                 "GROUP BY label", (now - timedelta(days=7),)).fetchall())
        total = sum(rows.values())
        return {"labelled": rows, "false_alarm_rate": rows.get("false_alarm", 0) / total if total else None}

    # ------------------------------------------------------------------ one check
    def run_once(self, now=None):
        t0 = time.time()
        now = now or datetime.now(timezone.utc)
        if self.reference is None:
            self.load_reference()
        records = self.live_records(now)
        cameras = {cam: {"frames": len(rows), "features": self.data_drift(rows)} for cam, rows in records.items()}
        with self.db() as conn:
            pred = self.prediction_drift(conn, now)
            concept = self.concept_drift(conn, now)
        drifted = {cam: [f for f, d in c["features"].items() if d["psi"] is not None and d["psi"] > PSI_DRIFT]
                   for cam, c in cameras.items()}
        self.report = {"time": now.isoformat(timespec="seconds"), "window_minutes": self.window.total_seconds() / 60,
                       "reference": {k: self.reference[k] for k in ("source", "created", "clips", "data_md5")},
                       "cameras": cameras, "drifted": {c: f for c, f in drifted.items() if f},
                       "prediction_drift": pred, "concept_drift": concept}
        self.publish()
        self.metrics.run_seconds.set(time.time() - t0)
        self.metrics.last_run.set(time.time())
        log.info("drift check: %s; drifted %s; prediction PSI %s; false-alarm rate %s",
                 ", ".join(f"{c} {v['frames']} frames" for c, v in cameras.items()) or "no stream frames",
                 self.report["drifted"] or "none", _fmt(pred["psi"]), _fmt(concept["false_alarm_rate"]))
        return self.report

    def publish(self):
        m, r = self.metrics, self.report
        for g in (m.frames, m.samples, m.mean, m.psi, m.detected):   # cameras that went quiet disappear
            g.clear()
        for cam, c in r["cameras"].items():
            m.frames.labels(cam).set(c["frames"])
            for f, d in c["features"].items():
                m.samples.labels(cam, f).set(d["samples"])
                m.mean.labels(cam, f, "reference").set(d["reference_mean"])
                if d["psi"] is not None:                            # no sample: no series (not a fake 0)
                    m.psi.labels(cam, f).set(d["psi"])
                    m.mean.labels(cam, f, "live").set(d["live_mean"])
            m.detected.labels(cam).set(1 if cam in r["drifted"] else 0)
        p = r["prediction_drift"]
        m.pred_psi.set(p["psi"] if p["psi"] is not None else float("nan"))
        for w in ("recent", "baseline"):
            m.pred_n.labels(w).set(p[f"{w}_samples"])
            m.pred_mean.labels(w).set(p[f"{w}_mean"] if p[f"{w}_mean"] is not None else float("nan"))
        c = r["concept_drift"]
        for label in ("real", "false_alarm"):
            m.labelled.labels(label).set(c["labelled"].get(label, 0))
        m.false_alarm.set(c["false_alarm_rate"] if c["false_alarm_rate"] is not None else float("nan"))

    def save_report(self):
        """latest.json + one report per hour in s3://processed/monitoring/drift/ (drift history)."""
        body = json.dumps(self.report, indent=1).encode()
        hour = self.report["time"][:13].replace("T", "/")
        for key in ("monitoring/drift/latest.json", f"monitoring/drift/{hour}.json"):
            self.lake.client().put_object(Bucket="processed", Key=key, Body=body, ContentType="application/json")

    # ------------------------------------------------------------------ Alertmanager webhook
    def handle_alertmanager(self, payload):
        """Store each notification; a firing alert with label action=retrain asks for retraining (once while a
        request is still pending)."""
        created = []
        with self.db() as conn:
            for a in payload.get("alerts", []):
                labels, status = a.get("labels", {}), a.get("status", payload.get("status", "firing"))
                name = labels.get("alertname", "unknown")
                conn.execute("INSERT INTO ops_events (status, alertname, severity, summary, labels, starts_at) "
                             "VALUES (%s, %s, %s, %s, %s, %s)",
                             (status, name, labels.get("severity"), a.get("annotations", {}).get("summary"),
                              json.dumps(labels), a.get("startsAt")))
                self.metrics.webhooks.labels(name, status).inc()
                if status != "firing" or labels.get("action") != "retrain":
                    continue
                for activity, dag in self.retrain_targets.items():        # one request per model activity
                    pending = conn.execute("SELECT id FROM retrain_requests WHERE activity = %s AND status IN "
                                           "('pending', 'running') LIMIT 1", (activity,)).fetchone()
                    if pending is not None:                               # one is already waiting / running
                        continue
                    rid = conn.execute("INSERT INTO retrain_requests (reason, details, activity) VALUES (%s, %s, %s) "
                                       "RETURNING id", (name, json.dumps({"alert": a, "drift": self.report}), activity)
                                       ).fetchone()[0]
                    self.metrics.retrain.labels(name).inc()
                    created.append((rid, dag))
                    log.warning("retraining requested (%s): request %d for %s", name, rid, activity)
        for rid, dag in created:                  # after the commit, so the DAG sees the request
            if self.trigger:
                try:
                    self.trigger(rid, dag)
                except Exception as e:            # Airflow down: its daily run picks the pending request up
                    log.warning("could not start %s (%s); it will run at the next schedule", dag, e)
        return {"stored": len(payload.get("alerts", [])), "retrain_requests": [rid for rid, _ in created]}


def airflow_trigger(url, user, password):
    """-> callable(request_id, dag) that starts a DAG run through Airflow's REST API."""
    import base64
    import urllib.request

    def trigger(request_id, dag):
        body = json.dumps({"conf": {"retrain_request": request_id}}).encode()
        req = urllib.request.Request(f"{url.rstrip('/')}/api/v1/dags/{dag}/dagRuns", data=body, method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": "Basic " +
                                              base64.b64encode(f"{user}:{password}".encode()).decode()})
        with urllib.request.urlopen(req, timeout=10) as r:            # noqa: S310 - URL from our own settings
            run = json.loads(r.read())
        log.warning("retraining DAG %s started: %s (request %d)", dag, run.get("dag_run_id"), request_id)
    return trigger


def _fmt(v):
    return "n/a" if v is None else f"{v:.3f}"


def serve(monitor, port):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/metrics":
                self._send(200, generate_latest(monitor.metrics.registry), CONTENT_TYPE_LATEST)
            elif self.path == "/report":
                self._send(200, json.dumps(monitor.report, indent=1).encode())
            elif self.path == "/health":
                self._send(200, b'{"status": "ok"}')
            else:
                self._send(404, b'{"detail": "not found"}')

        def do_POST(self):
            if self.path != "/alertmanager":
                return self._send(404, b'{"detail": "not found"}')
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                self._send(200, json.dumps(monitor.handle_alertmanager(payload)).encode())
            except Exception as e:                                   # never crash the server on a bad payload
                log.exception("webhook failed")
                self._send(400, json.dumps({"detail": str(e)}).encode())

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)       # noqa: S104 - container service
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    import psycopg

    from mlops import lake
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    url = os.environ["DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://")
    monitor = Monitor(lake=lake, db=lambda: psycopg.connect(url, autocommit=True),
                      window_minutes=float(os.environ.get("DRIFT_WINDOW_MINUTES", "60")),
                      min_samples=int(os.environ.get("DRIFT_MIN_SAMPLES", "200")),
                      min_frames=int(os.environ.get("DRIFT_MIN_FRAMES", "1000")),
                      trigger=airflow_trigger(os.environ["AIRFLOW_URL"], os.environ.get("AIRFLOW_USER", "admin"),
                                              os.environ.get("AIRFLOW_PASSWORD", ""))
                      if os.environ.get("AIRFLOW_URL") else None)
    with monitor.db() as conn:
        conn.execute(DDL)
    serve(monitor, int(os.environ.get("PORT", "9200")))
    every = float(os.environ.get("DRIFT_EVERY_SECONDS", "300"))
    log.info("drift monitor on :%s, check every %.0f s over the last %.0f min", os.environ.get("PORT", "9200"),
             every, monitor.window.total_seconds() / 60)
    while True:
        try:
            monitor.run_once()
            monitor.save_report()
        except Exception:
            monitor.metrics.errors.inc()
            log.exception("drift check failed")
        time.sleep(every)


if __name__ == "__main__":
    main()
