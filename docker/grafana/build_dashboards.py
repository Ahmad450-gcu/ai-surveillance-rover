"""Generate the Grafana dashboards (dashboards as code) into docker/grafana/dashboards/*.json.

    python docker/grafana/build_dashboards.py

Panels are declared below in a few lines each; Grafana loads the generated JSON at start-up (provisioning).
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "dashboards"
PROM = {"type": "prometheus", "uid": "prometheus"}
PG = {"type": "grafana-postgresql-datasource", "uid": "postgres"}


class Layout:
    """Places panels left to right, wrapping at 24 columns."""

    def __init__(self):
        self.x = self.y = self.row_h = 0
        self.panels = []
        self.id = 0

    def add(self, panel, w, h):
        if self.x + w > 24:
            self.x, self.y, self.row_h = 0, self.y + self.row_h, 0
        self.id += 1
        panel.update(id=self.id, gridPos={"x": self.x, "y": self.y, "w": w, "h": h})
        self.x += w
        self.row_h = max(self.row_h, h)
        self.panels.append(panel)

    def row(self, title):
        self.x, self.y = 0, self.y + self.row_h
        self.id += 1
        self.panels.append({"id": self.id, "type": "row", "title": title, "collapsed": False,
                            "gridPos": {"x": 0, "y": self.y, "w": 24, "h": 1}})
        self.y, self.row_h = self.y + 1, 0


def targets(exprs):
    return [{"datasource": PROM, "expr": e, "legendFormat": legend, "refId": chr(65 + i)}
            for i, (e, legend) in enumerate(exprs)]


def thresholds(steps):
    return {"mode": "absolute", "steps": [{"color": c, "value": v} for v, c in steps]}


def timeseries(title, exprs, unit="short", steps=None, desc="", stack=False, show_thresholds=False):
    custom = {"lineWidth": 2, "fillOpacity": 10, "showPoints": "never"}
    if stack:
        custom["stacking"] = {"mode": "normal"}
    if show_thresholds:
        custom["thresholdsStyle"] = {"mode": "line+area"}
    return {"type": "timeseries", "title": title, "description": desc, "datasource": PROM, "targets": targets(exprs),
            "fieldConfig": {"defaults": {"unit": unit, "custom": custom,
                                         "thresholds": thresholds(steps or [(None, "green")])}, "overrides": []},
            "options": {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}}}


def stat(title, exprs, unit="short", steps=None, mappings=None, desc="", text_mode="auto"):
    return {"type": "stat", "title": title, "description": desc, "datasource": PROM, "targets": targets(exprs),
            "fieldConfig": {"defaults": {"unit": unit, "mappings": mappings or [],
                                         "thresholds": thresholds(steps or [(None, "green")])}, "overrides": []},
            "options": {"colorMode": "background", "graphMode": "none", "textMode": text_mode,
                        "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}}}


def bars(title, exprs, desc=""):
    return {"type": "bargauge", "title": title, "description": desc, "datasource": PROM, "targets": targets(exprs),
            "fieldConfig": {"defaults": {"unit": "short", "thresholds": thresholds([(None, "orange")])},
                            "overrides": []},
            "options": {"orientation": "horizontal", "displayMode": "gradient",
                        "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}}}


def heatmap(title, expr, desc=""):
    return {"type": "heatmap", "title": title, "description": desc, "datasource": PROM,
            "targets": [{"datasource": PROM, "expr": expr, "format": "heatmap", "legendFormat": "{{le}}",
                         "refId": "A"}],
            "options": {"calculate": False, "yAxis": {"unit": "percentunit"}, "color": {"scheme": "Oranges"},
                        "cellGap": 1}}


def sql_table(title, sql, desc=""):
    return {"type": "table", "title": title, "description": desc, "datasource": PG,
            "targets": [{"datasource": PG, "rawSql": sql, "format": "table", "rawQuery": True, "editorMode": "code",
                         "refId": "A"}],
            "fieldConfig": {"defaults": {}, "overrides": []}, "options": {"showHeader": True}}


UP_DOWN = [{"type": "value", "options": {"0": {"text": "DOWN", "color": "red"}, "1": {"text": "UP", "color": "green"}}}]
PSI_STEPS = [(None, "green"), (0.1, "orange"), (0.2, "red")]


def dashboard(uid, title, layout, desc, variables=()):
    return {"uid": uid, "title": title, "description": desc, "tags": ["rover"], "timezone": "browser",
            "schemaVersion": 39, "version": 1, "refresh": "30s", "time": {"from": "now-1h", "to": "now"},
            "templating": {"list": list(variables)}, "panels": layout.panels, "editable": False}


# ----------------------------------------------------------------------------------------------- system health
def system():
    L = Layout()
    L.row("Services")
    L.add(stat("Service status", [("up", "{{job}}")], mappings=UP_DOWN,
               steps=[(None, "red"), (1, "green")], text_mode="value_and_name",
               desc="Every scraped service (Prometheus 'up')."), 24, 4)
    L.row("Model serving")
    L.add(stat("Model in production", [('serving_model_info', "{{activity}} v{{version}} ({{arch}})")],
               text_mode="name", desc="The registry @champion the serving API runs (hot-reloaded)."), 6, 5)
    L.add(timeseries("Requests / s", [('sum by (status) (rate(serving_requests_total[1m]))', "HTTP {{status}}")],
                     "reqps", stack=True), 9, 5)
    L.add(timeseries("Latency", [
        ('histogram_quantile(0.5, sum by (le) (rate(serving_request_seconds_bucket[5m])))', "p50"),
        ('histogram_quantile(0.95, sum by (le) (rate(serving_request_seconds_bucket[5m])))', "p95")], "s",
        steps=[(None, "green"), (0.1, "red")], show_thresholds=True), 9, 5)
    L.row("Streaming pipeline (Kafka + Spark)")
    L.add(timeseries("Frames / s into Kafka", [
        ('sum by (topic) (rate(kafka_topic_partition_current_offset{topic=~"rover.*"}[1m]))', "{{topic}}")],
        "short", desc="Messages per second written to each topic."), 8, 7)
    L.add(timeseries("Data quality: frames checked / s", [
        ('sum by (result) (rate(rover_stream_frames_total[1m]))', "{{result}}")], "short", stack=True,
        desc="Valid frames go to the lake, invalid ones to the dead-letter queue."), 8, 7)
    L.add(bars("Rejected frames by reason (24 h)", [
        ('sum by (reason) (increase(rover_stream_rejected_total[24h]))', "{{reason}}")]), 8, 7)
    L.add(timeseries("Event lag (camera → scored)", [
        ('rover_stream_event_lag_seconds', "{{query}}")], "s", steps=[(None, "green"), (30, "red")],
        show_thresholds=True, desc="Now minus the newest frame time of the last micro-batch."), 8, 7)
    L.add(timeseries("Micro-batch duration p95", [
        ('histogram_quantile(0.95, sum by (le, query) (rate(rover_stream_batch_seconds_bucket[5m])))', "{{query}}")],
        "s"), 8, 7)
    L.add(timeseries("Judgements and alerts / min", [
        ('rate(rover_stream_judgements_total[5m]) * 60', "pair judgements"),
        ('rate(rover_stream_alerts_total[5m]) * 60', "alerts")], "short"), 8, 7)
    spark = lambda m: (f'label_replace({{__name__=~"metrics_rover_driver_spark_streaming_.*_{m}_Value"}}, "query", '   # noqa: E731
                       f'"$1", "__name__", "metrics_rover_driver_spark_streaming_(.*)_{m}_Value")')
    L.add(timeseries("Spark: input vs processing rate (records / s)", [
        (spark("inputRate_total"), "input {{query}}"), (spark("processingRate_total"), "processed {{query}}")],
        "short", desc="Spark's own streaming metrics. Processing below input for long = the job falls behind."), 8, 6)
    L.add(timeseries("Spark: scoring state", [(spark("states_rowsTotal"), "camera sessions in state")], "short",
                     desc="applyInPandasWithState: one row per active camera session."), 8, 6)
    L.add(timeseries("Alert bridge backlog", [
        ('sum by (consumergroup, topic) (kafka_consumergroup_lag_sum)', "{{consumergroup}} / {{topic}}")], "short",
        desc="Alerts in Kafka not yet stored by the backend."), 8, 6)
    L.row("Backend and detector")
    L.add(timeseries("Alerts stored / h", [
        ('sum by (type, source) (increase(rover_alerts_received_total[1h]))', "{{type}} ({{source}})")], "short"),
        8, 6)
    L.add(timeseries("Detector", [("rover_detector_fps", "FPS"), ("rover_detector_people", "people in view")],
                     "short"), 8, 6)
    L.add(timeseries("Backend requests / s", [
        ('sum by (status) (rate(rover_backend_requests_total[1m]))', "HTTP {{status}}")], "reqps", stack=True), 8, 6)
    L.row("Ops alerts")
    L.add(sql_table("Alerts from Alertmanager (latest)",
                    "SELECT received, status, alertname, severity, summary FROM ops_events ORDER BY id DESC LIMIT 20"),
          24, 7)
    return dashboard("rover-system", "Rover · System health", L,
                     "Services, model serving, Kafka + Spark pipeline, backend.")


# ----------------------------------------------------------------------------------------------- model + drift
def model():
    camera = {"name": "camera", "label": "Camera", "type": "query", "datasource": PROM,
              "query": {"query": "label_values(rover_drift_psi, camera)", "refId": "cam"},
              "definition": "label_values(rover_drift_psi, camera)", "refresh": 2, "includeAll": True,
              "multi": True, "current": {"text": "All", "value": "$__all"}}
    L = Layout()
    L.row("Data drift: live frames vs training data (PSI per feature)")
    L.add(stat("Drift status per camera", [('rover_drift_detected{camera=~"$camera"}', "{{camera}}")],
               mappings=[{"type": "value", "options": {"0": {"text": "STABLE", "color": "green"},
                                                       "1": {"text": "DRIFT", "color": "red"}}}],
               steps=[(None, "green"), (1, "red")], text_mode="value_and_name",
               desc="DRIFT when any feature's PSI > 0.2 (alert DataDriftDetected → retraining request)."), 8, 6)
    L.add(stat("Frames in the window", [('rover_drift_window_frames{camera=~"$camera"}', "{{camera}}")],
               text_mode="value_and_name"), 8, 6)
    L.add(stat("Last drift check", [('time() - rover_drift_last_run_timestamp_seconds', "seconds ago")], "s",
               steps=[(None, "green"), (900, "red")]), 8, 6)
    L.add(timeseries("PSI per feature", [('rover_drift_psi{camera=~"$camera"}', "{{camera}} · {{feature}}")],
                     "short", steps=PSI_STEPS, show_thresholds=True,
                     desc="< 0.1 stable · 0.1-0.2 moderate · > 0.2 drift"), 12, 9)
    L.add(timeseries("Feature means: live vs training", [
        ('rover_drift_feature_mean{camera=~"$camera", set="live"}', "live {{camera}} · {{feature}}"),
        ('max by (feature) (rover_drift_feature_mean{set="reference"})', "training · {{feature}}")], "short"), 12, 9)
    L.row("Prediction drift (model outputs)")
    L.add(timeseries("Prediction PSI (last window vs previous 7 days)", [("rover_prediction_psi", "PSI")], "short",
                     steps=PSI_STEPS, show_thresholds=True), 8, 8)
    L.add(timeseries("Mean fight probability", [('rover_prediction_mean', "{{window}}")], "percentunit"), 8, 8)
    L.add(heatmap("Probability distribution (serving)",
                  'sum by (le) (increase(serving_probability_bucket[5m]))',
                  desc="How the served model's probabilities spread over time."), 8, 8)
    L.add(timeseries("Windows above the threshold", [
        ('sum(rate(serving_above_threshold_total[5m])) / clamp_min(sum(rate(serving_windows_total[5m])), 1e-9)',
         "share above threshold")], "percentunit"), 12, 7)
    L.add(stat("Model in production", [('serving_model_info', "{{activity}} v{{version}} ({{arch}})")],
               text_mode="name"), 12, 7)
    L.row("Concept drift (human feedback)")
    L.add(stat("False-alarm rate (labelled alerts, 7 d)", [("rover_false_alarm_rate", "")], "percentunit",
               steps=[(None, "green"), (0.3, "orange"), (0.5, "red")]), 8, 6)
    L.add(stat("Labelled alerts (7 d)", [('rover_alerts_labelled', "{{label}}")], text_mode="value_and_name"), 8, 6)
    L.add(stat("Retraining requests", [('sum(rover_retrain_requests_total)', "")],
               steps=[(None, "green"), (1, "orange")]), 8, 6)
    L.add(sql_table("Retraining requests (→ Phase 7 pipeline)",
                    "SELECT id, created, reason, status, handled_at FROM retrain_requests ORDER BY id DESC LIMIT 20"),
          12, 7)
    L.add(sql_table("Stream judgements per camera (last hour)",
                    "SELECT camera_id, count(*) AS judgements, count(*) FILTER (WHERE alert) AS alerts, "
                    "round(avg(probability)::numeric, 3) AS mean_probability, max(model_version) AS model "
                    "FROM predictions WHERE event_time > now() - interval '1 hour' GROUP BY 1 ORDER BY 2 DESC"),
          12, 7)
    return dashboard("rover-model", "Rover · Model & drift", L,
                     "Data drift vs the training data, prediction drift, concept drift, retraining requests.",
                     [camera])


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for d in (system(), model()):
        path = OUT / f"{d['uid']}.json"
        path.write_text(json.dumps(d, indent=1) + "\n")
        print(f"{path.name}: {len(d['panels'])} panels")
