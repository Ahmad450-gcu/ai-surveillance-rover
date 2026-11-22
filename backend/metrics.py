"""Prometheus metrics of the backend (own module: created once, even when tests reload backend.app)."""
from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter("rover_backend_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram("rover_backend_request_seconds", "HTTP request duration", ["route"],
                         buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5))
ALERTS_IN = Counter("rover_alerts_received_total", "Alerts stored, by type and source (edge detector or stream)",
                    ["type", "source"])
ALERTS_LABELLED = Counter("rover_alert_labels_total", "Labels given to alerts by people", ["label"])
ROVER_FPS = Gauge("rover_detector_fps", "Detector frames per second (last status)")
ROVER_PEOPLE = Gauge("rover_detector_people", "People in view (last status)")
ROVER_STATUS_AGE = Gauge("rover_detector_status_age_seconds", "Seconds since the detector last reported")
ROVER_ONLINE = Gauge("rover_online", "1 if the rover (Pi agent) is online (last status)")
