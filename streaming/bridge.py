"""Alert bridge: Kafka topic rover.alerts -> backend (POST /api/device/alerts), so stream alerts reach the dashboard
and the app exactly like alerts from the detector.

    python -m streaming.bridge          # or the "alert-bridge" service: docker compose --profile streaming up -d

At-least-once with deduplication: an alert's Kafka offset is committed only after the backend has stored it; if the
backend is down, the bridge retries (backing off up to 30 s) and nothing is lost. Alerts carry a unique id, so one
delivered twice (e.g. a Spark micro-batch replayed after a crash) is stored once.
"""
import json
import logging
import os
import signal
import time
import urllib.error
import urllib.request
from collections import OrderedDict

log = logging.getLogger("bridge")
FIELDS = ("time", "type", "message", "track_ids", "snapshot", "clip", "score", "model_version", "rover_id", "evidence")


class Seen:
    """The last N alert ids (bounded memory)."""

    def __init__(self, size=10000):
        self.ids = OrderedDict()
        self.size = size

    def add(self, key):
        """True if new."""
        if key in self.ids:
            return False
        self.ids[key] = None
        if len(self.ids) > self.size:
            self.ids.popitem(last=False)
        return True


def to_backend(alert):
    """Stream alert -> the backend's AlertIn body (unknown fields dropped)."""
    body = {k: alert[k] for k in FIELDS if alert.get(k) is not None}
    if alert.get("session") is not None and alert.get("frame") is not None:   # where the skeleton frames are
        body["evidence"] = f"stream://{alert.get('rover_id')}/{alert['session']}/{alert['frame']}"
    body["message"] = f"{body.get('message', '')} [stream]"
    return body


def post(url, token, body, timeout=5):
    req = urllib.request.Request(url + "/api/device/alerts", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "X-Device-Token": token})
    with urllib.request.urlopen(req, timeout=timeout) as r:      # noqa: S310 - URL from our own settings
        return json.loads(r.read())


def run():
    from confluent_kafka import Consumer, KafkaException
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    backend = os.environ.get("BACKEND_URL", "http://backend:8080").rstrip("/")
    token = os.environ["DEVICE_TOKEN"]
    topic = os.environ.get("ALERTS_TOPIC", "rover.alerts")
    consumer = Consumer({"bootstrap.servers": os.environ.get("KAFKA_BOOTSTRAP", "kafka:9092"),
                         "group.id": "alert-bridge", "enable.auto.commit": False, "auto.offset.reset": "earliest"})
    consumer.subscribe([topic])
    seen = Seen()
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    log.info("forwarding %s -> %s/api/device/alerts", topic, backend)
    try:
        while not stop:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                log.warning("kafka: %s", msg.error())
                continue
            try:
                alert = json.loads(msg.value())
            except ValueError:
                log.error("skipping malformed alert at offset %d", msg.offset())
                consumer.commit(msg, asynchronous=False)
                continue
            done = not seen.add(alert.get("id") or f"{msg.partition()}:{msg.offset()}")    # duplicate: done
            delay = 1
            while not done and not stop:
                try:
                    stored = post(backend, token, to_backend(alert))
                    log.info("alert %s -> backend id %s: %s", alert.get("id"), stored.get("id"), alert["message"])
                    done = True
                except urllib.error.HTTPError as e:
                    if 400 <= e.code < 500 and e.code != 429:       # rejected (bad token / body): retrying won't help
                        log.error("backend rejected alert %s: HTTP %d %s", alert.get("id"), e.code, e.read()[:200])
                        done = True
                    else:
                        log.warning("backend error HTTP %d, retry in %d s", e.code, delay)
                except OSError as e:                                # backend down: retry, don't lose the alert
                    log.warning("backend unreachable (%s), retry in %d s", e, delay)
                if not done:
                    time.sleep(delay)
                    delay = min(delay * 2, 30)
            if done:                                                # stopped mid-retry: not committed, redone later
                try:
                    consumer.commit(msg, asynchronous=False)
                except KafkaException as e:
                    log.warning("commit failed: %s", e)
    finally:
        consumer.close()


if __name__ == "__main__":
    run()
