"""API tests for the FastAPI backend, on a throwaway SQLite database (CI also runs them against PostgreSQL)."""
import importlib
import os

import pytest
from fastapi.testclient import TestClient

DEVICE = {"X-Device-Token": "test-device-token"}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    if not os.environ.get("TEST_DATABASE_URL"):
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    else:
        monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    monkeypatch.setenv("ROVER__BACKEND__DEVICE_TOKEN", "test-device-token")
    monkeypatch.setenv("ROVER__BACKEND__JWT_SECRET", "test-jwt-secret-that-is-at-least-32-bytes")
    import backend.app
    app_module = importlib.reload(backend.app)
    from backend.db import metadata
    metadata.drop_all(app_module.engine)
    metadata.create_all(app_module.engine)
    app_module.create_user("tester", "secret123")
    with TestClient(app_module.app) as c:
        yield c
    app_module.engine.dispose()


def login(client):
    r = client.post("/api/login", json={"username": "tester", "password": "secret123"})
    assert r.status_code == 200
    return {"Authorization": "Bearer " + r.json()["token"]}


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json()["status"] == "ready"


def test_login_rejects_wrong_password(client):
    assert client.post("/api/login", json={"username": "tester", "password": "nope"}).status_code == 401
    assert client.get("/api/alerts").status_code == 401


def test_alert_ingest_list_label_and_stats(client):
    alert = {"time": "2026-11-22T10:00:00", "type": "fight", "message": "FIGHT: ID 1 and ID 2",
             "track_ids": [1, 2], "score": 0.91, "model_version": "fight-gru/3", "rover_id": "rover-01"}
    assert client.post("/api/device/alerts", json=alert).status_code == 401          # no device token
    r = client.post("/api/device/alerts", json=alert, headers=DEVICE)
    assert r.status_code == 200
    alert_id = r.json()["id"]

    auth = login(client)
    listed = client.get("/api/alerts", headers=auth).json()
    assert listed[0]["id"] == alert_id and listed[0]["track_ids"] == [1, 2] and listed[0]["score"] == 0.91

    assert client.patch(f"/api/alerts/{alert_id}", json={"label": "false_alarm"}, headers=auth).status_code == 200
    assert client.patch("/api/alerts/9999", json={"label": "real"}, headers=auth).status_code == 404
    assert client.get("/api/alerts/stats", headers=auth).json() == {"fight": {"total": 1, "real": 0, "false_alarm": 1}}


def test_commands_are_delivered_once(client):
    auth = login(client)
    assert client.post("/api/command", json={"mode": "guard"}, headers=auth).status_code == 200
    assert client.post("/api/command", json={"mode": "fly"}, headers=auth).status_code == 400
    assert client.get("/api/device/commands", headers=DEVICE).json() == [{"mode": "guard"}]
    assert client.get("/api/device/commands", headers=DEVICE).json() == []


def test_status_reports_detector_online(client):
    auth = login(client)
    assert client.get("/api/status", headers=auth).json()["detector_online"] is False
    client.post("/api/device/status", json={"mode": "patrol", "fps": 11}, headers=DEVICE)
    s = client.get("/api/status", headers=auth).json()
    assert s["detector_online"] is True and s["mode"] == "patrol"


def test_metrics_count_alerts_by_source_and_labels(client):
    def value(text, name):
        line = [ln for ln in text.splitlines() if ln.startswith(name)]
        return float(line[0].split()[-1]) if line else 0.0
    before = client.get("/metrics").text
    alert = {"time": "2026-11-22T10:00:00", "type": "fight", "message": "FIGHT: ID 3 and ID 9 [stream]"}
    alert_id = client.post("/api/device/alerts", json=alert, headers=DEVICE).json()["id"]
    client.patch(f"/api/alerts/{alert_id}", json={"label": "false_alarm"}, headers=login(client))
    client.post("/api/device/status", json={"fps": 9.5, "people": 2}, headers=DEVICE)
    after = client.get("/metrics").text
    stream = 'rover_alerts_received_total{source="stream",type="fight"}'
    assert value(after, stream) - value(before, stream) == 1
    label = 'rover_alert_labels_total{label="false_alarm"}'
    assert value(after, label) - value(before, label) == 1
    assert value(after, "rover_detector_fps") == 9.5
    assert 'rover_backend_requests_total{method="POST",route="/api/device/alerts",status="200"}' in after


def test_existing_database_gets_the_evidence_column(tmp_path):
    """A database created before alerts.evidence existed is migrated when the backend starts."""
    from sqlalchemy import create_engine, inspect, text

    from backend.db import make_engine
    url = f"sqlite:///{tmp_path / 'old.db'}"
    old = create_engine(url)
    with old.begin() as conn:
        conn.execute(text("CREATE TABLE alerts (id INTEGER PRIMARY KEY, time VARCHAR(32) NOT NULL, type VARCHAR(32) "
                          "NOT NULL, message TEXT NOT NULL, track_ids JSON, snapshot TEXT, clip TEXT, score FLOAT, "
                          "model_version VARCHAR(64), rover_id VARCHAR(64), label VARCHAR(16), received DATETIME)"))
    old.dispose()
    engine = make_engine(url)
    assert "evidence" in {c["name"] for c in inspect(engine).get_columns("alerts")}
    make_engine(url).dispose()                                 # running it again changes nothing
    engine.dispose()


def test_stream_alert_evidence_is_stored(client):
    alert = {"time": "2026-11-22T10:00:00", "type": "fight", "message": "FIGHT [stream]",
             "evidence": "stream://rover-1/abc123/87"}
    client.post("/api/device/alerts", json=alert, headers=DEVICE)
    rows = client.get("/api/alerts", headers=login(client)).json()
    rows = rows if isinstance(rows, list) else rows.get("alerts", rows)
    assert any(r.get("evidence") == "stream://rover-1/abc123/87" for r in rows)


def test_activities_come_from_the_registry(client):
    acts = client.get("/api/activities").json()
    assert [a["name"] for a in acts] == ["loitering", "access", "fight", "theft", "crowding"]
    assert {a["alert_type"]: a["kind"] for a in acts}["fight"] == "model"
