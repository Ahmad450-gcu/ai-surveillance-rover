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
