"""FastAPI backend: stores alerts, relays rover status/commands, serves the dashboard.

Run (from the project folder):
    python -m backend.manage create-user admin      # once
    python -m backend                               # http://<laptop-ip>:8080
or as a container:  docker compose up -d backend   (PostgreSQL via DATABASE_URL)

Who talks to it:
    run.py (the detector)  -> /api/device/*   with header X-Device-Token (config backend.device_token)
    dashboard / phone app  -> /api/*          with a login token (JWT)
    Docker / monitoring    -> /health, /ready
"""
import hashlib
import hmac
import logging
import secrets
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
from sqlalchemy import case, func, insert, select, text, update
from sqlalchemy.exc import IntegrityError

from backend.db import alerts, commands, make_engine, telemetry, users
from backend.metrics import (
    ALERTS_IN,
    ALERTS_LABELLED,
    HTTP_LATENCY,
    HTTP_REQUESTS,
    ROVER_FPS,
    ROVER_ONLINE,
    ROVER_PEOPLE,
    ROVER_STATUS_AGE,
)
from rover_ai.settings import load_config

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
CFG = load_config(ROOT / "config.yaml")
ALERTS_DIR = (ROOT / CFG["alerts"]["dir"]).resolve()
DEVICE_TOKEN = str(CFG["backend"]["device_token"])
SECRET_PATH = HERE / ".jwt_secret"
TOKEN_HOURS = 12

log = logging.getLogger("backend")


def _activities():
    """The activity registry (activities.yaml); the built-in four if it can't be read."""
    try:
        from rover_ai import activities
        return [dict(name=a.name, title=a.title, kind=a.kind, alert_type=a.alert_type, description=a.description)
                for a in activities.load().values()]
    except Exception as e:                                       # noqa: BLE001
        log.warning("activities.yaml not readable (%s): using the built-in activity list", e)
        return [dict(name=n, title=n.capitalize(), kind="rule", alert_type=t, description="")
                for n, t in (("loitering", "loitering"), ("access", "restricted"), ("fight", "fight"), ("theft", "theft"))]


ACTIVITIES = _activities()
ALERT_TYPES = tuple(a["alert_type"] for a in ACTIVITIES)
MODES = ("manual", "guard", "patrol")
DRIVES = ("forward", "back", "left", "right", "stop")


def _jwt_secret():
    """From the environment (ROVER__BACKEND__JWT_SECRET) in Docker/production, else a generated local file."""
    if CFG["backend"].get("jwt_secret"):
        return str(CFG["backend"]["jwt_secret"])
    if not SECRET_PATH.exists():
        SECRET_PATH.write_text(secrets.token_hex(32))
    return SECRET_PATH.read_text().strip()


JWT_SECRET = _jwt_secret()


# ============================== database ==============================
engine = make_engine()


def rows(stmt):
    with engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(stmt)]


def one(stmt):
    found = rows(stmt)
    return found[0] if found else None


def execute(stmt):
    with engine.begin() as conn:
        return conn.execute(stmt)


def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000).hex()
    return f"{salt}${digest}"


def check_password(password, stored):
    salt, _ = stored.split("$", 1)
    return hmac.compare_digest(hash_password(password, salt), stored)


def create_user(username, password):
    """Raises ValueError if the user already exists."""
    try:
        execute(insert(users).values(username=username, pw_hash=hash_password(password)))
    except IntegrityError:
        raise ValueError(f"User {username} already exists.") from None


# ============================== auth ==============================
def current_user(authorization: str = Header(default=""), token: str = Query(default="")):
    """Login token from the Authorization header, or ?token= (needed for <img>/<video> links)."""
    raw = authorization.removeprefix("Bearer ").strip() or token
    try:
        return jwt.decode(raw, JWT_SECRET, algorithms=["HS256"])["sub"]
    except jwt.PyJWTError:
        raise HTTPException(401, "Please log in") from None


def device(x_device_token: str = Header(default="")):
    if not hmac.compare_digest(x_device_token, DEVICE_TOKEN):
        raise HTTPException(401, "Bad device token")


# ============================== app ==============================
app = FastAPI(title="AI Surveillance Rover")
LIVE = {"status": None, "seen": 0.0, "stored": 0.0}
TELEMETRY_EVERY_S = 10

# ---------------- Prometheus metrics (GET /metrics) ----------------
@app.middleware("http")
async def count_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    path = route.path if route is not None else "other"       # the route template, not the raw URL (low cardinality)
    if path not in ("/metrics", "/health", "/ready"):
        HTTP_REQUESTS.labels(request.method, path, str(response.status_code)).inc()
        HTTP_LATENCY.labels(path).observe(time.perf_counter() - t0)
    return response


@app.get("/metrics")
def metrics():
    st = LIVE["status"] or {}
    if LIVE["seen"]:
        ROVER_STATUS_AGE.set(time.time() - LIVE["seen"])
        ROVER_FPS.set(st.get("fps") or 0)
        ROVER_PEOPLE.set(st.get("people") or 0)
        ROVER_ONLINE.set(1 if (st.get("rover") or {}).get("online") else 0)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


class Login(BaseModel):
    username: str
    password: str


class AlertIn(BaseModel):
    time: str
    type: str
    message: str
    track_ids: list = []
    snapshot: str | None = None
    clip: str | None = None
    score: float | None = None
    model_version: str | None = None
    rover_id: str | None = None
    evidence: str | None = None


class Label(BaseModel):
    label: str          # "real" | "false_alarm"


class Command(BaseModel):
    mode: str | None = None
    drive: str | None = None


@app.get("/health")
def health():
    """Liveness probe (Docker healthcheck / load balancer)."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness probe: the database answers."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"database unavailable: {type(e).__name__}") from None
    return {"status": "ready", "database": engine.dialect.name}


@app.post("/api/login")
def login(body: Login):
    user = one(select(users).where(users.c.username == body.username))
    if not user or not check_password(body.password, user["pw_hash"]):
        raise HTTPException(401, "Wrong username or password")
    exp = datetime.now(timezone.utc) + timedelta(hours=TOKEN_HOURS)
    return {"token": jwt.encode({"sub": user["username"], "exp": exp}, JWT_SECRET, algorithm="HS256"),
            "username": user["username"]}


@app.get("/api/me")
def me(user=Depends(current_user)):
    return {"username": user}


# ---------------- alerts ----------------
@app.get("/api/activities")
def list_activities():
    """Registered activities (activities.yaml): what can raise an alert, and whether it is a rule or a model."""
    return ACTIVITIES


@app.get("/api/alerts")
def list_alerts(limit: int = 50, type: str | None = None, after_id: int = 0, user=Depends(current_user)):
    stmt = select(alerts).where(alerts.c.id > after_id)
    if type:
        stmt = stmt.where(alerts.c.type == type)
    return rows(stmt.order_by(alerts.c.id.desc()).limit(min(limit, 500)))


@app.get("/api/alerts/stats")
def alert_stats(user=Depends(current_user)):
    stmt = select(
        alerts.c.type,
        func.count().label("n"),
        func.sum(case((alerts.c.label == "real", 1), else_=0)).label("real"),
        func.sum(case((alerts.c.label == "false_alarm", 1), else_=0)).label("false_alarm"),
    ).group_by(alerts.c.type)
    return {r["type"]: {"total": r["n"], "real": r["real"] or 0, "false_alarm": r["false_alarm"] or 0}
            for r in rows(stmt)}


@app.patch("/api/alerts/{alert_id}")
def label_alert(alert_id: int, body: Label, user=Depends(current_user)):
    if body.label not in ("real", "false_alarm", ""):
        raise HTTPException(400, "label must be 'real' or 'false_alarm'")
    if execute(update(alerts).where(alerts.c.id == alert_id).values(label=body.label or None)).rowcount == 0:
        raise HTTPException(404, "No such alert")
    ALERTS_LABELLED.labels(body.label or "cleared").inc()
    return {"ok": True}


@app.get("/media/{path:path}")
def media(path: str, user=Depends(current_user)):
    file = (ALERTS_DIR / path).resolve()
    if not file.is_relative_to(ALERTS_DIR) or not file.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(file)


# ---------------- rover status & commands ----------------
@app.get("/api/status")
def status(user=Depends(current_user)):
    online = time.time() - LIVE["seen"] < 5
    return {"detector_online": online, "last_seen": LIVE["seen"] or None, **((LIVE["status"] or {}) if online else {})}


@app.post("/api/command")
def command(body: Command, user=Depends(current_user)):
    if body.mode and body.mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    if body.drive and body.drive not in DRIVES:
        raise HTTPException(400, f"drive must be one of {DRIVES}")
    if not body.mode and not body.drive:
        raise HTTPException(400, "send a mode or a drive command")
    execute(insert(commands).values(data=body.model_dump(exclude_none=True), username=user))
    return {"ok": True}


# ---------------- device (detector) side ----------------
@app.post("/api/device/alerts", dependencies=[Depends(device)])
def device_alert(body: AlertIn):
    with engine.begin() as conn:
        alert_id = conn.execute(insert(alerts).values(**body.model_dump()).returning(alerts.c.id)).scalar_one()
    ALERTS_IN.labels(body.type, "stream" if body.message.endswith("[stream]") else "edge").inc()
    return {"id": alert_id}


@app.post("/api/device/status", dependencies=[Depends(device)])
def device_status(body: dict):
    LIVE["status"], LIVE["seen"] = body, time.time()
    if time.time() - LIVE["stored"] >= TELEMETRY_EVERY_S:     # keep one telemetry row every 10 s
        LIVE["stored"] = time.time()
        execute(insert(telemetry).values(data=body))
    return {"ok": True}


@app.get("/api/device/commands", dependencies=[Depends(device)])
def device_commands():
    with engine.begin() as conn:
        pending = conn.execute(select(commands.c.id, commands.c.data)
                               .where(commands.c.delivered.is_(False)).order_by(commands.c.id)).all()
        if pending:
            conn.execute(update(commands).where(commands.c.id.in_([r.id for r in pending])).values(delivered=True))
    return [r.data for r in pending]


# ---------------- dashboard ----------------
@app.get("/")
def dashboard():
    return FileResponse(HERE / "static" / "index.html")
