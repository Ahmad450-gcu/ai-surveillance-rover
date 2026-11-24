"""Operational database of the backend (alerts, telemetry, rover commands, users).

SQLAlchemy Core, so the same code runs on:
  - PostgreSQL in Docker / production   DATABASE_URL=postgresql+psycopg://user:pass@postgres:5432/rover
  - a local SQLite file for quick dev   (default when DATABASE_URL is not set)
"""
import os
from pathlib import Path

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON

HERE = Path(__file__).resolve().parent
DEFAULT_URL = f"sqlite:///{HERE / 'surveillance.db'}"

metadata = MetaData()
JsonType = JSON().with_variant(JSONB(), "postgresql")

users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True),
    Column("username", String(64), unique=True, nullable=False),
    Column("pw_hash", String(200), nullable=False),
    Column("created", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

alerts = Table(
    "alerts", metadata,
    Column("id", Integer, primary_key=True),
    Column("time", String(32), nullable=False),          # when the detector raised it (ISO, detector clock)
    Column("type", String(32), nullable=False),          # fight | theft | loitering | restricted | ...
    Column("message", Text, nullable=False),
    Column("track_ids", JsonType, nullable=False, default=list),
    Column("snapshot", Text),
    Column("clip", Text),
    Column("score", Float),                              # model probability, when the activity is a model
    Column("model_version", String(64)),                 # which model version raised it (lineage)
    Column("rover_id", String(64)),
    Column("evidence", Text),                            # stream alerts: stream://<camera>/<session>/<frame> (the
                                                         # skeleton frames in the lake; a label turns them into data)
    Column("label", String(16)),                         # human feedback: real | false_alarm (feeds retraining)
    Column("received", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Index("alerts_time", "time"),
    Index("alerts_type", "type"),
)

alert_media = Table(
    # photo + clip that the laptop saved for an alert the STREAM raises (it only has skeletons, no pictures).
    # Matched to that alert by camera, session and nearest frame, whichever of the two arrives first.
    "alert_media", metadata,
    Column("id", Integer, primary_key=True),
    Column("type", String(32), nullable=False),
    Column("camera", String(64), nullable=False),
    Column("session", String(64), nullable=False),
    Column("frame", Integer, nullable=False),
    Column("track_ids", JsonType, nullable=False, default=list),
    Column("snapshot", Text),
    Column("clip", Text),
    Column("alert_id", Integer),                         # the stream alert it was attached to (null: not yet)
    Column("received", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Index("alert_media_session", "camera", "session"),
)

telemetry = Table(
    "telemetry", metadata,
    Column("id", Integer, primary_key=True),
    Column("time", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("data", JsonType, nullable=False),
)

commands = Table(
    "commands", metadata,
    Column("id", Integer, primary_key=True),
    Column("time", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("data", JsonType, nullable=False),
    Column("username", String(64)),
    Column("delivered", Boolean, nullable=False, default=False),
)


def database_url():
    return os.environ.get("DATABASE_URL") or DEFAULT_URL


def make_engine(url=None):
    url = url or database_url()
    kwargs = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_engine(url, **kwargs)
    metadata.create_all(engine)
    migrate(engine)
    return engine


def migrate(engine):
    """Columns added after a table was first created (create_all only creates missing tables)."""
    from sqlalchemy import inspect, text
    have = {c["name"] for c in inspect(engine).get_columns("alerts")}
    with engine.begin() as conn:
        if "evidence" not in have:
            conn.execute(text("ALTER TABLE alerts ADD COLUMN evidence TEXT"))
