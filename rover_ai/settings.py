"""Configuration loading: config.yaml for defaults, environment variables for anything deployment-specific.

Any config value can be overridden with an environment variable named ROVER__<SECTION>__<KEY>, e.g.

    ROVER__BACKEND__DEVICE_TOKEN=...      -> cfg["backend"]["device_token"]
    ROVER__BACKEND__URL=http://backend:8080

Values are parsed as YAML, so "true", "12" and "null" become bool / int / None. This keeps secrets and
host names out of git (they live in .env, see .env.example) and lets the same image run locally, in
Docker Compose, or on a cloud VM without code changes.
"""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ENV_PREFIX = "ROVER__"


def load_dotenv(path=ROOT / ".env"):
    """Minimal .env reader (KEY=VALUE lines). Real environment variables always win."""
    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def apply_env_overrides(cfg, environ=None):
    environ = os.environ if environ is None else environ
    for name, raw in environ.items():
        if not name.startswith(ENV_PREFIX):
            continue
        *parents, leaf = name[len(ENV_PREFIX):].lower().split("__")
        node = cfg
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw) if raw != "" else ""
    return cfg


def load_config(path=ROOT / "config.yaml"):
    load_dotenv()
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return apply_env_overrides(cfg)
