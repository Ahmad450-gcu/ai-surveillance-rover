"""Environment-variable overrides for config.yaml (rover_ai/settings.py)."""
from rover_ai.settings import apply_env_overrides, load_config


def test_env_overrides_nested_values():
    cfg = {"backend": {"url": None, "device_token": "dev"}, "loitering": {"seconds": 20}}
    env = {"ROVER__BACKEND__URL": "http://backend:8080", "ROVER__LOITERING__SECONDS": "8",
           "ROVER__NEW__FLAG": "true", "UNRELATED": "x"}
    apply_env_overrides(cfg, env)
    assert cfg["backend"]["url"] == "http://backend:8080"
    assert cfg["loitering"]["seconds"] == 8
    assert cfg["new"]["flag"] is True
    assert cfg["backend"]["device_token"] == "dev"


def test_repo_config_loads():
    cfg = load_config()
    assert {"models", "fight", "alerts", "backend"} <= cfg.keys()
