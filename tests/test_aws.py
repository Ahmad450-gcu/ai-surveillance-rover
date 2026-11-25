"""The AWS tool and the files it depends on (no AWS account needed)."""
import re
from pathlib import Path

import pytest
import yaml

from tools import aws

ROOT = Path(__file__).resolve().parents[1]


def test_the_server_gets_its_own_passwords():
    local = {"POSTGRES_USER": "rover", "POSTGRES_PASSWORD": "easy", "GRAFANA_ADMIN_PASSWORD": "easy",
             "ROVER__BACKEND__DEVICE_TOKEN": "laptop", "MINIO_ROOT_USER": "rover-admin"}
    env = aws.server_env(local)
    assert "easy" not in env.values() and "laptop" not in env.values()
    secret = [v for k, v in env.items() if re.search("PASSWORD|TOKEN|SECRET|FERNET", k)]
    assert len(set(secret)) == len(secret) and all(len(v) >= 32 for v in secret)
    assert all(re.fullmatch(r"[A-Za-z0-9_=-]+", v) for v in secret)        # safe in URLs, in .env and in the shell
    assert aws.server_env(local)["POSTGRES_PASSWORD"] != env["POSTGRES_PASSWORD"]
    assert env["AIRFLOW_UID"] == "1000" and env["COMPOSE_PROFILES"] == "streaming,monitoring,mlops"


def test_every_setting_compose_requires_is_in_the_server_env():
    required = set(re.findall(r"\$\{([A-Z_]+):\?", (ROOT / "docker-compose.yml").read_text(encoding="utf-8")))
    assert required and required <= set(aws.server_env({}))


def test_tunnel_ports_are_the_published_ones():
    services = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))["services"]
    published = {int(re.search(r":-(\d+)\}", p)[1]) for s in services.values() for p in s.get("ports", [])}
    assert set(aws.PORTS.values()) <= published
    assert aws.LIVE_VIEW_PORT not in aws.PORTS.values()
    assert set(aws.DB_USERS.split()) <= set(services) and "postgres" not in aws.DB_USERS and "minio" not in aws.DB_USERS
    opts = aws.tunnel_options(["MinIO API"], offset=10000)
    assert opts[-1] == "127.0.0.1:19000:127.0.0.1:9000"                   # only this computer can use the tunnel


def test_settings_come_from_tfvars_or_the_defaults(tmp_path, monkeypatch):
    assert aws.tf_setting("region") == "ap-southeast-2" and aws.tf_setting("instance_type") == "m7i-flex.large"
    (tmp_path / "variables.tf").write_text((aws.TF / "variables.tf").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "terraform.tfvars").write_text('stop_at       = "22:00"\n')
    monkeypatch.setattr(aws, "TF", tmp_path)
    assert aws.tf_setting("stop_at") == "22:00" and aws.tf_setting("region") == "ap-southeast-2"


def test_prepare_keeps_the_passwords_and_the_other_logins(tmp_path, monkeypatch):
    key = tmp_path / "rover_aws"
    key.write_text("x")
    monkeypatch.setattr(aws, "KEY", key)
    monkeypatch.setattr(aws, "ROOT", tmp_path)
    monkeypatch.setattr(aws, "ENV_FILE", tmp_path / "server.env")
    monkeypatch.setattr(aws, "LOGINS", tmp_path / "MY_LOGINS.txt")
    (tmp_path / "MY_LOGINS.txt").write_text("my laptop logins\n", encoding="utf-8")
    aws.prepare([])
    first = (tmp_path / "server.env").read_text()
    aws.prepare([])
    logins = (tmp_path / "MY_LOGINS.txt").read_text(encoding="utf-8")
    assert (tmp_path / "server.env").read_text() == first
    assert logins.startswith("my laptop logins") and logins.count("Grafana") == 1
    assert aws.read_env(tmp_path / "server.env")["DASHBOARD_PASSWORD"] in logins


def test_a_server_made_by_hand(tmp_path, monkeypatch):
    monkeypatch.setattr(aws, "USER_DATA", tmp_path / "user-data.yaml")
    monkeypatch.setattr(aws, "MANUAL", tmp_path / "server.json")
    monkeypatch.setattr(aws, "ROOT", tmp_path)
    aws.user_data([])
    text = (tmp_path / "user-data.yaml").read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    assert "${" not in text and doc["timezone"] == "Asia/Karachi" and "fallocate -l 4G" in text and "23:30" in text
    with pytest.raises(SystemExit):
        aws.use(["not-an-ip"])
    aws.use(["13.54.1.2"])
    assert aws.outputs()["public_ip"] == "13.54.1.2"
    with pytest.raises(SystemExit):                                       # no instance id: switched in the console
        aws.start([])


def test_the_detector_settings_exist_in_the_config():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    for name in aws.run_environment({"ROVER__BACKEND__DEVICE_TOKEN": "t"}):
        if name.startswith("ROVER__"):
            section, key = name[len("ROVER__"):].lower().split("__")
            assert key in cfg[section]


def test_cloud_init_template_is_valid_yaml():
    text = (ROOT / "deploy/aws/cloud-init.yaml").read_text(encoding="utf-8")
    names = set(re.findall(r"\$\{(\w+)\}", text))
    passed = set(re.findall(r"^    (\w+)\s+= ", (ROOT / "deploy/aws/main.tf").read_text(encoding="utf-8"), re.M))
    assert names <= passed
    doc = yaml.safe_load(re.sub(r"\$\{\w+\}", "1", text))
    assert any("rover-stop.timer" in str(c) for c in doc["runcmd"]) and len(doc["write_files"]) == 2
