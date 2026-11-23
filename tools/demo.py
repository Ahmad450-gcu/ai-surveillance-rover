"""Demo kit: show the whole pipeline in four acts, each starting the parts of the stack it needs (and, on a machine
with little memory, stopping the ones it doesn't). Run the acts in order; each prints what to open and what to say.

    python tools/demo.py check       Docker, .env, demo data, which services answer
    python tools/demo.py prepare     builds the drift demo stream (data/streams/val100.jsonl.gz) if it is missing
    python tools/demo.py act1        real-time path: Kafka -> Spark (validation, dead-letter queue) -> serving -> alert
    python tools/demo.py act2        monitoring + data drift: a normal and a "dark" camera; drift -> retraining request
    python tools/demo.py act3        retraining: Airflow run, MLflow registry, model rollback and back
    python tools/demo.py act4        delivery: smoke tests of the running release, links to CI/CD and the images
    python tools/demo.py urls        every address and login
    python tools/demo.py stop        stop everything (data is kept)

Memory: with less than 6 GB for Docker (an 8 GB laptop), act3 stops streaming + monitoring before starting Airflow
and act1 stops Airflow. A 16 GB laptop keeps everything running. Override: --keep (never stop anything).
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PY = sys.executable
STREAMING = ["kafka", "spark", "alert-bridge"]
MONITORING = ["prometheus", "grafana", "alertmanager", "kafka-exporter", "drift-monitor"]
MLOPS = ["airflow-scheduler", "airflow-webserver"]
VAL100 = ROOT / "data" / "streams" / "val100.jsonl.gz"
FIGHT = ROOT / "tests" / "data" / "fight_frames.jsonl.gz"


def env():
    from rover_ai.settings import load_dotenv
    load_dotenv(ROOT / ".env")
    return os.environ


def port(name, default):
    return os.environ.get(name) or str(default)


def urls():
    e = env()
    return [
        ("Dashboard (alerts, labels)", f"http://localhost:{port('BACKEND_PORT', 8080)}", "user from backend.manage create-user"),
        ("MLflow (runs, registry)", f"http://localhost:{port('MLFLOW_PORT', 5000)}", "-"),
        ("Serving API", f"http://127.0.0.1:{port('SERVING_PORT', 8000)}/v1/models", "-"),
        ("MinIO (data lake)", f"http://localhost:{port('MINIO_CONSOLE_PORT', 9001)}", f"{e.get('MINIO_ROOT_USER', 'MINIO_ROOT_USER')} / MINIO_ROOT_PASSWORD"),
        ("Spark (streaming)", f"http://localhost:{port('SPARK_UI_PORT', 4040)}", "-"),
        ("Prometheus", f"http://localhost:{port('PROMETHEUS_PORT', 9090)}", "-"),
        ("Grafana", f"http://localhost:{port('GRAFANA_PORT', 3000)}", "admin / GRAFANA_ADMIN_PASSWORD"),
        ("Alertmanager", f"http://localhost:{port('ALERTMANAGER_PORT', 9093)}", "-"),
        ("Drift report", f"http://127.0.0.1:{port('DRIFT_MONITOR_PORT', 9200)}/report", "-"),
        ("Airflow (retraining)", f"http://localhost:{port('AIRFLOW_PORT', 8081)}", "admin / AIRFLOW_ADMIN_PASSWORD"),
        ("GitHub Actions (CI/CD)", "https://github.com/Ahmad450-gcu/ai-surveillance-rover/actions", "-"),
    ]


def say(title, lines=()):
    print(f"\n=== {title}")
    for line in lines:
        print("   ", line)


def run(*cmd, check=True, quiet=False):
    print("$", " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, text=True,
                       stdout=subprocess.DEVNULL if quiet else None, stderr=subprocess.STDOUT if quiet else None)
    if check and r.returncode:
        sys.exit(f"failed ({r.returncode}): {' '.join(str(c) for c in cmd)}")
    return r.returncode


def compose(*args, check=True):
    return run("docker", "compose", *args, check=check)


def docker_memory_gb():
    try:
        out = subprocess.run(["docker", "info", "--format", "{{.MemTotal}}"], capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip()) / 2**30
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 0.0


def small_machine(args):
    return not args.keep and docker_memory_gb() < 6


def up(profiles, services=()):
    flags = [x for p in profiles for x in ("--profile", p)]
    compose(*flags, "up", "-d", "--wait", *services)


def get(url, timeout=5, auth=None):
    req = urllib.request.Request(url, headers={"Authorization": "Basic " + base64.b64encode(auth.encode()).decode()}
                                 if auth else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:            # noqa: S310 - local demo addresses
        return json.loads(r.read() or b"null")


def answers(url):
    try:
        urllib.request.urlopen(url, timeout=3)                         # noqa: S310
        return True
    except urllib.error.HTTPError:
        return True                                                    # it answered (e.g. 401 / 404)
    except OSError:
        return False


# ------------------------------------------------------------------------------------------------------- commands
def check(args):
    ok = True
    if run("docker", "info", check=False, quiet=True):
        print("Docker: NOT running -> start Docker Desktop and wait for 'Engine running'")
        ok = False
    else:
        print(f"Docker: running, {docker_memory_gb():.1f} GB memory")
    print(".env:", "found" if (ROOT / ".env").exists() else "MISSING -> copy .env.example .env and set the passwords")
    print("fight replay data:", "ok" if FIGHT.exists() else "missing (tests/data/fight_frames.jsonl.gz)")
    print("drift demo stream:", "ok" if VAL100.exists() else "missing -> python tools/demo.py prepare")
    say("Services answering now")
    for name, url, _ in urls()[:-1]:
        print(f"    {'up  ' if answers(url) else 'down'}  {name:<28} {url}")
    return 0 if ok else 1


def prepare(args):
    if VAL100.exists():
        print("drift demo stream already there:", VAL100.relative_to(ROOT))
        return 0
    src = ROOT / "data" / "validated" / "pose" / "rwf2000" / "val"
    if not src.exists():
        sys.exit("data/validated/pose/rwf2000 is missing: run  dvc pull  first (needs the data lake running)")
    run(PY, "tools/frames.py", "from-clips", src, VAL100, "--clips", "100")
    return 0


def act1(args):
    say("Act 1: real-time path", ["Kafka -> Spark validation + dead-letter queue -> lake -> serving -> PostgreSQL -> alert"])
    if small_machine(args):
        compose("stop", *MLOPS, check=False)
    up(["streaming", "monitoring"])
    run(PY, "tools/stream_smoke_test.py")
    run(PY, "tools/frames.py", "replay", FIGHT, "--camera", "demo-1")
    say("Show now", [
        "Spark UI http://localhost:4040 -> Structured Streaming: input and processing rates",
        "Dashboard http://localhost:8080 -> the FIGHT alert ending in [stream]; label it 'Yes, real'",
        "MinIO http://localhost:9001 -> bucket rejected/: the 12 invalid test records, each with its reason",
        "Grafana -> Rover · System health: data quality, rejected by reason, event lag",
        "Say: 'Bad data never reaches the model; it goes to a dead-letter queue with the reason.'"])
    return 0


def act2(args):
    say("Act 2: monitoring and data drift", ["a camera like the training data vs a 'dark' camera"])
    if not VAL100.exists():
        prepare(args)
    up(["streaming", "monitoring"])
    run(PY, "tools/frames.py", "replay", VAL100, "--speed", "0", "--camera", "cam-normal")
    run(PY, "tools/frames.py", "replay", VAL100, "--speed", "0", "--camera", "cam-dark", "--drift", "dark")
    print("waiting for the drift check (runs every few minutes) ...")
    for _ in range(60):
        try:
            report = get(f"http://127.0.0.1:{port('DRIFT_MONITOR_PORT', 9200)}/report") or {}
            cams, drifted = report.get("cameras", {}), report.get("drifted", {})
            if "cam-dark" in cams and "cam-normal" in cams:
                for cam in ("cam-normal", "cam-dark"):
                    feats = {f: d["psi"] for f, d in cams[cam]["features"].items() if d.get("psi") is not None}
                    worst = max(feats, key=feats.get, default=None)
                    state = f"DRIFT in {', '.join(drifted[cam])}" if cam in drifted else "stable"
                    print(f"    {cam}: {state}" + (f" (highest PSI: {worst} {feats[worst]:.2f})" if worst else ""))
                break
        except (OSError, ValueError):
            pass
        time.sleep(10)
    say("Show now", [
        "Drift report http://127.0.0.1:9200/report (cam-normal stable, cam-dark drifted)",
        "Grafana -> Rover · Model & drift: PSI per feature, drift status per camera, one row per activity",
        "Alertmanager http://localhost:9093: DataDriftDetected (fires after 2 minutes of drift)",
        "Grafana -> 'Retraining requests': a new request for the fight model (Airflow starts it if running)",
        "Say: 'Nobody changed the code, but the camera data changed: the system notices and asks to retrain.'"])
    return 0


def act3(args):
    say("Act 3: retraining and the model registry")
    if small_machine(args):
        print("less than 6 GB for Docker: stopping streaming + monitoring so training has room")
        compose("stop", *STREAMING, *MONITORING, check=False)
    up(["mlops"])
    e = env()
    try:
        run_info = get_post_dag(e)
        print(f"    started fight_retraining run {run_info.get('dag_run_id')}")
    except OSError as err:
        print("    could not start the DAG through the API:", err, "-> trigger it in the Airflow UI")
    run(PY, "-m", "mlops.registry", "list", check=False)
    say("Show now", [
        "Airflow http://localhost:8081 -> fight_retraining -> Grid: the new run starting (takes ~15 min);",
        "    open the finished run of 22 Nov: every task green, promote = rejected, and its log says why",
        "MLflow http://localhost:5000 -> Models -> fight-detector: v4 @champion, v5 @challenger",
        "Rollback live:  bash deploy/model.sh rollback 1   then   bash deploy/model.sh rollback 4",
        "Say: 'A new model replaces the old one only if it is better on the same test clips; rollback takes 20 s.'"])
    return 0


def get_post_dag(e):
    body = json.dumps({"conf": {"force": True}}).encode()
    req = urllib.request.Request(f"http://localhost:{port('AIRFLOW_PORT', 8081)}/api/v1/dags/fight_retraining/dagRuns",
                                 data=body, method="POST", headers={
                                     "Content-Type": "application/json",
                                     "Authorization": "Basic " + base64.b64encode(
                                         f"admin:{e.get('AIRFLOW_ADMIN_PASSWORD', '')}".encode()).decode()})
    with urllib.request.urlopen(req, timeout=20) as r:                 # noqa: S310
        return json.loads(r.read())


def act4(args):
    say("Act 4: delivery (CI/CD) and the activity registry")
    run(PY, "-m", "rover_ai.activities", "list")
    run(PY, "-m", "rover_ai.activities", "check", check=False)
    if run("bash", "deploy/smoke.sh", check=False):
        print("    (smoke tests need Git Bash and the core stack running)")
    say("Show now", [
        "GitHub -> Actions: CI (lint, tests, image builds, DAG test, shellcheck, Terraform) then CD (5 images, deploy)",
        "GitHub -> Packages: rover-* images tagged sha-<commit>",
        "docs/DEPLOYMENT.md: automatic rollback tests and the final deployment",
        "activities.yaml: a new activity = one entry + one plugin class (crowding was added this way):",
        "    python run.py --source tests/videos/walking.avi --set crowding.enabled=true --set crowding.max_people=1 "
        "--set crowding.seconds=3   (alert at 60 s)",
        "Say: 'Every push is tested, packaged and deployed with health checks and automatic rollback.'"])
    return 0


def show_urls(args):
    for name, url, login in urls():
        print(f"{name:<28} {url:<62} {login}")
    return 0


def stop(args):
    return compose("--profile", "streaming", "--profile", "monitoring", "--profile", "mlops", "stop", check=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "prepare", "act1", "act2", "act3", "act4", "urls", "stop"])
    ap.add_argument("--keep", action="store_true", help="never stop services to save memory")
    args = ap.parse_args(argv)
    env()
    return {"check": check, "prepare": prepare, "act1": act1, "act2": act2, "act3": act3, "act4": act4,
            "urls": show_urls, "stop": stop}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
