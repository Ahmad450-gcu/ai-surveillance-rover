"""Back up / restore the stack's data (Docker volumes), e.g. to move everything to another laptop.

    python tools/volumes.py backup  E:\\rover-backup          stops the stack, packs the volumes into .tgz files
    python tools/volumes.py restore E:\\rover-backup          on the new machine, BEFORE the first docker compose up
    python tools/volumes.py list                              volumes and their sizes

What is in the volumes: PostgreSQL (users, alerts with labels, predictions, MLflow runs and registry, Airflow
history) and MinIO (the 12 GB dataset, skeletons, stream frames, model files, DVC data versions). Both are needed;
--all also copies Grafana, Prometheus, Kafka and the Spark checkpoints (history only, not needed to run).
A restore never overwrites a volume that already holds data unless --force is given.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "rover"                                      # compose project name (docker-compose.yml: name)
ESSENTIAL = ["pgdata", "miniodata"]
EXTRA = ["grafanadata", "prometheusdata", "kafkadata", "sparkcheckpoints", "airflowlogs"]
IMAGE = "alpine:3.20"


def sh(*cmd, capture=False, check=True):
    r = subprocess.run(list(cmd), cwd=ROOT, text=True, capture_output=capture)
    if check and r.returncode:
        sys.exit(f"failed: {' '.join(cmd)}\n{r.stderr if capture else ''}")
    return r


def volume(name):
    return f"{PROJECT}_{name}"


def exists(name):
    return sh("docker", "volume", "inspect", volume(name), capture=True, check=False).returncode == 0


def is_empty(name):
    r = sh("docker", "run", "--rm", "-v", f"{volume(name)}:/v:ro", IMAGE, "sh", "-c", "ls -A /v | head -1",
           capture=True)
    return not r.stdout.strip()


def backup(folder, names):
    folder.mkdir(parents=True, exist_ok=True)
    print("stopping the stack (data stays in the volumes) ...")
    sh("docker", "compose", "--profile", "streaming", "--profile", "monitoring", "--profile", "mlops", "stop")
    for n in names:
        if not exists(n):
            print(f"  skip {volume(n)} (does not exist)")
            continue
        print(f"  {volume(n)} -> {n}.tgz")
        sh("docker", "run", "--rm", "-v", f"{volume(n)}:/v:ro", "-v", f"{folder.resolve()}:/b", IMAGE,
           "tar", "czf", f"/b/{n}.tgz", "-C", "/v", ".")
    print(f"done: {folder}. Copy this folder and your .env to the other machine; there: "
          f"python tools/volumes.py restore <folder>, then docker compose up -d")


def restore(folder, force):
    files = sorted(folder.glob("*.tgz"))
    if not files:
        sys.exit(f"no .tgz files in {folder}")
    for f in files:
        n = f.stem
        if exists(n) and not is_empty(n) and not force:
            print(f"  skip {volume(n)}: it already holds data (use --force to overwrite it)")
            continue
        # with Compose's labels, so `docker compose up` treats it as its own volume (no "not created by Compose" warning)
        sh("docker", "volume", "create", "--label", f"com.docker.compose.project={PROJECT}",
           "--label", f"com.docker.compose.volume={n}", volume(n), capture=True)
        print(f"  {f.name} -> {volume(n)}")
        sh("docker", "run", "--rm", "-v", f"{volume(n)}:/v", "-v", f"{folder.resolve()}:/b:ro", IMAGE, "sh", "-c",
           f"rm -rf /v/..?* /v/.[!.]* /v/* && tar xzf /b/{f.name} -C /v")
    print("done: now  docker compose up -d  (and dvc pull, to fill the project's local data folders)")


def list_volumes():
    r = sh("docker", "system", "df", "-v", "--format", "{{json .Volumes}}", capture=True)
    import json
    for v in json.loads(r.stdout or "[]"):
        if v["Name"].startswith(f"{PROJECT}_"):
            print(f"{v['Name']:<28} {v['Size']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["backup", "restore", "list"])
    ap.add_argument("folder", nargs="?", type=Path)
    ap.add_argument("--all", action="store_true", help="also history volumes (Grafana, Prometheus, Kafka, ...)")
    ap.add_argument("--force", action="store_true", help="restore over volumes that already hold data")
    args = ap.parse_args(argv)
    if args.command == "list":
        return list_volumes()
    if not args.folder:
        ap.error("a folder is needed")
    if args.command == "backup":
        backup(args.folder, ESSENTIAL + (EXTRA if args.all else []))
    else:
        restore(args.folder, args.force)


if __name__ == "__main__":
    sys.exit(main())
