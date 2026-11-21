"""Point DVC at the data lake (MinIO bucket "dvc") using the credentials in .env.

    python tools/configure_dvc.py

The remote URL and endpoint go into .dvc/config (committed); the keys go into .dvc/config.local,
which DVC never commits. Re-run after changing .env (e.g. when moving to a cloud VM / AWS S3).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mlops.lake import settings  # noqa: E402


def dvc(*args):
    exe = Path(sys.executable).with_name("dvc.exe" if sys.platform == "win32" else "dvc")
    subprocess.run([str(exe) if exe.exists() else "dvc", *args], cwd=ROOT, check=True)


def main():
    s = settings()
    if not s["aws_access_key_id"]:
        sys.exit("No lake credentials: set MINIO_ROOT_USER / MINIO_ROOT_PASSWORD (or LAKE_ACCESS_KEY / LAKE_SECRET_KEY) in .env")
    dvc("remote", "add", "--default", "--force", "lake", "s3://dvc")
    dvc("remote", "modify", "lake", "endpointurl", s["endpoint_url"])
    dvc("remote", "modify", "--local", "lake", "access_key_id", s["aws_access_key_id"])
    dvc("remote", "modify", "--local", "lake", "secret_access_key", s["aws_secret_access_key"])
    print(f"DVC remote 'lake' -> s3://dvc at {s['endpoint_url']}")


if __name__ == "__main__":
    main()
