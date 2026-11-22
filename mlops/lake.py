"""Data lake client: MinIO (S3 API). The same code works against AWS S3 by changing the endpoint in .env.

Buckets (created by docker/minio/init.sh):
    datasets   public source datasets, immutable (e.g. datasets/rwf2000/RWF-2000.tar.gz)
    raw        data as produced, before any check (rover records, extracted pose)
    validated  data that passed the quality checks
    rejected   data that failed them, next to a .reason.json explaining why
    labelled   human-verified data (alert labels, annotation tool)
    processed  training-ready datasets (splits) for one pipeline run
    mlflow     MLflow artifacts          dvc   DVC remote

Settings (.env):  LAKE_ENDPOINT_URL (default http://localhost:9000), LAKE_ACCESS_KEY / LAKE_SECRET_KEY
                  (default: MINIO_ROOT_USER / MINIO_ROOT_PASSWORD).
"""
import hashlib
import json
import os
import time
from functools import lru_cache
from pathlib import Path

from rover_ai.settings import load_dotenv

BUCKETS = ("datasets", "raw", "validated", "rejected", "labelled", "processed", "mlflow", "dvc")


def settings():
    load_dotenv()
    return {
        "endpoint_url": os.environ.get("LAKE_ENDPOINT_URL", "http://localhost:9000"),
        "aws_access_key_id": os.environ.get("LAKE_ACCESS_KEY") or os.environ.get("MINIO_ROOT_USER"),
        "aws_secret_access_key": os.environ.get("LAKE_SECRET_KEY") or os.environ.get("MINIO_ROOT_PASSWORD"),
    }


@lru_cache(maxsize=1)
def client():
    import boto3
    from botocore.config import Config
    return boto3.client("s3", **settings(), region_name="us-east-1",
                        config=Config(retries={"max_attempts": 5, "mode": "standard"}, s3={"addressing_style": "path"}))


def head(bucket, key):
    """Object metadata, or None if it does not exist."""
    from botocore.exceptions import ClientError
    try:
        return client().head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def upload_file(path, bucket, key, metadata=None, attempts=3):
    """Upload a file (multipart above 64 MB, 2 parts at a time so a memory-capped MinIO keeps up); retried."""
    from boto3.s3.transfer import TransferConfig
    cfg = TransferConfig(multipart_threshold=64 * 1024 * 1024, multipart_chunksize=64 * 1024 * 1024, max_concurrency=2)
    for attempt in range(1, attempts + 1):
        try:
            client().upload_file(str(path), bucket, key, ExtraArgs={"Metadata": metadata or {}}, Config=cfg)
            return
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(5 * attempt)


def download_file(bucket, key, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    client().download_file(bucket, key, str(path))


def open_stream(bucket, key):
    """A file-like stream of an object (read sequentially, e.g. with tarfile mode 'r|gz')."""
    return client().get_object(Bucket=bucket, Key=key)["Body"]


def put_json(bucket, key, obj):
    client().put_object(Bucket=bucket, Key=key, Body=json.dumps(obj, indent=2).encode(), ContentType="application/json")


def get_json(bucket, key):
    return json.loads(client().get_object(Bucket=bucket, Key=key)["Body"].read())


def list_keys(bucket, prefix=""):
    paginator = client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        yield from page.get("Contents", [])


def md5_file(path, chunk=1 << 20):
    h = hashlib.md5()  # noqa: S324 - matches the S3 ETag of single-part uploads, not used for security
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _stored_md5(bucket, key, etag):
    """Multipart uploads have an ETag that is not the file's MD5 ("<hash>-<parts>"): use the md5 stored as metadata."""
    if "-" not in etag:
        return None
    return client().head_object(Bucket=bucket, Key=key).get("Metadata", {}).get("md5")


def sync_dir(local_dir, bucket, prefix, pattern="**/*"):
    """Upload every file under local_dir whose content differs from the lake copy. Returns (uploaded, unchanged)."""
    local_dir = Path(local_dir)
    remote = {o["Key"]: o["ETag"].strip('"') for o in list_keys(bucket, prefix)}
    uploaded = unchanged = 0
    for path in sorted(p for p in local_dir.glob(pattern) if p.is_file()):
        key = f"{prefix.rstrip('/')}/{path.relative_to(local_dir).as_posix()}"
        md5 = md5_file(path)
        if key in remote and md5 in (remote[key], _stored_md5(bucket, key, remote[key])):
            unchanged += 1
            continue
        upload_file(path, bucket, key, metadata={"md5": md5})
        uploaded += 1
    return uploaded, unchanged


def ping():
    """True if the lake answers and every expected bucket exists."""
    names = {b["Name"] for b in client().list_buckets()["Buckets"]}
    missing = [b for b in BUCKETS if b not in names]
    if missing:
        raise RuntimeError(f"lake is up but buckets are missing: {missing} (run: docker compose run --rm minio-init)")
    return True
