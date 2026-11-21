"""Pipeline stage "ingest": public dataset -> data lake (bucket "datasets"), verified and versioned.

    python training/ingest.py                 # settings from params.yaml, section "dataset"

1. If the lake already holds the archive with the expected SHA-256, nothing is downloaded (idempotent).
2. Otherwise the archive is downloaded to data/cache/ (resumable: an interrupted download continues),
   its SHA-256 is checked against params.yaml, and it is uploaded to the lake with its checksum and source
   as object metadata. The local copy is then deleted (the lake is the single source of truth).
3. A small manifest (data/manifests/<name>.json) records exactly which dataset version the pipeline used;
   it is tracked by git/DVC, so every model can be traced back to its source data.
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mlops import lake  # noqa: E402

CHUNK = 1 << 20


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(url, dst, expected_size=None):
    """Resumable HTTP download (Range requests)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    have = dst.stat().st_size if dst.exists() else 0
    if expected_size and have == expected_size:
        return
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=60) as resp:
        if have and resp.status != 206:            # server ignored the range: start again
            have = 0
        total = have + int(resp.headers.get("Content-Length", 0))
        t0, last = time.time(), 0.0
        with open(dst, "ab" if have else "wb") as f:
            done = have
            while block := resp.read(CHUNK):
                f.write(block)
                done += len(block)
                if time.time() - last > 15:
                    last = time.time()
                    rate = (done - have) / max(time.time() - t0, 1e-6) / 1e6
                    print(f"  {done / 1e9:.2f} / {total / 1e9:.2f} GB  ({rate:.1f} MB/s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="keep the local download in data/cache after uploading")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "params.yaml").read_text())["dataset"]
    name, bucket, key = cfg["name"], cfg["bucket"], cfg["key"]
    manifest_path = ROOT / "data" / "manifests" / f"{name}.json"

    lake.ping()
    meta = lake.head(bucket, key)
    if meta and meta.get("Metadata", {}).get("sha256") == cfg["sha256"]:
        print(f"{name}: already in the lake at s3://{bucket}/{key} (sha256 verified at ingestion), nothing to do")
    else:
        local = ROOT / "data" / "cache" / Path(key).name
        print(f"{name}: downloading {cfg['url']}\n  -> {local}")
        download(cfg["url"], local, cfg.get("size"))
        print("  checking SHA-256 ...", flush=True)
        digest = sha256_file(local)
        if digest != cfg["sha256"]:
            sys.exit(f"SHA-256 mismatch: got {digest}, expected {cfg['sha256']}. Delete {local} and run again.")
        print(f"  uploading to s3://{bucket}/{key} ...", flush=True)
        lake.upload_file(local, bucket, key, metadata={
            "sha256": digest, "source": cfg["url"], "ingested": time.strftime("%Y-%m-%dT%H:%M:%S")})
        if not args.keep:
            local.unlink()
        meta = lake.head(bucket, key)

    manifest = {"name": name, "uri": f"s3://{bucket}/{key}", "sha256": cfg["sha256"],
                "size_bytes": meta["ContentLength"], "source": cfg["url"], "license": cfg.get("license")}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest written: {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
