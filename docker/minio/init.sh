#!/bin/sh
# Creates the data lake layout in MinIO. Safe to run again (idempotent).
set -e
mc alias set lake http://minio:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD"

# Data lake zones (medallion layout)
#   raw        footage and records exactly as received from the rover, partitioned rover/yyyy/mm/dd
#   validated  records that passed the data-quality checks (Parquet)
#   rejected   records that failed them, with the reason (kept for debugging)
#   labelled   human-verified clips / windows (alert labels, annotation tool)
#   processed  training-ready datasets for one pipeline run (train/val/test splits)
# Tool buckets
#   datasets   public datasets (RWF-2000 archive / extracted pose)
#   mlflow     MLflow run artifacts and registered models
#   dvc        DVC remote (versioned data and model files)
for b in raw validated rejected labelled processed datasets mlflow dvc; do
  mc mb --ignore-existing "lake/$b"
done

# Versioning: every overwrite or delete of curated data can be rolled back
for b in labelled processed datasets mlflow dvc; do
  mc version enable "lake/$b"
done

# Retention (data minimisation): unflagged raw footage and rejected records expire automatically
# (rules are replaced on every run so re-running never stacks duplicates)
mc ilm rule rm --all --force lake/raw >/dev/null 2>&1 || true
mc ilm rule add --expire-days 30 lake/raw
mc ilm rule rm --all --force lake/rejected >/dev/null 2>&1 || true
mc ilm rule add --expire-days 14 lake/rejected

echo "Data lake ready:"
mc ls lake
