# Demo runbook: the whole pipeline in four acts

One script, [`tools/demo.py`](../tools/demo.py), starts what each act needs and prints what to open and what to say.
On an 8 GB laptop it stops the parts an act doesn't need (Docker gets less than 6 GB); on a 16 GB laptop everything
keeps running (`--keep` forces that anywhere).

| Act | Command | Shows | Time |
|---|---|---|---|
| 0 | `python tools/demo.py check` | Docker, `.env`, demo data, which services answer | 1 min |
| 1 | `python tools/demo.py act1` | real-time path: Kafka → Spark validation + dead-letter queue → lake → serving → alert | 5 min |
| 2 | `python tools/demo.py act2` | monitoring and data drift: a normal and a "dark" camera; drift → retraining request | 5 min |
| 3 | `python tools/demo.py act3` | retraining: Airflow run, MLflow registry, model rollback and back | 5 min |
| 4 | `python tools/demo.py act4` | delivery and extensibility: activity registry, smoke tests, CI/CD runs, published images | 3 min |

`python tools/demo.py urls` prints every address and login; `python tools/demo.py stop` stops everything (data kept).

## Before the day

1. Docker Desktop running; `venv\Scripts\activate`.
2. `python tools/demo.py check`; if the drift stream is missing: `python tools/demo.py prepare`.
3. `docker compose exec backend python -m backend.manage create-user admin` if there is no dashboard login yet.
4. Run all four acts once, in order, and record the screen (the backup if anything fails on the day).
5. Open browser tabs: dashboard, MLflow, Grafana, Airflow, GitHub Actions.

## Act 1: real-time path

`act1` starts core + streaming + monitoring, runs the end-to-end stream check (12 invalid records → dead-letter queue
with the right reason; the fight clip → judgements → 1 alert stored by the backend) and replays the fight clip as
camera `demo-1`.

Show: Spark UI (rates), the `[stream]` FIGHT alert in the dashboard (label it "Yes, real"), MinIO `rejected/` with
the reasons, Grafana *System health* (data quality, rejected by reason, event lag).

## Act 2: monitoring and data drift

`act2` replays 100 unseen validation clips twice: as `cam-normal` and as `cam-dark` (keypoint confidences × 0.55,
like night or a dirty lens), then prints the drift monitor's verdict per camera.

Show: the drift report, Grafana *Model & drift* (PSI per feature, drift status, one row per activity),
Alertmanager (`DataDriftDetected` after 2 minutes), the new retraining request.

## Act 3: retraining and the registry

`act3` starts Airflow (stopping streaming + monitoring on a small machine), triggers `fight_retraining` with
`{"force": true}` and lists the registry.

Show: the run starting (about 15 minutes in total; don't wait), then the **finished run of 22 Nov**: every task green,
`promote` rejected the challenger (F1 0.6706 vs 0.6744 on the same 400 clips) and the log says why. MLflow: v4
`@champion`, v5 `@challenger`. Live rollback: `bash deploy/model.sh rollback 1`, then `bash deploy/model.sh rollback 4`.

## Act 4: delivery and extensibility

`act4` lists the activity registry, runs its consistency check and the deployment smoke tests.

Show: GitHub Actions (CI → CD), the published images, `docs/DEPLOYMENT.md` (automatic rollback, final deployment),
and `activities.yaml`: a fifth activity (crowding) was added with one entry and one plugin class:

```powershell
python run.py --source tests/videos/walking.avi --set crowding.enabled=true --set crowding.max_people=1 --set crowding.seconds=3
```

## Moving to another laptop

```powershell
python tools/volumes.py backup E:\rover-backup      # this laptop: stops the stack, packs PostgreSQL + MinIO
python tools/volumes.py restore E:\rover-backup     # other laptop, before the first docker compose up
```

Copy `.env` by hand (it is not in Git). Images: `IMAGE_PREFIX=ghcr.io/ahmad450-gcu/rover` in `.env`, then
`docker compose pull`. `--all` also copies the history volumes (Grafana, Prometheus, Kafka).
