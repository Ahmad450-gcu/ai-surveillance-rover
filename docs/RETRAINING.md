# Automated retraining (Phase 7)

The model is not trained once and left alone. Live data is watched, people review the alerts, and both feed back
into training. A new model replaces the one in service only if it is measurably better on the same test clips.

```mermaid
flowchart LR
  STREAM[Kafka + Spark<br/>stream alerts] --> BE[backend<br/>alerts table]
  BE -- "reviewer: real / false alarm" --> BE
  DRIFT[drift monitor<br/>PSI per camera] -- Alertmanager --> REQ[(retrain_requests)]
  REQ -- REST API --> AF
  CRON[daily 03:00] --> AF
  UI[Airflow UI: Trigger] --> AF
  subgraph AF["Airflow DAG fight_retraining"]
    D[decide] --> E[export_feedback] --> F[feedback<br/>DVC] --> FE[features<br/>DVC] --> T[train<br/>DVC] --> R[drift_reference<br/>DVC] --> P[promote]
    P --> V[verify_serving] --> PU[push_data] --> FI[finish]
  end
  BE -. labelled alerts .-> E
  LAKE[(MinIO lake<br/>stream frames)] -. skeletons .-> F
  T --> MLF[(MLflow registry<br/>@challenger)]
  P -- gate passed --> CH[(@champion)]
  CH -- hot reload --> SERVE[serving API]
```

## 1. Where the new training data comes from: the reviewers

Every alert raised on the stream stores where its frames are: `alerts.evidence = stream://<camera>/<session>/<frame>`
(the Spark job adds the session and frame; the alert bridge builds the reference). The skeletons of those frames are
already in the lake (`s3://validated/stream/pose/…`, Phase 5). When someone answers **"Was this correct?"** in the dashboard or
the app, the label turns that alert into a training clip, with no new video processing:

| Label | Becomes | Why it matters |
|---|---|---|
| `real` | `Fight` clip | more examples of the cases the model already catches |
| `false_alarm` | `NonFight` clip | exactly the frames that fooled the model, which is the most useful data there is |

`training/feedback.py`:

- `export`: labelled stream alerts (PostgreSQL) → `data/manifests/feedback_labels.json` (in git: which alerts and which labels built which model).
- `build` (DVC stage `feedback`): for each alert, the 4 s before and 1 s after the alert frame, resampled to 10 fps, in the same `.npz` format as the RWF-2000 clips (`data/validated/pose/rover/{train,val}/{Fight,NonFight}`). The same quality gate as `validate` applies (≥ 20 frames, ≥ 10 with two people). Copies go to the lake's labelled zone (`s3://labelled/fight/rover/`). Until there are 10 labelled clips all of them go to training; after that one in five (chosen deterministically by alert id) is kept for testing.

The `features` stage reads them next to RWF-2000 (`params.yaml` → `features.extra_data`), so the new data simply joins
the training set. The `drift_reference` stage also includes them, so the drift reference describes what the model was trained on.

## 2. When retraining runs

| Trigger | How |
|---|---|
| **Drift** | Alertmanager → drift monitor webhook → a row in `retrain_requests` → `POST /api/v1/dags/fight_retraining/dagRuns` (Airflow REST API, basic auth). If Airflow is down, the request stays `pending` and the next daily run picks it up |
| **New labels** | daily at 03:00: retrain if ≥ `RETRAIN_MIN_NEW_LABELS` (5) labelled alerts are not in the training data yet |
| **Age** | daily at 03:00: retrain if the last retraining is older than `RETRAIN_MAX_AGE_DAYS` (7) |
| **By hand** | Airflow UI → *Trigger DAG w/ config* `{"force": true}` |

If none applies, `decide` skips the rest of the run (grey in the UI), so the daily run costs nothing.

## 3. The DAG

| Task | What it does |
|---|---|
| `decide` | collects the reasons; marks pending retraining requests `running` |
| `export_feedback` | labelled alerts → manifest |
| `feedback`, `features`, `train`, `drift_reference` | `dvc repro --single-item <stage>`: DVC reruns a stage only if its inputs changed, and records the exact data, code and params in `dvc.lock` |
| `promote` | promotion gate (below); writes `reports/promotion.json` |
| `verify_serving` | if promoted: waits until the serving API reports the new version (it polls the registry every 60 s) |
| `push_data` | `dvc push`: the new data version goes to the lake (bucket `dvc`) |
| `finish` | closes the retraining requests with the decision; `reports/retraining_last.json` |

A failed task is retried once; if it fails again, its requests are marked `failed` (Grafana → *Retraining requests*). Only
one run at a time (`max_active_runs=1`). Training runs in its own virtualenv (`/opt/trainer`, CPU PyTorch) inside the
Airflow image, on the project folder mounted at `/opt/rover`.

## 4. Champion vs challenger: comparing on the same clips

Every training run registers a new version as `@challenger`. Before, the gate compared the challenger's test score with the
score the champion had on *its own* test set. Once new data joins the test split, those numbers come from different clips
and can't be compared. Now `train_fight.py` also downloads the current `@champion` and scores it on the challenger's
test clips with the champion's own alert rule (threshold, consecutive windows). Both numbers are logged to MLflow
(`test_f1`, `champion_test_f1`, tag `champion_version_compared`), and `promote` uses them:

- promote if the challenger's F1 is ≥ 0.01 higher **and** its ROC AUC is at most 0.02 lower (`params.yaml` → `registry`);
- otherwise the champion stays and the challenger keeps its version (it can be inspected or promoted by hand).

Rollback is one command, and the serving API follows within a minute:

```powershell
python -m mlops.registry set-alias champion 4
```

## 5. Running it

```powershell
docker compose --profile mlops up -d --build
```

Airflow: http://localhost:8081 (user `admin`, password `AIRFLOW_ADMIN_PASSWORD` in `.env`). Training needs about 2.5 GB of
Docker memory. With the default WSL limit on an 8 GB laptop, stop the other profiles while it runs:

```powershell
docker compose stop spark alert-bridge kafka prometheus grafana alertmanager kafka-exporter drift-monitor
```

After a run, `dvc.lock`, `data/manifests/feedback_labels.json` and `reports/` are changed in the project folder. Commit them:
they record which labelled alerts and which data produced the model now in service.

## 6. Results

First end-to-end run (22 Nov 2026, laptop, CPU only, streaming and monitoring stopped). Before it:

- the drift monitor had raised `DataDriftDetected` for camera `cam-dark` (TC-73), so retraining request #1 was `pending`;
- the walking and fight recordings were replayed as camera `rover-1`. The walking video raised its known false alarm (alert 93, score 0.96 under v4, frame 537 ≈ 54 s), the fight raised alert 94. A reviewer labelled them `false_alarm` and `real`.

Airflow started the run on its own (the daily schedule found the pending request): `decide` → *"retraining because: alert: DataDriftDetected"*.

| Task | Result | Time |
|---|---|---|
| `export_feedback` | 2 labelled stream alerts (1 real, 1 false alarm) → manifest | 8 s |
| `feedback` | 2 clips → `rover/train/Fight/alert-94.npz`, `rover/train/NonFight/alert-93.npz` (+ `s3://labelled/fight/rover/`) | 15 s |
| `features` | 2,002 clips (2,000 RWF-2000 + 2 feedback) → 11,818 training windows | 4 min |
| `train` | 5-fold CV + final fit → version **5** = `@challenger`; champion v4 re-scored on the same 400 test clips | 8.7 min |
| `drift_reference` | new reference from the training data incl. the feedback clips | < 1 min |
| `promote` | **rejected** (below) | – |
| `verify_serving` | skipped (nothing to verify) | – |
| `push_data`, `finish` | data version pushed to the lake; request #1 → `done` with the decision | – |

| Same 400 test clips | F1 | ROC AUC | Precision | Recall |
|---|---|---|---|---|
| challenger v5 | 0.6706 | 0.7187 | 0.804 | 0.575 |
| champion v4 (re-scored) | **0.6744** | **0.7209** | 0.806 | 0.580 |

`reports/promotion.json`: *"f1 -0.0038 (needs +0.01)"*, so v4 stays `@champion` and the serving API kept serving v4 without interruption.
This is the right outcome: two new clips among 1,600 can't change the model measurably, and the gate stops a retrained
model that is merely different from replacing one that works. As reviewers label more alerts (every false alarm on
this site becomes a NonFight example), the feedback set grows and the same pipeline promotes a challenger once it
actually beats the champion.

One issue found and fixed by this run: the first `train` attempt failed because the DVC stage's `python` was Airflow's
own interpreter (no PyTorch). The DAG now puts the training environment first on `PATH` for every DVC stage
(checked in `tests/test_dag.py`), and Airflow's automatic retry finished the run.

## 7. Tests

- `tests/test_feedback.py`: evidence parsing, 10 fps resampling around the alert, gaps, train/val split, the stage with a fake lake (missing frames and one-person clips are skipped).
- `tests/test_dag.py` (in the Airflow image): the DAG imports, its task order, trigger rules after a rejected challenger, schedule and timeouts.
- `tests/test_monitoring.py`: a drift alert starts the DAG; Airflow being down still records the request.
- `tests/test_backend.py`: the `evidence` column is added to an existing database and stored for stream alerts.
- Manual test cases TC-74 … TC-77 in [TESTING.md](TESTING.md).
