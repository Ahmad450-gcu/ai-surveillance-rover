# Deployment and model rollout (Phase 8)

From a `git push` to a running, checked system without manual steps, and from a newly trained model to production
without guesswork. The same images and the same script deploy to the laptop or to an AWS EC2 server.

```mermaid
flowchart LR
  PUSH[git push main] --> CI[CI<br/>lint, 58 tests, image builds,<br/>DAG test, shellcheck, terraform validate]
  CI -- green --> PUB[CD: publish<br/>5 images, tag sha-xxxxxxx]
  PUB --> GHCR[(GitHub Container Registry<br/>+ Docker Hub)]
  PUB --> TGT{DEPLOY_TARGET}
  TGT -- local --> RUN[self-hosted runner<br/>on the laptop]
  TGT -- ec2 --> SSH[SSH to the<br/>EC2 server]
  RUN & SSH --> DEP[deploy/deploy.sh<br/>pull → up --wait → smoke tests]
  DEP -- fails --> RB[automatic rollback<br/>to the previous release]
  GHCR -.pull.-> DEP
```

## 1. Images

| Service | Image | Size |
|---|---|---|
| backend | `ghcr.io/<owner>/rover-backend` | 0.3 GB |
| serving | `ghcr.io/<owner>/rover-serving` (includes the release model) | 0.5 GB |
| mlflow | `ghcr.io/<owner>/rover-mlflow` | 1.3 GB |
| spark (also alert bridge, drift monitor) | `ghcr.io/<owner>/rover-spark` | 2.3 GB |
| airflow (with the training environment) | `ghcr.io/<owner>/rover-airflow` | 4.7 GB |

Tags: `sha-<7 chars>` (exactly one commit, what deployments use), `latest`, and `1.2.3` for a git tag `v1.2.3`.
Each image carries the commit as a label (`org.opencontainers.image.revision`). With the repository secrets
`DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` (Docker Hub → Account settings → Personal access tokens) the same images
are also pushed to `docker.io/<user>/rover-<service>`.

## 2. What a deployment does (`deploy/deploy.sh <commit>`)

1. **Checkout** of the commit in the deployment clone: the compose file, Prometheus rules, DAGs and the release model
   must belong to the images that are being started.
2. **Pull** of its images. If a pull fails, nothing has been touched and the running release stays.
3. **`docker compose up -d --wait`**: only containers whose image or settings changed are replaced; Compose waits
   for every health check (backend `/ready`, serving `/ready` = model loaded, MLflow, PostgreSQL, …). Data stays in
   the Docker volumes; the backend and Airflow migrate their databases on start.
4. **Smoke tests** (`deploy/smoke.sh`): readiness of backend, MLflow and serving, **a real prediction** with the
   served model's input shape, and the UIs of the enabled profiles (Prometheus, Grafana, Airflow, Spark).
5. **Registry seed** (mlops profile): a new server's empty registry gets the release model as v1 `@champion`, so
   the first retraining there has a model to beat.

If step 3 or 4 fails, the script deploys the previous release the same way (**automatic rollback**) and exits
with an error, so the CD run is red. `.deploy/current` holds the deployed commit, `.deploy/history` every attempt.

Which parts run is set once in the target's `.env`: `COMPOSE_PROFILES=` (core only), `mlops`, `monitoring,mlops`, ….

## 3. Model rollout

A model goes through four stages before and after it serves anyone:

| Stage | Where | What decides |
|---|---|---|
| **Challenger** | registry alias `@challenger` (retraining, Phase 7) | k-fold CV + hold-out test; the champion is re-scored on the same test clips |
| **Shadow** | the serving API loads `@challenger` next to `@champion` and scores every request with both; only the champion answers | live traffic: agreement rate and alert rate in Grafana (*Model & drift → Model rollout*) |
| **Champion** | `@champion`; the serving API hot-reloads it within `RELOAD_SECONDS` | the promotion gate (automatic in the DAG), or a person via the *Model rollout* workflow |
| **Rollback** | `@champion` moved back to an older version | *Model rollout* workflow → `rollback` + version; serving follows within seconds |

The *Model rollout* workflow (Actions → Model rollout → Run workflow) runs `deploy/model.sh` on the deployment target:
`list`, `promote` (gate), `promote-force` (a person overrides the gate) or `rollback <version>`. After a change it
asks serving to reload and waits until `/v1/models/fight` reports the new version, so a green run means the new model is
actually live. The run log is the audit trail of who changed the production model when.

**A new server and its first model.** The serving image contains `models/release/` (the champion when the image was
built: `python -m mlops.registry release`, committed with the code). While the server's registry has no champion,
serving answers with this model (`"source": "bundled"`) instead of failing its health check; the registry seed
(step 5 above) then makes it the registry's champion and serving switches to it.

## 4. Setting up a target

### Local (the laptop)

1. GitHub → repository → Settings → Actions → Runners → **New self-hosted runner** (Windows), install it, and give it the
   label `rover`. It runs only jobs of this repository.
2. A deployment clone, separate from the development folder:
   ```powershell
   git clone https://github.com/Ahmad450-gcu/ai-surveillance-rover.git C:\rover-deploy
   copy .env C:\rover-deploy\.env
   ```
   and in `C:\rover-deploy\.env`: `IMAGE_PREFIX=ghcr.io/ahmad450-gcu/rover`, `COMPOSE_PROFILES=` as wanted.
3. Settings → Secrets and variables → Actions → **Variables**: `DEPLOY_TARGET=local`, `DEPLOY_DIR=C:/rover-deploy`.

Both folders use the Compose project `rover`, so the deployment takes over the same containers and volumes (the same
database, lake and registry): a development `docker compose up` and a deployment simply replace each other's containers.

### AWS EC2

```powershell
cd deploy/aws
copy terraform.tfvars.example terraform.tfvars      # region, key pair, your IP
terraform init
terraform apply                                     # prints the address and the next steps
```

Terraform creates the server (Ubuntu 24.04, `t3.large` with 8 GB; `t3.xlarge` for every profile at once), an encrypted
60 GB disk, a fixed IP and a security group that opens SSH and the dashboard (8080) only to the addresses in
`terraform.tfvars`. cloud-init installs Docker from Ubuntu's repository, adds swap and clones the project to `/opt/rover`.
Then:

1. `ssh ubuntu@<ip>`, `cd /opt/rover && cp .env.example .env`, fill in the passwords, `IMAGE_PREFIX`, `COMPOSE_PROFILES`.
2. Variables: `DEPLOY_TARGET=ec2`, `EC2_HOST=<ip>`; secret `EC2_SSH_KEY` = the key pair's private key.
3. Actions → CD → Run workflow (or the next push to main).

GitHub's hosted runners connect from changing addresses, so SSH must either be open to them (`ssh_cidrs =
["0.0.0.0/0"]`; the server accepts keys only, no passwords) or, more strictly, the server runs its own
self-hosted runner with the label `rover` and the target is `local` with `DEPLOY_DIR=/opt/rover`, so no inbound SSH
from GitHub is needed at all. Moving between the laptop and EC2 only changes these variables: same images, same script.

**Cost:** an on-demand `t3.large` costs roughly USD 0.08–0.10 per hour depending on the region, plus the disk, so
about USD 2–3 per day of demo. `terraform destroy` removes everything when it is no longer needed.

## 5. Rolling back

| What | How |
|---|---|
| A deployment that fails its checks | automatic (previous release redeployed by `deploy.sh`) |
| A deployment that passed but misbehaves | Actions → CD → Run workflow with `ref` = the previous commit |
| A model | Actions → Model rollout → `rollback` + version (or `python -m mlops.registry set-alias champion <v>`) |

## 6. Results

Measured on the laptop (Windows 11, 7.4 GB RAM, Docker VM 3.6 GB), 22 Nov 2026. The deployment clone was a separate
folder with `COMPOSE_PROFILES=` (core) and `IMAGE_PREFIX=ghcr.io/ahmad450-gcu/rover`.

**Publishing.** Every push ran CI (4 jobs) and then CD: the 5 images were built and pushed in parallel to
`ghcr.io/ahmad450-gcu/rover-*` with `sha-…` and `latest` tags. The packages are public, so a server pulls them without
logging in. Docker Hub was skipped because its secrets are not set yet.

| Test | What happened | Time |
|---|---|---|
| First deployment (`fa5acec`) | pull (core images ≈ 2 GB) → `up --wait` → smoke tests: backend, MLflow, serving, prediction by registry v4: **deployed** | 7 min 17 s (≈ 6 min pulling) |
| Forced failure (`3e37bb5` + a check that can't pass) | `FAILED at smoke tests` → **rolled back to `fa5acec`** (images cached: 1 min) | 12 min |
| Model rollback v4 → v1 → v4 (`deploy/model.sh rollback`) | alias moved, serving reloaded, script confirmed `serving: v1 is live`, then `v4 is live` | 22 s, 19 s |
| Shadow scoring (400 real windows from the test set, champion v4, shadow v5) | every answer from v4; v5 scored all 400; decisions agreed on **82 %** of windows; windows above threshold 128 (v4) vs 129 (v5) | – |

The 82 % agreement is the reason for the shadow stage: v5 and v4 have almost the same test F1 and raise almost the same
number of alerts, yet they disagree on one window in five. A promotion that looks neutral by the numbers still changes
*which* situations raise an alarm, and the shadow metrics show this on real traffic before anyone promotes.

**Problems these tests found, now fixed:**

- The rollback re-ran the failing check that was meant for the new release, so a successful rollback was reported as
  failed. Extra checks (`SMOKE_EXTRA_URLS`) now apply only to the new release.
- `deploy.sh` checks out other commits while it runs, and bash reads a script while executing it. A commit that
  changes `deploy.sh` could have made the running deployment execute the wrong lines. The script now runs inside a
  function, so bash reads all of it before it starts.
- `model.sh` read the version from MLflow's multi-line JSON incorrectly. Serving had already switched, but the script
  waited 2 minutes and then reported a failure.
- On the overloaded laptop (Docker swap almost full), the backend sometimes answered on the host port only 1–3 minutes
  after Docker reported it healthy. The smoke tests now wait up to about 6 minutes per service. A plain restart
  answers within 6 s, so this is memory pressure, not the deployment.

FINAL_DEPLOY_PLACEHOLDER

## 7. Tests

- `tests/test_serving.py`: the bundled release model serves until the registry has a champion and is never used
  again after that; the shadow model scores every request but never answers, and its metrics count each window;
  a broken challenger doesn't affect the champion.
- `tests/test_models.py`: `seed` registers the release model once, with the artifacts serving reads and the metrics
  the promotion gate needs.
- CI: `shellcheck` on every deployment script, `terraform fmt` + `validate` on the EC2 configuration.
- Manual test cases TC-78 … TC-81 in [TESTING.md](TESTING.md).
