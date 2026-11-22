#!/usr/bin/env bash
# Model rollout on a deployment: the registry commands run in a one-off container of the Airflow image (it has
# the training environment with the MLflow client; the mlops profile doesn't need to be running), then the script
# waits until the serving API actually serves the new champion.
#
#   deploy/model.sh list                  versions, test scores, aliases, promotion decisions
#   deploy/model.sh promote               challenger -> champion if it passes the gate (params.yaml registry)
#   deploy/model.sh promote --force       ... even if it doesn't (a person decided, e.g. after the shadow results)
#   deploy/model.sh rollback <version>    champion -> that version
#   deploy/model.sh seed                  empty registry -> models/release becomes the champion (deploy.sh does this)
#
# Called by .github/workflows/model.yml (Actions -> "Model rollout" -> Run workflow).
set -euo pipefail
cd "$(dirname "$0")/.."
env_get() { grep -E "^$1=" .env 2>/dev/null | tail -1 | cut -d= -f2-; }
SERVING="http://127.0.0.1:$(env_get SERVING_PORT || true)"; [ "$SERVING" = "http://127.0.0.1:" ] && SERVING=http://127.0.0.1:8000
MLFLOW="http://127.0.0.1:$(env_get MLFLOW_PORT || true)"; [ "$MLFLOW" = "http://127.0.0.1:" ] && MLFLOW=http://127.0.0.1:5000
NAME=$(sed -n 's/^ *model_name: *//p' params.yaml | head -1); NAME=${NAME:-fight-detector}

registry() {
  docker compose --profile mlops run --rm --no-deps -T airflow-scheduler \
    bash -c "cd /opt/rover && /opt/trainer/bin/python -m mlops.registry $*"
}

champion_version() {
  curl -fsS -m 5 "$MLFLOW/api/2.0/mlflow/registered-models/alias?name=$NAME&alias=champion" \
    | sed -E 's/.*"version": ?"?([0-9]+)"?.*/\1/'
}

wait_for_serving() {             # the serving API polls the registry every RELOAD_SECONDS; ask it to look now
  local want token served=""
  want=$(champion_version)
  token=$(env_get SERVING_ADMIN_TOKEN || true)
  curl -fsS -m 10 -o /dev/null -X POST ${token:+-H "Authorization: Bearer $token"} "$SERVING/admin/reload" || true
  for _ in $(seq 1 40); do
    served=$(curl -fsS -m 5 "$SERVING/v1/models/fight" | sed -E 's/.*"version": ?"?([0-9]+)"?.*/\1/' || true)
    if [ "$served" = "$want" ]; then echo "serving: v$served is live"; return 0; fi
    sleep 3
  done
  echo "serving still runs v${served:-?}, registry champion is v$want"; return 1
}

case "${1:-}" in
  list)     registry list ;;
  seed)     registry seed ;;
  promote)  shift; registry promote "$@"; wait_for_serving ;;
  rollback) [ -n "${2:-}" ] || { echo "usage: deploy/model.sh rollback <version>"; exit 2; }
            registry set-alias champion "$2"; wait_for_serving ;;
  *)        sed -n '2,13p' "$0"; exit 2 ;;
esac
