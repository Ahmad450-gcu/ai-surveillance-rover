#!/usr/bin/env bash
# Smoke tests of a running deployment (deploy/deploy.sh runs them after every deployment; also fine by hand).
# Beyond the health checks: the serving API answers a real prediction, and each enabled profile's UI responds.
#   SMOKE_EXTRA_URLS="http://..."   more URLs that must answer (space-separated)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
env_get() { grep -E "^$1=" .env 2>/dev/null | tail -1 | cut -d= -f2-; }
port() { local v; v=$(env_get "$1"); echo "${v:-$2}"; }
PROFILES=${COMPOSE_PROFILES:-$(env_get COMPOSE_PROFILES)}
H=http://127.0.0.1
failed=0

check() {                        # name url [tries]
  for _ in $(seq 1 "${3:-20}"); do
    if curl -fsS -m 5 -o /dev/null "$2"; then echo "  ok    $1"; return 0; fi
    sleep 3
  done
  echo "  FAIL  $1 ($2)"; failed=1
}

check backend "$H:$(port BACKEND_PORT 8080)/ready"
check mlflow "$H:$(port MLFLOW_PORT 5000)/health"
check serving "$H:$(port SERVING_PORT 8000)/ready"

# a prediction with the served model's own input shape
SERVING="$H:$(port SERVING_PORT 8000)/v1/models/fight"
if info=$(curl -fsS -m 5 "$SERVING"); then
  shape=$(echo "$info" | sed -E 's/.*"window": ?([0-9]+).*"n_feat": ?([0-9]+).*/\1 \2/')
  read -r T F <<< "$shape"
  zeros=$(printf '0,%.0s' $(seq 1 "$F")); row="[${zeros%,}]"
  rows=$(printf "$row,%.0s" $(seq 1 "$T")); body="{\"instances\": [[${rows%,}]]}"
  if out=$(curl -fsS -m 10 -H 'Content-Type: application/json' -d "$body" "$SERVING/predict") \
     && echo "$out" | grep -q '"probabilities"'; then
    echo "  ok    prediction ($(echo "$info" | sed -E 's/.*"source": ?"([a-z]+)".*"version": ?"?([0-9a-z]*)"?.*/\1 v\2/'))"
  else
    echo "  FAIL  prediction"; failed=1
  fi
else
  echo "  FAIL  serving model info"; failed=1
fi

[[ "$PROFILES" == *monitoring* ]] && { check prometheus "$H:$(port PROMETHEUS_PORT 9090)/-/ready"
                                       check grafana "$H:$(port GRAFANA_PORT 3000)/api/health"; }
[[ "$PROFILES" == *mlops* ]] && check airflow "$H:$(port AIRFLOW_PORT 8081)/health" 40
[[ "$PROFILES" == *streaming* ]] && check spark "$H:$(port SPARK_UI_PORT 4040)" 40
for url in ${SMOKE_EXTRA_URLS:-}; do check "$url" "$url" 3; done
exit $failed
