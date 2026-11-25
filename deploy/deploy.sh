#!/usr/bin/env bash
# Deploy one commit on this machine: the laptop ("local") or a cloud server (AWS EC2). Called by the CD workflow
# (.github/workflows/cd.yml); can also be run by hand. Runs inside the deployment clone, whose .env holds the
# secrets, IMAGE_PREFIX (where the published images are) and COMPOSE_PROFILES (which parts run).
#
#   deploy/deploy.sh <git-sha>
#
#   1. check out the commit       compose file, configs and DAGs must match the images built from it
#   2. pull its images            <IMAGE_PREFIX>-<service>:sha-<7 chars>; if this fails nothing has changed yet
#   3. docker compose up --wait   replaces only the containers whose image or config changed, waits for every
#                                 health check (data stays in the volumes; database migrations run on start)
#   4. smoke tests                deploy/smoke.sh: readiness, a real prediction, the profile's UIs
#   5. model registry seed        a new server's empty registry gets the release model as champion (mlops profile)
#
# If 3 or 4 fails, the previous release is deployed again the same way (automatic rollback) and the script fails,
# so the CD run is red. State: .deploy/current (deployed commit), .deploy/history (every attempt).
set -euo pipefail

# Everything is inside main(): bash reads a script while running it, and this one checks out other commits
# (possibly with a different deploy.sh). Wrapped in a function, the whole script is read before anything runs.
main() {
  cd "$(dirname "$0")/.."

  SHA=${1:?usage: deploy/deploy.sh <git-sha>}
  STATE=.deploy
  mkdir -p "$STATE"
  PREV=$(cat "$STATE/current" 2>/dev/null || true)
  log() { echo "[deploy $(date -u +%H:%M:%S)] $*"; }
  record() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $1 $2" >> "$STATE/history"; }

  [ -f .env ] || { log "no .env in $(pwd): copy .env.example and fill it in first (docs/DEPLOYMENT.md)"; exit 2; }
  prefix=$(grep -E '^IMAGE_PREFIX=' .env | tail -1 | cut -d= -f2-)
  case "${IMAGE_PREFIX:-$prefix}" in
    ""|rover) log "IMAGE_PREFIX in .env must name the published images, e.g. ghcr.io/<user>/rover"; exit 2 ;;
  esac

  # A retraining on this machine leaves new versions of committed files (dvc.lock, reports/, the feedback manifest).
  # git keeps them while the new commit doesn't change the same files; if it does, they are saved and replaced.
  checkout() {
    if git checkout --quiet --detach "$1" 2> "$STATE/checkout.err"; then return 0; fi
    local keep
    keep="$STATE/kept/$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$keep"
    git diff --name-only -z HEAD --diff-filter=d | xargs -0 -r cp --parents -t "$keep" || return 1
    log "local changes (a retraining's results) saved in $keep"
    git checkout --quiet --force --detach "$1"
  }

  set_tag() {                      # plain "docker compose" commands on this machine use the deployed images too
    if grep -q '^IMAGE_TAG=' .env; then sed -i "s/^IMAGE_TAG=.*/IMAGE_TAG=$1/" .env; else echo "IMAGE_TAG=$1" >> .env; fi
  }

  release() {                      # $1 = commit: checkout + pull + up + smoke
    checkout "$1" || return 10
    export IMAGE_TAG="sha-${1:0:7}"
    log "pulling images $IMAGE_TAG"
    docker compose pull --quiet || return 10
    log "starting containers (profiles: ${COMPOSE_PROFILES:-$(grep -E '^COMPOSE_PROFILES=' .env | cut -d= -f2-)})"
    docker compose up -d --no-build --remove-orphans --wait --wait-timeout "${DEPLOY_WAIT_SECONDS:-600}" || return 20
    log "smoke tests"
    bash deploy/smoke.sh || return 30
    set_tag "$IMAGE_TAG"
  }

  git fetch --quiet origin || log "git fetch failed; using the commits already here"
  git cat-file -e "$SHA^{commit}" 2>/dev/null || { log "unknown commit $SHA"; exit 2; }
  SHA=$(git rev-parse "$SHA")
  log "deploying ${SHA:0:7}${PREV:+ (now running ${PREV:0:7})}"

  set +e
  release "$SHA"
  rc=$?
  set -e
  if [ $rc -eq 0 ]; then
    if grep -qE '^COMPOSE_PROFILES=.*mlops' .env || [[ "${COMPOSE_PROFILES:-}" == *mlops* ]]; then
      bash deploy/model.sh seed || log "registry seed failed (serving keeps the bundled release model)"
    fi
    echo "$SHA" > "$STATE/current"
    record "$SHA" ok
    # older releases' images (this project's only); this one and the previous one stay for a quick rollback
    docker images --format '{{.Repository}}:{{.Tag}}' --filter "reference=${IMAGE_PREFIX:-$prefix}-*" \
      | { grep -vE ":(sha-${SHA:0:7}|sha-${PREV:0:7}|latest)\$" || true; } | xargs -r docker rmi > /dev/null 2>&1 || true
    log "deployed ${SHA:0:7}"
    exit 0
  fi

  stage=$([ $rc -eq 10 ] && echo "image pull" || ([ $rc -eq 20 ] && echo "health checks" || echo "smoke tests"))
  record "$SHA" "failed:$stage"
  log "FAILED at $stage"
  if [ $rc -eq 10 ]; then             # nothing was replaced: the running release is untouched
    if [ -n "$PREV" ]; then git checkout --quiet --detach "$PREV"; fi
    exit 1
  fi
  if [ -z "$PREV" ] || [ "$PREV" = "$SHA" ]; then
    log "no previous release to roll back to"
    exit 1
  fi
  log "rolling back to ${PREV:0:7}"
  unset SMOKE_EXTRA_URLS              # extra checks were for the new release; the old one gets the standard checks
  if release "$PREV"; then
    record "$PREV" rollback
    log "rolled back to ${PREV:0:7}: the previous release is serving again"
  else
    record "$PREV" rollback-failed
    log "ROLLBACK FAILED: check 'docker compose ps' and the logs"
  fi
  exit 1
}

main "$@"
exit $?
