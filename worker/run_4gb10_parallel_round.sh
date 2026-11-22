#!/usr/bin/env bash
set -euo pipefail

# 4-Independent-GB10 Parallel SparseLoCo Training + Owner Replay/FLTrust Gate + Offline Champion Scoring
# Runs ascent-05 (w4), ascent-06 (w5), ascent-07 (w6), ascent-08 (w7) simultaneously against https://draft.bertholomus.com

DSPINE=/tank/projects/glm53-dspine
SITE=/tank/projects/draft-site
W=/home/bertholomus/drs-gb10
IMG="nvcr.io/nvidia/pytorch:26.07-py3"
BORROW_LOGGED=0

log() { printf "[drs-4gb10 %s] %s\n" "$(date -u +%H:%M:%SZ)" "$*"; }

log_both() {
  printf -- "%s\n" "$1" >> "${SITE}/PROGRESS.md"
  printf -- "%s\n" "$1" >> "${DSPINE}/PROGRESS.md"
}

cleanup() {
  local rc=$?
  log "Cleaning up 4 GB10 worker containers and temp secrets (rc=${rc})..."
  for h in ascent-05 ascent-06 ascent-07 ascent-08; do
    ssh -o BatchMode=yes "$h" "docker rm -f drs-worker-w4 drs-worker-w5 drs-worker-w6 drs-worker-w7 drs-owner-gate >/dev/null 2>&1 || true; rm -rf ${W}/secrets" || true
  done
  if [ "$BORROW_LOGGED" -eq 1 ]; then
    log_both "- DRS NOTE borrow ascent-05..08 end 4-worker-parallel-SparseLoCo-round (rc=${rc}, GPUs freed) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

# 1. Verify all 4 Ascents are idle in nvidia-smi
for h in ascent-05 ascent-06 ascent-07 ascent-08; do
  procs=$(ssh -o BatchMode=yes "$h" "nvidia-smi --query-compute-apps=pid --format=csv,noheader" | tr -d "[:space:]" || true)
  if [ -n "$procs" ]; then
    log "ERROR: ${h} is busy (pids=${procs}); aborting."
    exit 1
  fi
done

BORROW_LOGGED=1
log_both "- DRS NOTE borrow ascent-05..08 start 4-worker-parallel-SparseLoCo-round (bertholomus-w4..w7 claiming glm53-dspine-train-r2 in parallel + owner gate + sim14/MP score) $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# 2. Ensure LAN shard fileserver is reachable on :19455
curl -sf http://192.168.1.49:19455/health >/dev/null

# 3. Copy worker keys (mode 600) to all 4 GB10s, plus admin key to ascent-05 for Owner Gate
for h in ascent-05 ascent-06 ascent-07 ascent-08; do
  ssh -o BatchMode=yes "$h" "mkdir -p ${W}/secrets ${W}/out && chmod 700 ${W}/secrets"
  scp -q "${SITE}/secrets/pilot_worker_keys.json" "${h}:${W}/secrets/"
  ssh -o BatchMode=yes "$h" "chmod 600 ${W}/secrets/pilot_worker_keys.json"
done
scp -q "${SITE}/secrets/admin_api_key.txt" "ascent-05:${W}/secrets/"
ssh -o BatchMode=yes ascent-05 "chmod 600 ${W}/secrets/admin_api_key.txt"

# 4. Launch ALL 4 INDEPENDENT GB10 WORKERS SIMULTANEOUSLY IN PARALLEL!
log "=== LAUNCHING 4 PARALLEL GB10 WORKER CONTAINERS (w4=A5, w5=A6, w6=A7, w7=A8) ==="
launch_one() {
  local host="$1" login="$2" slot="$3" cname="$4"
  log "Starting ${cname} (${login}) on ${host} -> slot ${slot}..."
  ssh -o BatchMode=yes "$host" "docker rm -f ${cname} >/dev/null 2>&1 || true; \
    docker run --rm --name ${cname} --gpus all --network host --ipc=host \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -v ${W}:/work -w /work ${IMG} \
    python3 -u /work/worker/gb10_worker.py \
      --url https://draft.bertholomus.com \
      --lan-base http://192.168.1.49:19455 \
      --keys-file /work/secrets/pilot_worker_keys.json \
      --login ${login} \
      --jobs glm53-dspine-train-r2 \
      --slot-id ${slot} \
      --out-dir /work/out" \
    > "${SITE}/notes/gb10-${login}.log" 2>&1
}

launch_one ascent-05 bertholomus-w4 r2_w0_bit200     drs-worker-w4 & P4=$!
launch_one ascent-06 bertholomus-w5 r2_w1_bit240     drs-worker-w5 & P5=$!
launch_one ascent-07 bertholomus-w6 r2_w2_bit280_lc  drs-worker-w6 & P6=$!
launch_one ascent-08 bertholomus-w7 r2_w3_bit322_400 drs-worker-w7 & P7=$!

# Power audit 45s into simultaneous 4-GB10 load
(
  sleep 45
  log "=== LIVE RACK POWER CHECK UNDER 4x GB10 PARALLEL LOAD ==="
  python3 /home/bertholomus/.hermes/profiles/minime/scripts/ha_power_total.py 2>&1 | sed 's/^/  /' || true
) &

RC=0
for pid in $P4 $P5 $P6 $P7; do
  if ! wait "$pid"; then
    RC=1
  fi
done
log "All 4 parallel GB10 workers finished (rc=${RC})"
for login in bertholomus-w4 bertholomus-w5 bertholomus-w6 bertholomus-w7; do
  log "--- tail ${login} ---"
  tail -n 6 "${SITE}/notes/gb10-${login}.log" || true
done
[ "$RC" -eq 0 ] || exit "$RC"

# 5. Run Owner Replay+FLTrust Gate, Aggregation, and Offline sim14+5-Profile MXFP8 Scoring on ascent-05!
log "=== RUNNING OWNER REPLAY+FLTRUST GATE & AGGREGATION (r2-4gb10-agg) ON ascent-05 ==="
ssh -o BatchMode=yes ascent-05 "docker rm -f drs-owner-gate >/dev/null 2>&1 || true; \
  docker run --rm --name drs-owner-gate --gpus all --network host --ipc=host \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v ${W}:/work -w /work ${IMG} \
  python3 -u /work/worker/gb10_worker.py \
    --url https://draft.bertholomus.com \
    --lan-base http://192.168.1.49:19455 \
    --keys-file /work/secrets/pilot_worker_keys.json \
    --admin-key-file /work/secrets/admin_api_key.txt \
    --login bertholomus-w4 \
    --role owner-gate \
    --base-ckpt v5-s2-w5-champion \
    --agg-name r2-4gb10-agg \
    --out-dir /work/out" \
  2>&1 | tee "${SITE}/notes/gb10-owner-gate.log"

# 6. Sync aggregated checkpoint and evidence back to FrankenNAS /tank/projects/draft-site/data/evals/
log "Syncing r2-4gb10-agg checkpoint and evidence back to ${SITE}/data/evals/..."
mkdir -p "${SITE}/data/evals/r2-4gb10-agg"
rsync -a "ascent-05:${W}/out/r2-4gb10-agg/" "${SITE}/data/evals/r2-4gb10-agg/"
rsync -a "ascent-05:${W}/out/r2-4gb10-agg-evidence.json" "${SITE}/data/evals/r2-4gb10-agg-evidence.json"
log "=== 4-GB10 PARALLEL ROUND + OWNER GATE + SCORING ALL COMPLETE! ==="
