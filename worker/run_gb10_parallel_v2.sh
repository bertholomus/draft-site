#!/usr/bin/env bash
set -euo pipefail

# Parallel multi-GB10 runner using all currently idle Ascents (ascent-05..08):
# - Runs DSpine R2's requested cross-profile eval (r2-v6-ema: 5-prof MXFP8 + sim14) on idle ascent-08 (bertholomus-w7)
# - Runs all 4 glm53-dspine-train-r2 SparseLoCo slots in parallel across all idle GB10s (w4 on A5, w7 on A8, plus w5/w6 if A6/A7 idle)
# - Runs Owner Held-Out Replay + FLTrust Cosine Gate + r2-4gb10-agg aggregation + offline sim14/MP champion scoring on ascent-05
# - Syncs r2-4gb10-agg, r2-4gb10-agg-evidence.json, and r2-v6-ema-eval-summary.json back to FrankenNAS and wipes temp secrets!

DSPINE=/tank/projects/glm53-dspine
SITE=/tank/projects/draft-site
W=/home/bertholomus/drs-gb10
IMG="nvcr.io/nvidia/pytorch:26.07-py3"
BORROW_HOSTS=()

log() { printf "[drs-gb10-v2 %s] %s\n" "$(date -u +%H:%M:%SZ)" "$*"; }

log_both() {
  printf -- "%s\n" "$1" >> "${SITE}/PROGRESS.md"
  printf -- "%s\n" "$1" >> "${DSPINE}/PROGRESS.md"
}

cleanup() {
  local rc=$?
  log "Cleaning up GB10 worker containers and temp secrets (rc=${rc})..."
  for h in "${BORROW_HOSTS[@]}"; do
    ssh -o BatchMode=yes "$h" "docker rm -f drs-worker-w4 drs-worker-w5 drs-worker-w6 drs-worker-w7 drs-eval-r2v6 drs-owner-gate >/dev/null 2>&1 || true; rm -rf ${W}/secrets" || true
  done
  if [ "${#BORROW_HOSTS[@]}" -gt 0 ]; then
    log_both "- DRS NOTE borrow ${BORROW_HOSTS[*]} end parallel-gb10-r2-batch (rc=${rc}, GPUs freed) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

# 1. Detect idle GB10s among ascent-05..08 (never touch busy GPUs!)
for h in ascent-05 ascent-06 ascent-07 ascent-08; do
  procs=$(ssh -o BatchMode=yes "$h" "nvidia-smi --query-compute-apps=pid --format=csv,noheader" | tr -d "[:space:]" || true)
  if [ -z "$procs" ]; then
    BORROW_HOSTS+=("$h")
  else
    log "Skipping busy host ${h} (pids=${procs})"
  fi
done

if [ "${#BORROW_HOSTS[@]}" -lt 2 ]; then
  log "ERROR: need at least 2 idle GB10s for parallel multi-host proof; found: ${BORROW_HOSTS[*]}"
  exit 1
fi
log "Idle GB10 fleet ready: ${BORROW_HOSTS[*]}"
log_both "- DRS NOTE borrow ${BORROW_HOSTS[*]} start parallel-gb10-r2-batch (r2-v6@500 cross-profile eval + 4-slot SparseLoCo train-r2 + owner gate + r2-4gb10-agg score) $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# 2. Sync fixed zero-mmap gb10_worker.py + verified keys to all borrowed GB10s
for h in "${BORROW_HOSTS[@]}"; do
  ssh -o BatchMode=yes "$h" "mkdir -p ${W}/{secrets,out,r2-v6-ema,eval/{bit200,bit240,bit280,bit322,bit400},dumps} && chmod 700 ${W}/secrets"
  rsync -a "${SITE}/repo/worker/client.py" "${SITE}/repo/worker/gb10_worker.py" "${h}:${W}/worker/"
  scp -q "${SITE}/secrets/pilot_worker_keys.json" "${h}:${W}/secrets/"
  ssh -o BatchMode=yes "$h" "chmod 600 ${W}/secrets/pilot_worker_keys.json"
done

# Stage admin key + eval/dumps + r2-v6@500 EMA onto primary eval/gate hosts
GATE_HOST="${BORROW_HOSTS[0]}"
EVAL_HOST="${BORROW_HOSTS[-1]}"
for h in "$GATE_HOST" "$EVAL_HOST"; do
  scp -q "${SITE}/secrets/admin_api_key.txt" "${h}:${W}/secrets/"
  ssh -o BatchMode=yes "$h" "chmod 600 ${W}/secrets/admin_api_key.txt"
done
for pr in bit200 bit240 bit280 bit322 bit400; do
  rsync -a "/var/tmp/dspine-shards-${pr}/eval/" "${EVAL_HOST}:${W}/eval/${pr}/"
done
rsync -a "${DSPINE}/rows/b2r-slot-20261120T071919Z/b3/b2-dfl-32k.bin" \
         "${DSPINE}/rows/b2r-slot-20261120T071919Z/b3/b2-dfl-100k.bin" \
         "${DSPINE}/rows/eval-mc-20261120T083643Z/split/"*.bin \
         "${EVAL_HOST}:${W}/dumps/"
rsync -a "${DSPINE}/out/dspine-r2-v6/ema_step_00500/" "${EVAL_HOST}:${W}/r2-v6-ema/"
log "Staged zero-mmap worker + r2-v6@500 EMA onto ${BORROW_HOSTS[*]}"

host_to_login() {
  case "$1" in
    ascent-05) echo "bertholomus-w4" ;;
    ascent-06) echo "bertholomus-w5" ;;
    ascent-07) echo "bertholomus-w6" ;;
    ascent-08) echo "bertholomus-w7" ;;
  esac
}

# 3. Launch IN PARALLEL:
#    - EVAL_HOST runs r2-v6-ema cross-profile MXFP8 + sim14 eval (requested by DSpine R2) then loops to claim remaining train-r2 slots!
#    - All other idle GB10 hosts immediately loop to claim glm53-dspine-train-r2 slots in parallel!
PIDS=()
for h in "${BORROW_HOSTS[@]}"; do
  login=$(host_to_login "$h")
  cname="drs-worker-${login#bertholomus-}"
  if [ "$h" = "$EVAL_HOST" ]; then
    (
      log "[${h}/${login}] Running r2-v6@500 EMA 5-profile MXFP8 + sim14 eval via https://draft.bertholomus.com..."
      ssh -o BatchMode=yes "$h" "docker rm -f drs-eval-r2v6 >/dev/null 2>&1 || true; \
        docker run --rm --name drs-eval-r2v6 --gpus all --network host --ipc=host \
        -e PYTHONDONTWRITEBYTECODE=1 -v ${W}:/work -w /work ${IMG} \
        python3 -u /work/worker/gb10_worker.py \
          --url https://draft.bertholomus.com \
          --lan-base http://192.168.1.49:19455 \
          --keys-file /work/secrets/pilot_worker_keys.json \
          --admin-key-file /work/secrets/admin_api_key.txt \
          --login ${login} \
          --role eval-ckpt \
          --eval-ckpt-dir /work/r2-v6-ema \
          --eval-ckpt-rev r2-v6-ema \
          --out-dir /work/out" \
        > "${SITE}/notes/gb10-eval-r2v6.log" 2>&1
      log "[${h}/${login}] r2-v6@500 EMA eval DONE! Joining parallel train-r2 pool..."
      tail -n 3 "${SITE}/notes/gb10-eval-r2v6.log" || true
      rsync -a "${h}:${W}/out/r2-v6-ema-eval-summary.json" "${SITE}/data/evals/r2-v6-ema-eval-summary.json" || true

      # Now claim remaining open glm53-dspine-train-r2 slots until queue empty!
      ssh -o BatchMode=yes "$h" "docker rm -f ${cname} >/dev/null 2>&1 || true; \
        docker run --rm --name ${cname} --gpus all --network host --ipc=host \
        -e PYTHONDONTWRITEBYTECODE=1 -v ${W}:/work -w /work ${IMG} \
        bash -c 'while python3 -u /work/worker/gb10_worker.py \
          --url https://draft.bertholomus.com \
          --lan-base http://192.168.1.49:19455 \
          --keys-file /work/secrets/pilot_worker_keys.json \
          --login ${login} \
          --jobs glm53-dspine-train-r2 \
          --out-dir /work/out | tee /tmp/last.log | grep -q \"Uploaded\"; do true; done'" \
        > "${SITE}/notes/gb10-${login}.log" 2>&1
    ) &
    PIDS+=($!)
  else
    (
      log "[${h}/${login}] Joining parallel glm53-dspine-train-r2 worker pool..."
      ssh -o BatchMode=yes "$h" "docker rm -f ${cname} >/dev/null 2>&1 || true; \
        docker run --rm --name ${cname} --gpus all --network host --ipc=host \
        -e PYTHONDONTWRITEBYTECODE=1 -v ${W}:/work -w /work ${IMG} \
        bash -c 'while python3 -u /work/worker/gb10_worker.py \
          --url https://draft.bertholomus.com \
          --lan-base http://192.168.1.49:19455 \
          --keys-file /work/secrets/pilot_worker_keys.json \
          --login ${login} \
          --jobs glm53-dspine-train-r2 \
          --out-dir /work/out | tee /tmp/last.log | grep -q \"Uploaded\"; do true; done'" \
        > "${SITE}/notes/gb10-${login}.log" 2>&1
    ) &
    PIDS+=($!)
  fi
done

# Live rack power check 60s into parallel GB10 load
(
  sleep 60
  log "=== LIVE RACK POWER CHECK UNDER PARALLEL GB10 LOAD ==="
  python3 /home/bertholomus/.hermes/profiles/minime/scripts/ha_power_total.py 2>&1 | sed 's/^/  /' || true
) &

RC=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    RC=1
  fi
done
log "Parallel GB10 worker pool finished (rc=${RC})"
for h in "${BORROW_HOSTS[@]}"; do
  login=$(host_to_login "$h")
  log "--- tail ${login} (${h}) ---"
  tail -n 6 "${SITE}/notes/gb10-${login}.log" || true
done
[ "$RC" -eq 0 ] || exit "$RC"

# 4. Run Owner Held-Out Replay + FLTrust Cosine Gate + r2-4gb10-agg Aggregation + sim14/MP Scoring on GATE_HOST!
GATE_LOGIN=$(host_to_login "$GATE_HOST")
log "=== RUNNING OWNER REPLAY+FLTRUST GATE & AGGREGATION (r2-4gb10-agg) ON ${GATE_HOST} (${GATE_LOGIN}) ==="
ssh -o BatchMode=yes "$GATE_HOST" "docker rm -f drs-owner-gate >/dev/null 2>&1 || true; \
  docker run --rm --name drs-owner-gate --gpus all --network host --ipc=host \
  -e PYTHONDONTWRITEBYTECODE=1 -v ${W}:/work -w /work ${IMG} \
  python3 -u /work/worker/gb10_worker.py \
    --url https://draft.bertholomus.com \
    --lan-base http://192.168.1.49:19455 \
    --keys-file /work/secrets/pilot_worker_keys.json \
    --admin-key-file /work/secrets/admin_api_key.txt \
    --login ${GATE_LOGIN} \
    --role owner-gate \
    --base-ckpt v5-s2-w5-champion \
    --agg-name r2-4gb10-agg \
    --out-dir /work/out" \
  2>&1 | tee "${SITE}/notes/gb10-owner-gate.log"

log "Syncing r2-4gb10-agg checkpoint and evidence back to ${SITE}/data/evals/..."
mkdir -p "${SITE}/data/evals/r2-4gb10-agg"
rsync -a "${GATE_HOST}:${W}/out/r2-4gb10-agg/" "${SITE}/data/evals/r2-4gb10-agg/"
rsync -a "${GATE_HOST}:${W}/out/r2-4gb10-agg-evidence.json" "${SITE}/data/evals/r2-4gb10-agg-evidence.json"
log "=== ALL PARALLEL GB10 R2 WORK + OWNER GATE + R2-V6 EVAL COMPLETE! ==="
