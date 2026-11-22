#!/usr/bin/env bash
set -euo pipefail

DSPINE=/tank/projects/glm53-dspine
SITE=/tank/projects/draft-site
PORT=19468
HF_PORT=19469
LEASE_ID=""
FS_PID=""
HF_PID=""

log() { printf "[drs-ms %s] %s\n" "$(date -u +%H:%M:%SZ)" "$*"; }

cleanup() {
  local rc=$?
  ssh -o BatchMode=yes mastersim-linux "docker rm -f drs-gpu-pilot >/dev/null 2>&1 || true; docker run --rm -v /home/bertholomus/drs-work:/s alpine rm -rf /s/secrets /s/out /s/vocab /s/anchor /s/champion /s/eval /s/dumps /s/exlr8-ref /s/pydeps /s/cap_cfg /s/cap_in >/dev/null 2>&1 || true; rm -rf /home/bertholomus/drs-work || true" || true
  [ -n "$FS_PID" ] && kill "$FS_PID" >/dev/null 2>&1 || true
  [ -n "$HF_PID" ] && kill "$HF_PID" >/dev/null 2>&1 || true
  if [ -n "$LEASE_ID" ]; then
    log "Releasing gpulease ${LEASE_ID} (rc=${rc})"
    gpulease release "$LEASE_ID" >/dev/null 2>&1 || true
    printf -- "- DRS NOTE released mastersim-pro6000 lease %s (rc=%d) %s\n" "$LEASE_ID" "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${DSPINE}/PROGRESS.md"
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

# 1. Acquire polite background lease (DSpine R2 keeps first call per Guardrail 2)
log "Requesting polite background gpulease on mastersim-pro6000..."
RAW=$(gpulease acquire mastersim-pro6000 --who neon-drs --why "DRS community drafter pilot (eval + 4w SparseLoCo train + Byzantine gate + offline champion score)" --priority background --expect-min 25 --max-hours 2 --bind ttl --key drs-pilot-gpu --json)
LEASE_ID=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d.get('lease') or d.get('id') or '')" "$RAW")
STATE=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d.get('state') or d.get('status') or '')" "$RAW")
log "Initial gpulease: lease=${LEASE_ID} state=${STATE}"
while [ "$STATE" != "active" ] && [ "$STATE" != "granted" ] && [ "$STATE" != "live" ]; do
  WRAW=$(gpulease wait "$LEASE_ID" --timeout 300 --json 2>&1 || true)
  STATE=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d.get('state') or d.get('status') or '')" "$WRAW" 2>/dev/null || echo "queued")
  log "Waiting for lease=${LEASE_ID} state=${STATE}"
  [ "$STATE" = "active" ] || [ "$STATE" = "granted" ] || [ "$STATE" = "live" ] && break
  sleep 10
done

# 2. Verify nvidia-smi shows 0 compute apps (Guardrail 2)
for _ in 1 2 3 4 5 6; do
  PROCS=$(ssh -o BatchMode=yes mastersim-linux "nvidia-smi --query-compute-apps=pid --format=csv,noheader" | tr -d "[:space:]" || true)
  [ -z "$PROCS" ] && break
  log "Waiting for GPU compute pids to clear (pids=${PROCS})..."
  sleep 5
done
if [ -n "$PROCS" ]; then
  log "ERROR: mastersim-pro6000 still has active compute pids=${PROCS}; aborting."
  exit 1
fi

printf -- "- DRS NOTE acquired mastersim-pro6000 lease %s for closed community-drafter pilot (eval + 4w SparseLoCo train + Byzantine gate + offline champion score) %s\n" "$LEASE_ID" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${DSPINE}/PROGRESS.md"

# 3. Ephemeral read-only LAN shard server (:19468) + LAN HF mirror (:19469) for fresh 78-layer exlr8 GPU capture
python3 -u "${DSPINE}/scripts/dspine_lan_fileserver.py" 0.0.0.0 "$PORT" >"${SITE}/notes/drs-lan-fs.log" 2>&1 &
FS_PID=$!
python3 -u "${DSPINE}/scripts/lan_hf_mirror.py" 0.0.0.0 "$HF_PORT" >"${SITE}/notes/drs-lan-hf.log" 2>&1 &
HF_PID=$!
for _ in 1 2 3 4 5 6 7 8 9 10; do
  curl -sf "http://192.168.1.49:${PORT}/health" >/dev/null && break
  sleep 0.5
done
ssh -o BatchMode=yes mastersim-linux "curl -sf http://192.168.1.49:${PORT}/health >/dev/null"
log "Ephemeral LAN shard server (:${PORT}) and HF exlr8 mirror (:${HF_PORT}) healthy"

# 4. Stage minimal read-only copies + keys into mastersim-linux:~/drs-work (335 GB free NVMe, wiped on exit!)
W=/home/bertholomus/drs-work
log "Staging vocab, anchor, champion, eval shards, dumps, scripts into mastersim-linux:${W}..."
ssh -o BatchMode=yes mastersim-linux "rm -rf ${W} && mkdir -p ${W}/{vocab,anchor,champion,dumps,cap_cfg,cap_in,scripts/tools/convert,worker,secrets,out,eval/{bit200,bit240,bit280,bit322,bit400}} && chmod 700 ${W}/secrets && cp -a /home/bertholomus/r2-cap/exlr8-ref ${W}/exlr8-ref && cp -a /home/bertholomus/r2-cap/pydeps ${W}/pydeps && cp -a /home/bertholomus/r2-cap/cfg/* ${W}/cap_cfg/ && cp -a /home/bertholomus/r2-cap/scripts/{b1_capture.py,batch_bf16_capture_v3.py,exlr8_capture.py,exlr8_fast_attn.py,exlr8_fast_capture.py,exlr8_lan_capture.py} ${W}/scripts/"
head -n 2 "${DSPINE}/data/labels/eval_labels.jsonl" | ssh -o BatchMode=yes mastersim-linux "cat > ${W}/cap_in/fresh_2seqs.jsonl"
rsync -a "${DSPINE}/out/glm53-vocab-ref/" "mastersim-linux:${W}/vocab/"
rsync -a "${DSPINE}/out/eval-dflash2-ref/" "mastersim-linux:${W}/anchor/"
rsync -a "${DSPINE}/out/dspine-v5-s2-w5/best_dspine/" "mastersim-linux:${W}/champion/"
for pr in bit200 bit240 bit280 bit322 bit400; do
  rsync -a "/var/tmp/dspine-shards-${pr}/eval/" "mastersim-linux:${W}/eval/${pr}/"
done
rsync -a "${DSPINE}/rows/b2r-slot-20261120T071919Z/b3/b2-dfl-32k.bin" \
         "${DSPINE}/rows/b2r-slot-20261120T071919Z/b3/b2-dfl-100k.bin" \
         "${DSPINE}/rows/eval-mc-20261120T083643Z/split/"*.bin \
         "mastersim-linux:${W}/dumps/"
rsync -a "${DSPINE}/scripts/dspine_v3.py" \
         "${DSPINE}/scripts/dspine_v4.py" \
         "${DSPINE}/scripts/train_dspine_v4.py" \
         "${DSPINE}/scripts/eval_dspine_v3.py" \
         "/tank/projects/glm53-gr8-tp4/engine/gr8-attic-ca7e874/engine/ref/"*.py \
         "${DSPINE}/research/community-drafter/work/gpu_pilots.py" \
         "mastersim-linux:${W}/scripts/"
rsync -a "/tank/projects/glm53-gr8-tp4/engine/gr8-attic-ca7e874/tools/convert/"*.py \
         "mastersim-linux:${W}/scripts/tools/convert/"
ssh -o BatchMode=yes mastersim-linux "touch ${W}/scripts/tools/__init__.py ${W}/scripts/tools/convert/__init__.py"
rsync -a "${SITE}/repo/worker/client.py" "${SITE}/repo/worker/run_gpu_pilot.py" "mastersim-linux:${W}/worker/"
rsync -a "${DSPINE}/NOTICE" "${DSPINE}/out/dspine-v5-s2-w5/sim14-mxfp8.json" "${DSPINE}/out/dspine-v5-s2-w5/multiprofile-mxfp8.json" "mastersim-linux:${W}/champion/"
scp -q "${SITE}/secrets/pilot_worker_keys.json" "${SITE}/secrets/admin_api_key.txt" "mastersim-linux:${W}/secrets/"
ssh -o BatchMode=yes mastersim-linux "chmod 600 ${W}/secrets/*"

# 5. Renew lease periodically in background while running
( while true; do sleep 600; gpulease renew "$LEASE_ID" >/dev/null 2>&1 || true; done ) &
RENEW_PID=$!

log "Launching drs-gpu-pilot container on mastersim-pro6000..."
set +e
ssh -o BatchMode=yes mastersim-linux "docker rm -f drs-gpu-pilot >/dev/null 2>&1 || true; \
  docker run --rm --name drs-gpu-pilot --gpus all --network host --ipc=host \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v ${W}:/work -w /work nvcr.io/nvidia/pytorch:26.07-py3 \
  python3 -u /work/worker/run_gpu_pilot.py \
    --url https://draft.bertholomus.com \
    --lan-base http://192.168.1.49:${PORT} \
    --hf-mirror-base http://192.168.1.49:${HF_PORT}/kindlingai/glm-5.3-exlr8-k2-k3-k4/resolve/main \
    --worker-keys-file /work/secrets/pilot_worker_keys.json \
    --admin-key-file /work/secrets/admin_api_key.txt \
    --out-dir /work/out" \
  2>&1 | tee "${SITE}/notes/gpu-pilot.log"
RUN_RC=${PIPESTATUS[0]}
set -e
kill "$RENEW_PID" >/dev/null 2>&1 || true

# 6. Copy evidence and aggregated checkpoint back to /tank/projects/draft-site/data/evals/
if [ "$RUN_RC" -eq 0 ]; then
  log "Syncing aggregated checkpoint and offline eval evidence to ${SITE}/data/evals/..."
  mkdir -p "${SITE}/data/evals/pilot-r1-agg"
  rsync -a "mastersim-linux:${W}/out/pilot-r1-agg/" "${SITE}/data/evals/pilot-r1-agg/"
  rsync -a "mastersim-linux:${W}/out/gpu-pilot-evidence.json" "${SITE}/data/evals/gpu-pilot-evidence.json"
  log "ALL GPU PILOT ARTIFACTS SYNCED!"
fi
exit "$RUN_RC"
