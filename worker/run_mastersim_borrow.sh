#!/usr/bin/env bash
set -euo pipefail

# Guardrail 2a borrow driver for mastersim-pro6000:
# - Announces borrow start/end in BOTH draft-site/PROGRESS.md and glm53-dspine/PROGRESS.md
# - Pauses DSpine Wave-6 LC cleanly if active (between shards, never mid-shard-write)
# - Runs drs-gpu-pilot container (~22-28 min) strictly against https://draft.bertholomus.com
# - Syncs outputs & evidence, wipes temp staging/secrets, verifies GPU free
# - Cleanly resumes DSpine Wave-6 LC pass if we paused it

DSPINE=/tank/projects/glm53-dspine
SITE=/tank/projects/draft-site
PORT=19468
HF_PORT=19469
FS_PID=""
HF_PID=""
BORROW_LOGGED=0
PAUSED_WAVE6=0
W=/home/bertholomus/drs-work

log() { printf "[drs-borrow %s] %s\n" "$(date -u +%H:%M:%SZ)" "$*"; }

log_both_progress() {
  local line="$1"
  printf -- "%s\n" "$line" >> "${SITE}/PROGRESS.md"
  printf -- "%s\n" "$line" >> "${DSPINE}/PROGRESS.md"
}

resume_wave6_lc() {
  log "Resuming DSpine Wave-6 LC pass on mastersim-pro6000..."
  local resume_script="${SITE}/notes/resume_wave6_lc.sh"
  cat << 'RSEOF' > "$resume_script"
#!/usr/bin/env bash
set -euo pipefail
DSPINE=/tank/projects/glm53-dspine
HTTP_BASE="http://192.168.1.49:19456/kindlingai/glm-5.3-exlr8-k2-k3-k4/resolve/main"
IMG="nvcr.io/nvidia/pytorch:26.07-py3"

log() { printf "[wave6-ms-resume %s] %s\n" "$(date -u +%H:%M:%SZ)" "$*"; }
prog() { printf -- "- DSP R2 %s %s\n" "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${DSPINE}/PROGRESS.md"; }

if ! curl -sf http://192.168.1.49:19456/health >/dev/null; then
  systemctl --user restart dspine-lan-hf.service 2>/dev/null || systemd-run --user --unit=dspine-lan-hf --collect python3 "${DSPINE}/scripts/lan_hf_mirror.py" 192.168.1.49 19456
  sleep 2
fi

log "=== Resuming Part 2/2: Wave-6 Long-Context (len=2049..16384) on mastersim-pro6000 ==="
for pass in 1 2 3; do
  log "Wave-6 LC resumed pass ${pass}..."
  before=$(ssh -o BatchMode=yes mastersim-linux "ls -1 ~/r2-cap/out_lc/seq_*.pt 2>/dev/null | wc -l")
  ssh -o BatchMode=yes mastersim-linux "docker rm -f dspine-w6-cap >/dev/null 2>&1 || true; docker run --rm --name dspine-w6-cap --gpus all --network host --ipc host \
    -v ~/r2-cap:/work \
    -e PYTHONPATH=/work/exlr8-ref:/work/scripts:/work/pydeps \
    ${IMG} python3 -u /work/scripts/exlr8_lan_capture.py \
      --http-base ${HTTP_BASE} \
      --config-dir /work/cfg \
      --manifest /work/cfg/exlr8-manifest-bit280.json \
      --specs train:/work/labels/train_labels.jsonl:/work/out_lc \
      --skip-ids-file /work/labels/r2_skip_lc.txt \
      --min-seq-len 2049 --max-seq-len 16384 --save-last-n 4096 \
      --max-total-tokens 3200000 --vram-h-gb 42.0 --mlp-chunk 16384 \
      --cache-dir /work/lan-cache --tap-spill-dir /work/taps-lc"
  after=$(ssh -o BatchMode=yes mastersim-linux "ls -1 ~/r2-cap/out_lc/seq_*.pt 2>/dev/null | wc -l")
  log "Wave-6 LC resumed pass ${pass}: ${before} -> ${after} shards"
  if [ "$after" -eq "$before" ]; then
    break
  fi
done

log "Syncing Wave-6 LC shards back to FrankenNAS..."
rsync -a mastersim-linux:~/r2-cap/out_lc/ /var/tmp/dspine-shards-bit280-lc/train/
rsync -a mastersim-linux:~/r2-cap/out_lc/ "${DSPINE}/data/captures_bit280_lc/train/"
n_short=$(ls -1 /var/tmp/dspine-shards-bit280/train/seq_*.pt 2>/dev/null | wc -l)
n_lc=$(ls -1 /var/tmp/dspine-shards-bit280-lc/train/seq_*.pt 2>/dev/null | wc -l)
n_new_lc=$(ssh -o BatchMode=yes mastersim-linux "ls -1 ~/r2-cap/out_lc/seq_*.pt 2>/dev/null | wc -l")
log "Part 2/2 DONE: added ${n_new_lc} lc shards (total bit280_lc=${n_lc})"
prog "DATA RESULT wave6-lc complete on mastersim-pro6000: +${n_new_lc} new OUR-exlr8 bit280-lc shards (total=${n_lc}; combined bit280+mp=$((n_short + n_lc + 4000)))"
log "=== WAVE-6 ALL COMPLETE ==="
RSEOF
  chmod +x "$resume_script"
  nohup "$resume_script" >> "${SITE}/notes/wave6-lc-resume.log" 2>&1 </dev/null &
  disown
  log "Wave-6 LC resumed in background (log: ${SITE}/notes/wave6-lc-resume.log)"
}

cleanup() {
  local rc=$?
  log "Cleaning up drs-gpu-pilot container and staging dir (rc=${rc})..."
  ssh -o BatchMode=yes mastersim-linux "docker rm -f drs-gpu-pilot >/dev/null 2>&1 || true; docker run --rm -v ${W}:/s nvcr.io/nvidia/pytorch:26.07-py3 rm -rf /s/secrets /s/out /s/vocab /s/anchor /s/champion /s/eval /s/dumps /s/exlr8-ref /s/pydeps /s/cap_cfg /s/cap_in /s/scripts /s/worker >/dev/null 2>&1 || true; rm -rf ${W} || true" || true
  [ -n "$FS_PID" ] && kill "$FS_PID" >/dev/null 2>&1 || true
  [ -n "$HF_PID" ] && kill "$HF_PID" >/dev/null 2>&1 || true
  if [ "$BORROW_LOGGED" -eq 1 ]; then
    local note="- DRS NOTE borrow mastersim-pro6000 end gpu-pilot-batch (rc=${rc}, GPU freed) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    log_both_progress "$note"
  fi
  if [ "$PAUSED_WAVE6" -eq 1 ]; then
    resume_wave6_lc || true
    log_both_progress "- DRS NOTE resumed DSpine Wave-6 LC pass on mastersim-pro6000 after borrow $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

# 1. Check mastersim-pro6000 GPU; if Wave-6 LC pass 2 is in early layer sweep (not mid-shard-write), pause cleanly
PROCS=$(ssh -o BatchMode=yes mastersim-linux "nvidia-smi --query-compute-apps=pid --format=csv,noheader" | tr -d "[:space:]" || true)
if [ -n "$PROCS" ]; then
  W6_RUNNING=$(ssh -o BatchMode=yes mastersim-linux "docker ps --filter name=^/dspine-w6-cap$ --format '{{.Names}}'" || true)
  if [ "$W6_RUNNING" = "dspine-w6-cap" ]; then
    LAST_LOG=$(ssh -o BatchMode=yes mastersim-linux "docker logs --tail 3 dspine-w6-cap 2>&1" || true)
    if echo "$LAST_LOG" | grep -q "Computing bit280 top-20 logits\|saved "; then
      log "dspine-w6-cap is mid-shard-write! Waiting for shard write to finish before taking..."
      while ssh -o BatchMode=yes mastersim-linux "docker ps --filter name=^/dspine-w6-cap$ --format '{{.Names}}'" | grep -q dspine-w6-cap; do
        sleep 5
      done
    else
      log "Pausing DSpine Wave-6 LC pass cleanly (pre-shard-write; existing 692 shards preserved)..."
      # Stop outer run_wave6_mastersim.sh wrapper on FrankenNAS so it doesn't abort/relaunch mid-borrow
      pkill -f "run_wave6_mastersim.sh" || true
      pkill -f "ssh mastersim-linux docker rm -f dspine-w6-cap" || true
      ssh -o BatchMode=yes mastersim-linux "docker stop -t 10 dspine-w6-cap >/dev/null 2>&1 || docker rm -f dspine-w6-cap >/dev/null 2>&1 || true"
      # Clean only partial spill dir from aborted early pass (out_lc shards untouched!)
      ssh -o BatchMode=yes mastersim-linux "docker run --rm -v /home/bertholomus/r2-cap:/w nvcr.io/nvidia/pytorch:26.07-py3 rm -rf /w/taps-lc /w/lan-cache >/dev/null 2>&1 || true"
      PAUSED_WAVE6=1
    fi
  fi
fi

# Verify GPU has 0 compute processes
for _ in 1 2 3 4 5 6; do
  PROCS=$(ssh -o BatchMode=yes mastersim-linux "nvidia-smi --query-compute-apps=pid --format=csv,noheader" | tr -d "[:space:]" || true)
  [ -z "$PROCS" ] && break
  log "Waiting for GPU compute pids to clear (pids=${PROCS})..."
  sleep 3
done
if [ -n "$PROCS" ]; then
  log "ERROR: mastersim-pro6000 still has active compute pids=${PROCS}; aborting."
  exit 1
fi

BORROW_LOGGED=1
log_both_progress "- DRS NOTE borrow mastersim-pro6000 start gpu-pilot-batch (cap_bit280_gpu_fresh + 11 eval + 4w SparseLoCo + bridge + Byzantine gate + offline champion score) $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# 2. Ephemeral read-only LAN shard server (:19468) + LAN HF mirror (:19469)
python3 -u "${DSPINE}/scripts/dspine_lan_fileserver.py" 0.0.0.0 "$PORT" >"${SITE}/notes/drs-lan-fs.log" 2>&1 &
FS_PID=$!
python3 -u "${DSPINE}/scripts/lan_hf_mirror.py" 0.0.0.0 "$HF_PORT" >"${SITE}/notes/drs-lan-hf.log" 2>&1 &
HF_PID=$!
for _ in 1 2 3 4 5 6 7 8 9 10; do
  curl -sf "http://192.168.1.49:${PORT}/health" >/dev/null && curl -sf "http://192.168.1.49:${HF_PORT}/health" >/dev/null && break
  sleep 0.5
done
ssh -o BatchMode=yes mastersim-linux "curl -sf http://192.168.1.49:${PORT}/health >/dev/null && curl -sf http://192.168.1.49:${HF_PORT}/health >/dev/null"
log "Ephemeral LAN shard server (:${PORT}) and HF exlr8 mirror (:${HF_PORT}) healthy"

# 3. Stage minimal read-only copies + keys into mastersim-linux:~/drs-work (wiped on exit!)
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
ssh -o BatchMode=yes mastersim-linux "chmod 600 ${W}/secrets/* && df -h /"

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

if [ "$RUN_RC" -eq 0 ]; then
  log "Syncing aggregated checkpoint and offline eval evidence to ${SITE}/data/evals/..."
  mkdir -p "${SITE}/data/evals/pilot-r1-agg"
  rsync -a "mastersim-linux:${W}/out/pilot-r1-agg/" "${SITE}/data/evals/pilot-r1-agg/"
  rsync -a "mastersim-linux:${W}/out/gpu-pilot-evidence.json" "${SITE}/data/evals/gpu-pilot-evidence.json"
  log "ALL GPU PILOT ARTIFACTS SYNCED!"
fi
exit "$RUN_RC"
