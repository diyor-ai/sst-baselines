#!/usr/bin/env bash
# B2 residual-variant chain (2026-11-10). Stops at the first problem and writes results/b2_chain_res_STOPPED.txt (nothing is worked around).
#  1. tests; 2. 12 residual runs (3 pinned workers x [2 resunet, 2 resconvlstm_ed]); 3. checks; 4. manifest + evaluation (plain vs residual) -> commit. STOP.
set -uo pipefail
cd /home/d1yordev/Desktop/Projects/sst-baselines
source .venv/bin/activate
export PYTHONPATH=src
LOG=results/b2_chain_res_log.txt
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
stop() { log "STOP: $*"; echo "$(date -u +%FT%TZ) $*" >> results/b2_chain_res_STOPPED.txt; pkill -P $$ 2>/dev/null; exit 1; }
commit() { git add "$@" && git commit -q -m "$MSG

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" || stop "git commit failed"; }

log "residual chain started"
[ -z "$(git status --porcelain -- src/sst/b2_data.py src/sst/b2_models.py src/sst/b2_train.py config_b2.yaml docs/b2_protocol.md tests/test_b2.py)" ] || stop "code/protocol not committed before training"
python -m pytest tests/test_b2.py -q -x 2>&1 | tail -n 5 | tee -a "$LOG"; [ "${PIPESTATUS[0]}" -eq 0 ] || stop "tests failed"
mkdir -p results/b2_logs
CPUS=("0-3" "4-7" "8-11")
PIDS=()
for i in 0 1 2; do
  ( for job in "resunet kuroshio 32 3" "resunet subtropical 32 3" "resconvlstm_ed kuroshio 16 4" "resconvlstm_ed subtropical 16 4"; do set -- $job
      taskset -c "${CPUS[$i]}" python -m sst.b2_train --model "$1" --region "$2" --C "$3" --seed "$i" --threads "$4" >> results/b2_logs/res_w$i.log 2>&1 || exit 1
    done ) &
  PIDS+=($!)
done
log "residual workers started: ${PIDS[*]}"
FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || { FAIL=1; pkill -f "[m] sst.b2_train --model res"; }; done
[ "$FAIL" -eq 0 ] || stop "a residual worker failed (see results/b2_logs/res_w*.log)"
for m in resunet resconvlstm_ed; do
  python -m sst.b2_check --model $m --expect 6 2>&1 | tee -a "$LOG"; [ "${PIPESTATUS[0]}" -eq 0 ] || stop "$m run check failed"
done
python -m sst.b2_manifest --models resunet,resconvlstm_ed --tag _res 2>&1 | tee -a "$LOG"; [ "${PIPESTATUS[0]}" -eq 0 ] || stop "manifest failed"
python -m sst.b2_eval_res 2>&1 | grep -v Warning | tee -a "$LOG"; [ "${PIPESTATUS[0]}" -eq 0 ] || stop "evaluation failed"
MSG="B2: residual variants (U-Net C=32, ConvLSTM-ED C=16; 12 runs), plain vs residual validation report 2016-2019, manifest, curves, maps" commit results/b2_runs results/b2_train_log.txt results/b2_manifest_res.json results/b2_val_res.json results/b2_val_res_report.md results/b2_val_maps_res.npz results/figures/figB2_curves_res.png results/figures/figB2_maps_res.png results/b2_chain_res_log.txt
log "RESIDUAL CHAIN DONE. Report committed. STOP: test period untouched, nothing pushed."
echo "$(date -u +%FT%TZ) residual chain finished OK" > results/b2_chain_res_DONE.txt
