#!/usr/bin/env bash
# B2 overnight chain (2026-11-09). Stops at the first problem and writes results/b2_chain_STOPPED.txt (nothing is worked around).
#  1. wait for the 12 U-Net runs -> check -> tests -> U-Net validation report -> commit
#  2. profile (idle P-cores): single process 4 vs 6 threads, and 3 concurrent jobs x 2 vs x 4 threads -> pick the faster 3-job setting
#  3. 6 ConvLSTM runs, 3 jobs, max 30 epochs
#  4. check -> full validation report -> commit. STOP (no test period, no push).
set -uo pipefail
cd /home/d1yordev/Desktop/Projects/sst-baselines
source .venv/bin/activate
export PYTHONPATH=src
LOG=results/b2_chain_log.txt
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
stop() { log "STOP: $*"; echo "$(date -u +%FT%TZ) $*" >> results/b2_chain_STOPPED.txt; pkill -P $$ 2>/dev/null; exit 1; }
commit() { git add "$@" && git commit -q -m "$MSG

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" || stop "git commit failed"; }

log "chain started; waiting for U-Net workers"
while pgrep -f "[m] sst.b2_train" > /dev/null; do sleep 60; done
log "no b2_train process left"
python -m sst.b2_check --model unet --expect 12 2>&1 | tee -a "$LOG"; [ "${PIPESTATUS[0]}" -eq 0 ] || stop "U-Net run check failed (see log)"
python -m pytest tests/test_b2.py -q -x 2>&1 | tail -n 5 | tee -a "$LOG"; [ "${PIPESTATUS[0]}" -eq 0 ] || stop "tests failed"
python -m sst.b2_manifest --models unet --tag _unet 2>&1 | tee -a "$LOG"; [ "${PIPESTATUS[0]}" -eq 0 ] || stop "manifest (unet) failed"
python -m sst.b2_eval --models unet --tag _unet 2>&1 | grep -v Warning | tee -a "$LOG"; [ "${PIPESTATUS[0]}" -eq 0 ] || stop "U-Net evaluation failed"
MSG="B2: U-Net validation report (12 runs, 2016-2019 only), manifest, curves, maps" commit results/b2_runs results/b2_train_log.txt results/b2_manifest_unet.json results/b2_val_unet.json results/b2_val_unet_report.md results/b2_val_maps_unet.npz results/figures/figB2_curves_unet.png results/figures/figB2_maps_unet.png results/b2_chain_log.txt
log "stage 1 done (U-Net report committed)"

# ---- stage 2: profile on idle P-cores ----
mkdir -p results/b2_profile
PJ=results/b2_profile/profile_P.jsonl
prof() { # cpus threads
  taskset -c "$1" python -m sst.b2_profile --threads "$2" --det 1 --batches 8 2>/dev/null; }
rm -f results/b2_profile/decision.json
log "profile: single process 4 threads"; prof 0,2,4,6 4 | tee -a "$PJ" | sed 's/^/[single4] /' >> "$LOG"
log "profile: single process 6 threads"; prof 0,2,4,6,8,10 6 | tee -a "$PJ" | sed 's/^/[single6] /' >> "$LOG"
log "profile: 3 concurrent jobs x 2 threads"
for c in 0,2 4,6 8,10; do (prof $c 2 | sed 's/^/{"conc":"3x2"} /' >> "$PJ") & done; wait
log "profile: 3 concurrent jobs x 4 threads"
for c in 0-3 4-7 8-11; do (prof $c 4 | sed 's/^/{"conc":"3x4"} /' >> "$PJ") & done; wait
python - <<'PY' 2>&1 | tee -a "$LOG"
import json, re
rows = []
for l in open("results/b2_profile/profile_P.jsonl"):
    l = l.strip()
    conc = None
    m = re.match(r'^\{"conc":"(\w+)"\} (\{.*)$', l)
    if m: conc, l = m.group(1), m.group(2)
    rows.append((conc, json.loads(l)))
# aggregate throughput (batches/s summed over the concurrent processes) per concurrent setting; the last 3 lines of each setting
out = {}
for key in ("3x2", "3x4"):
    t = [r["sec_per_batch"]["total"] for c, r in rows if c == key][-3:]
    assert len(t) == 3, key
    out[key] = {"sec_per_batch": t, "agg_batches_per_s": sum(1 / x for x in t)}
single = {r["threads"]: r["sec_per_batch"]["total"] for c, r in rows if c is None and r["deterministic"]}
out["single_process_sec_per_batch"] = {str(k): v for k, v in single.items()}
best = max(("3x2", "3x4"), key=lambda k: out[k]["agg_batches_per_s"])
out["chosen"] = {"setting": best, "threads_per_job": int(best[-1]), "cpus": ["0,2", "4,6", "8,10"] if best == "3x2" else ["0-3", "4-7", "8-11"],
                 "rule": "highest aggregate throughput (sum of 1/sec_per_batch) of 3 concurrent jobs; single-process 4 vs 6 threads reported for information"}
json.dump(out, open("results/b2_profile/decision.json", "w"), indent=1)
print(json.dumps(out, indent=1))
PY
THREADS=$(python -c "import json;print(json.load(open('results/b2_profile/decision.json'))['chosen']['threads_per_job'])") || stop "profile decision failed"
if [ "$THREADS" = "2" ]; then CPUS=("0,2" "4,6" "8,10"); else CPUS=("0-3" "4-7" "8-11"); fi
log "profile done; ConvLSTM jobs: 3 x $THREADS threads on cpus ${CPUS[*]}"
NOTE="
## Chain note $(date -u +%FT%TZ)
Profile on idle P-cores (results/b2_profile/decision.json): ConvLSTM jobs run as 3 concurrent processes x $THREADS threads on cpus ${CPUS[*]}. Timing is not a result."
echo "$NOTE" >> docs/b2_protocol.md; echo "$NOTE" >> notes/2026-11-09.md
MSG="B2: profile on idle P-cores and 3-job thread decision (timing is not a result)" commit results/b2_profile results/b2_chain_log.txt docs/b2_protocol.md

# ---- stage 3: ConvLSTM, 3 workers x 2 runs ----
JOBS=("kuroshio 0 kuroshio 1" "kuroshio 2 subtropical 0" "subtropical 1 subtropical 2")
PIDS=()
for i in 0 1 2; do
  set -- ${JOBS[$i]}
  ( for pair in "$1 $2" "$3 $4"; do set -- $pair
      taskset -c "${CPUS[$i]}" python -m sst.b2_train --model convlstm_ed --region "$1" --C 16 --seed "$2" --threads "$THREADS" >> results/b2_logs/convlstm_ed_w$i.log 2>&1 || exit 1
    done ) &
  PIDS+=($!)
done
log "ConvLSTM workers started: ${PIDS[*]}"
FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || { FAIL=1; pkill -f "[m] sst.b2_train --model convlstm_ed"; }; done
[ "$FAIL" -eq 0 ] || stop "a ConvLSTM worker failed (see results/b2_logs/convlstm_ed_w*.log)"
python -m sst.b2_check --model convlstm_ed --expect 6 2>&1 | tee -a "$LOG"; [ "${PIPESTATUS[0]}" -eq 0 ] || stop "ConvLSTM run check failed"

# ---- stage 4: full report ----
python -m sst.b2_manifest --models unet,convlstm_ed --tag "" 2>&1 | tee -a "$LOG"; [ "${PIPESTATUS[0]}" -eq 0 ] || stop "manifest failed"
python -m sst.b2_eval --models unet,convlstm_ed --tag "" 2>&1 | grep -v Warning | tee -a "$LOG"; [ "${PIPESTATUS[0]}" -eq 0 ] || stop "full evaluation failed"
MSG="B2: full validation report (U-Net + ConvLSTM, 2016-2019 only), manifest, curves, skill maps" commit results/b2_runs results/b2_train_log.txt results/b2_manifest.json results/b2_val.json results/b2_val_report.md results/b2_val_maps.npz results/figures/figB2_curves.png results/figures/figB2_maps.png results/b2_chain_log.txt
log "CHAIN DONE. Validation report committed. STOP: test period untouched, nothing pushed."
echo "$(date -u +%FT%TZ) chain finished OK" > results/b2_chain_DONE.txt
