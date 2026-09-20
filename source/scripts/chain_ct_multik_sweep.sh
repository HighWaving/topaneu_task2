#!/usr/bin/env bash
# 2026-08-17, peer-directed operating-point sweep, CT. Runs INDEPENDENTLY of
# chain_ct_baseline.sh (which is mid-flight on the old single-k=20 pass --
# NOT interrupted, per explicit instruction that pass is a valid data point
# and must be allowed to finish). CT's detector boxes are already fully on
# disk (109/109, from the same inference run) so this sweep needs NO GPU at
# all -- pure CPU/classifier work, safe to run concurrently with anything.
set -uo pipefail

ROOT=/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2
PY=/home/jovyan/rtx4claude-datavol-1/conda_envs/topcow_claim_official/bin/python
LOG=$ROOT/logs/chain_ct_multik_sweep.log
HANDOFF=$ROOT/HANDOFF_20260815.md
mkdir -p "$ROOT/logs" "$ROOT/reports"

log() { echo "[$(date -u +%H:%M:%S)] $1" | tee -a "$LOG"; }

CT_BOXES_DIR=$ROOT/artifacts/ct_fold2_all109_boxes
CT_ALL_IDS_JSON=$ROOT/artifacts/ct_all_case_ids.json
TA36_CT_OUT=$ROOT/artifacts/ta36_ct_output
K_VALUES="1,2,3,5,10,20"

log "=== CT multi-k sweep start: k=$K_VALUES, reusing existing 109/109 boxes ==="

$PY scripts/build_vessel_baseline_masks.py --boxes-dir "$CT_BOXES_DIR" --vessel-mask-dir "$TA36_CT_OUT" \
  --output-dir "$ROOT/artifacts/ct_baseline_masks_predicted_ta36_sweep" \
  --held-out-center center2 --held-out-center center4 --k-values "$K_VALUES" \
  >> "$ROOT/logs/chain_ct_multik_maskbuild_predicted.log" 2>&1
log "predicted-vessel multi-k masks built"

$PY scripts/build_vessel_baseline_masks.py --boxes-dir "$CT_BOXES_DIR" \
  --vessel-mask-dir "/home/jovyan/rtx4claude-datavol-1/data_topaneu26/vessel_masks" \
  --output-dir "$ROOT/artifacts/ct_baseline_masks_oracle_gt_sweep" \
  --held-out-center center2 --held-out-center center4 --k-values "$K_VALUES" \
  >> "$ROOT/logs/chain_ct_multik_maskbuild_oracle.log" 2>&1
log "oracle-vessel multi-k masks built"

for K in 1 2 3 5 10 20; do
  KDIR=$(printf "k%02d" "$K")
  $PY scripts/score_m1.py \
    --predictions-dir "$ROOT/artifacts/ct_baseline_masks_predicted_ta36_sweep/$KDIR" \
    --case-ids-json "$CT_ALL_IDS_JSON" \
    --output "$ROOT/reports/ct_baseline_score_predicted_ta36_all109_${KDIR}.json" \
    >> "$ROOT/logs/chain_ct_multik_score.log" 2>&1
  $PY scripts/score_m1.py \
    --predictions-dir "$ROOT/artifacts/ct_baseline_masks_oracle_gt_sweep/$KDIR" \
    --case-ids-json "$CT_ALL_IDS_JSON" \
    --output "$ROOT/reports/ct_baseline_score_oracle_gt_all109_${KDIR}.json" \
    >> "$ROOT/logs/chain_ct_multik_score.log" 2>&1
  log "scored k=$K (predicted + oracle, all 109 cases)"
done

log "=== CT multi-k sweep complete ==="
{
  echo ""
  echo "## CT operating-point sweep complete ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
  echo "k in {1,2,3,5,10,20} boxes/case (score>=0.3), predicted(TA36)/oracle(GT), all 109 cases."
  echo "**Every number below must be read together with its k -- do not report a bare metric without stating the operating point (peer's own instruction, since CT's ~1.2 true lesions/case means k=20 was ~16x over-prediction and badly dilutes DICE/HD95/VOLSIM's TP+FN+FP denominator)."
  echo "See reports/ct_baseline_score_{predicted_ta36,oracle_gt}_all109_k{01,02,03,05,10,20}.json."
} >> "$HANDOFF"
echo "CT_MULTIK_SWEEP_DONE" >> "$ROOT/logs/ct_multik_sweep.marker"
