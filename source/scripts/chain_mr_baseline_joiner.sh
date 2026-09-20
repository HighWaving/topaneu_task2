#!/usr/bin/env bash
# Joins Phase A (MR TA36 vessel masks) and Phase B (fold1 detector boxes on
# center2) once BOTH are done, builds the MR honest baseline (predicted +
# oracle vessel, side by side), scores it, writes the final HANDOFF summary.
# 2026-08-17, peer-directed auto-chain.
set -uo pipefail

ROOT=/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2
PY=/home/jovyan/rtx4claude-datavol-1/conda_envs/topcow_claim_official/bin/python
LOG=$ROOT/logs/chain_mr_baseline_joiner.log
HANDOFF=$ROOT/HANDOFF_20260815.md
mkdir -p "$ROOT/logs" "$ROOT/reports"

log() { echo "[$(date -u +%H:%M:%S)] $1" | tee -a "$LOG"; }

log "=== joiner: waiting for MR TA36 (Phase A) AND fold1 boxes (Phase B) ==="
while true; do
  if grep -q "FAILED" "$ROOT/logs/mr_ta36.marker" 2>/dev/null || \
     grep -q "FAILED" "$ROOT/logs/mr_fold1_boxes.marker" 2>/dev/null; then
    log "*** an upstream stage failed (see mr_ta36.marker / mr_fold1_boxes.marker) -- aborting joiner ***"
    exit 1
  fi
  ta36_done=0; boxes_done=0
  grep -q "MR_TA36_DONE" "$ROOT/logs/mr_ta36.marker" 2>/dev/null && ta36_done=1
  grep -q "MR_FOLD1_BOXES_DONE" "$ROOT/logs/mr_fold1_boxes.marker" 2>/dev/null && boxes_done=1
  if [ "$ta36_done" -eq 1 ] && [ "$boxes_done" -eq 1 ]; then break; fi
  sleep 180
done
log "both prerequisites ready -- building MR honest baseline"

cd "$ROOT"
CASE_IDS_JSON=$ROOT/artifacts/m1_center2_mr_case_ids.json
FOLD1_BOXES_DIR=$ROOT/artifacts/mr_fold1_final_center2_boxes
K_VALUES="1,2,3,5,10,20"

# 2026-08-17, peer-directed: MR gets its own operating-point sweep too
# (nnDetection scores aren't calibrated, MR's own point must not reuse CT's
# 0.3/20 threshold uncritically even though M1 showed it was less saturated
# there -- mean 2.98 candidates/case at 0.3, not CT's 18.77).
$PY scripts/build_vessel_baseline_masks.py --boxes-dir "$FOLD1_BOXES_DIR" \
  --vessel-mask-dir "$ROOT/artifacts/ta36_mr_center2_output" \
  --output-dir "$ROOT/artifacts/mr_baseline_masks_predicted_ta36_sweep" \
  --held-out-center center2 --k-values "$K_VALUES" \
  >> "$ROOT/logs/chain_mr_maskbuild_predicted.log" 2>&1

$PY scripts/build_vessel_baseline_masks.py --boxes-dir "$FOLD1_BOXES_DIR" \
  --vessel-mask-dir "/home/jovyan/rtx4claude-datavol-1/data_topaneu26/vessel_masks" \
  --output-dir "$ROOT/artifacts/mr_baseline_masks_oracle_gt_sweep" \
  --held-out-center center2 --k-values "$K_VALUES" \
  >> "$ROOT/logs/chain_mr_maskbuild_oracle.log" 2>&1

log "MR multi-k masks built (predicted_ta36 + oracle_gt, k=$K_VALUES)"

$PY scripts/m1_lesion_diagnostics.py --boxes-dir "$FOLD1_BOXES_DIR" --case-ids-json "$CASE_IDS_JSON" \
  --output "$ROOT/reports/mr_baseline_lesion_diagnostics.json" \
  >> "$ROOT/logs/chain_mr_lesion_diag.log" 2>&1

for K in 1 2 3 5 10 20; do
  KDIR=$(printf "k%02d" "$K")
  $PY scripts/score_m1.py --predictions-dir "$ROOT/artifacts/mr_baseline_masks_predicted_ta36_sweep/$KDIR" \
    --case-ids-json "$CASE_IDS_JSON" \
    --output "$ROOT/reports/mr_baseline_score_predicted_ta36_${KDIR}.json" \
    >> "$ROOT/logs/chain_mr_score_predicted.log" 2>&1
  $PY scripts/score_m1.py --predictions-dir "$ROOT/artifacts/mr_baseline_masks_oracle_gt_sweep/$KDIR" \
    --case-ids-json "$CASE_IDS_JSON" \
    --output "$ROOT/reports/mr_baseline_score_oracle_gt_${KDIR}.json" \
    >> "$ROOT/logs/chain_mr_score_oracle.log" 2>&1
  log "MR scored k=$K (predicted + oracle)"
done

# Class-assignment accuracy with single/junction/position group breakdown
# (same peer-requested check as CT: does the tie-break rule systematically
# favor single-segment classes over the 20 junction classes)
$PY scripts/class_assignment_diagnostics.py --boxes-dir "$FOLD1_BOXES_DIR" \
  --vessel-mask-dir "$ROOT/artifacts/ta36_mr_center2_output" --case-ids-json "$CASE_IDS_JSON" \
  --held-out-center center2 --label predicted_ta36 \
  --output "$ROOT/reports/mr_class_assignment_predicted_ta36.json" \
  >> "$ROOT/logs/chain_mr_class_assign_predicted.log" 2>&1

$PY scripts/class_assignment_diagnostics.py --boxes-dir "$FOLD1_BOXES_DIR" \
  --vessel-mask-dir "/home/jovyan/rtx4claude-datavol-1/data_topaneu26/vessel_masks" \
  --case-ids-json "$CASE_IDS_JSON" \
  --held-out-center center2 --label oracle_gt \
  --output "$ROOT/reports/mr_class_assignment_oracle_gt.json" \
  >> "$ROOT/logs/chain_mr_class_assign_oracle.log" 2>&1

log "MR honest baseline complete (predicted + oracle scored across k=$K_VALUES, lesion diagnostics + class-assignment group breakdown written)"
{
  echo ""
  echo "## MR baseline auto-chain: complete ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
  echo "fold1 FINAL checkpoint x center2 (40 cases) x predicted(TA36)/oracle(GT) vessel classification x official scorer, k in {1,2,3,5,10,20} boxes/case (score>=0.3)."
  echo "**Every number must be read with its k -- do not report a bare metric without the operating point.**"
  echo "See reports/mr_baseline_score_{predicted_ta36,oracle_gt}_k{01,02,03,05,10,20}.json, reports/mr_baseline_lesion_diagnostics.json."
} >> "$HANDOFF"

echo "MR_BASELINE_DONE" >> "$ROOT/logs/mr_baseline.marker"
echo "FULL_CHAIN_DONE" >> "$ROOT/logs/full_chain.marker"
log "=== FULL CHAIN DONE ==="
