#!/usr/bin/env bash
# Phase A (GPU3 sequential): wait for the already-running CT TA36 batch (PID
# owned by peer, NOT launched here) -> CT fold2 val-10 detector inference ->
# CT honest baseline (predicted TA36 vessel + oracle GT vessel, side by
# side) -> launch MR TA36 on center2's 40 cases -> wait for it to finish.
# 2026-08-17, peer-directed auto-chain. Does NOT touch GPU0/1/2.
set -uo pipefail

ROOT=/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2
PY=/home/jovyan/rtx4claude-datavol-1/conda_envs/topcow_claim_official/bin/python
PY_NNDET=/home/jovyan/rtx4claude-datavol-1/conda_envs/nndet/bin/python
LOG=$ROOT/logs/chain_ct_baseline.log
HANDOFF=$ROOT/HANDOFF_20260815.md
mkdir -p "$ROOT/logs" "$ROOT/reports"

log() { echo "[$(date -u +%H:%M:%S)] $1" | tee -a "$LOG"; }

log "=== Phase A start: waiting for CT TA36 batch (GPU3, 109 cases) to finish ==="
TA36_CT_OUT=$ROOT/artifacts/ta36_ct_output
while true; do
  n=$(ls "$TA36_CT_OUT"/*.nii.gz 2>/dev/null | wc -l)
  if [ "$n" -ge 109 ]; then break; fi
  if ! pgrep -f "run_inference.py.*ta36_ct_input_all" >/dev/null 2>&1; then
    n=$(ls "$TA36_CT_OUT"/*.nii.gz 2>/dev/null | wc -l)
    if [ "$n" -lt 109 ]; then
      log "*** CT TA36 PROCESS DIED EARLY: only $n/109 outputs -- ABORTING, needs human attention ***"
      echo "CT_TA36_FAILED n=$n" >> "$ROOT/logs/chain_ct_baseline.marker"
      exit 1
    fi
    break
  fi
  sleep 180
done
log "CT TA36 done: $n/109 vessel masks in $TA36_CT_OUT"

# --- 2026-08-17 peer correction: run detector inference on ALL 109 CT cases,
# not just fold2's 10-case val. The detector itself is optimistic on 99/109
# (seen during fold2 training) -- so DETECTION diagnostics (sensitivity/FN/
# FP/candidates) are still reported ONLY on the clean 10-case val subset,
# small-N caveat and all. But the vessel-geometric classifier is rule-based
# and has seen ZERO of the 109 CT cases regardless of detector training --
# so CLASSIFICATION accuracy, coverage, and the official 6 metrics are
# reported on the FULL 109 cases (10x the sample, and the honest number for
# the part of the pipeline that's actually the current bottleneck).
CT_VAL_IDS_JSON=$ROOT/artifacts/ct_fold2_val10_ids.json
$PY -c "
import json
ids = ['topaneu_center2_ct_125','topaneu_center2_ct_162','topaneu_center2_ct_173',
       'topaneu_center2_ct_175','topaneu_center2_ct_190','topaneu_center4_ct_048',
       'topaneu_center4_ct_051_2','topaneu_center4_ct_064','topaneu_center4_ct_072',
       'topaneu_center4_ct_201']
json.dump(ids, open('$CT_VAL_IDS_JSON', 'w'), indent=2)
"
log "CT fold2 val-10 case ids (detection-diagnostics-only subset) written to $CT_VAL_IDS_JSON"

CT_ALL_IDS_JSON=$ROOT/artifacts/ct_all_case_ids.json
CT_BOXES_DIR=$ROOT/artifacts/ct_fold2_all109_boxes
mkdir -p "$CT_BOXES_DIR"
cd /home/jovyan/rtx4claude-datavol-1/new_aneurysms
export PYTHONPATH="/home/jovyan/rtx4claude-datavol-1/new_aneurysms"
export det_data="/home/jovyan/rtx4claude-datavol-1/nndet_data"
export det_models="/home/jovyan/rtx4claude-datavol-1/nndet_models"
export OMP_NUM_THREADS=1
CASE_ID_ARGS=$($PY -c "print(' '.join(f'--case-id {i}' for i in __import__('json').load(open('$CT_ALL_IDS_JSON'))))")
log "running CT fold2 detector inference on all 109 cases (this is the slow step, ~2-3h) ..."
CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 $PY_NNDET \
  src/scripts/infer_nndet_adam.py \
  --training-dir "$ROOT/artifacts/checkpoint_safety_backups/ct_fold2_FINAL_20260816T183639" \
  --source-dir "/home/jovyan/rtx4claude-datavol-1/nndet_data/Task031FG_TopAneuCT/preprocessed/D3V001_3d/imagesTr" \
  --output-dir "$CT_BOXES_DIR" \
  --checkpoint last --device cuda:0 --num-tta 1 --batch-size 1 --max-detections 200 \
  $CASE_ID_ARGS >> "$ROOT/logs/chain_ct_all109_inference.log" 2>&1

N=$(ls "$CT_BOXES_DIR"/*_boxes.pkl 2>/dev/null | wc -l)
log "CT all-109 inference done: $N/109 boxes"
if [ "$N" -lt 109 ]; then
  log "*** CT INFERENCE INCOMPLETE: $N/109, ABORTING ***"
  echo "CT_BASELINE_FAILED n=$N" >> "$ROOT/logs/chain_ct_baseline.marker"
  exit 1
fi

# --- build masks for all 109: predicted (TA36) and oracle (GT), side by side ---
cd "$ROOT"
$PY scripts/build_vessel_baseline_masks.py --boxes-dir "$CT_BOXES_DIR" --vessel-mask-dir "$TA36_CT_OUT" \
  --output-dir "$ROOT/artifacts/ct_baseline_masks_predicted_ta36" \
  --held-out-center center2 --held-out-center center4 \
  >> "$ROOT/logs/chain_ct_maskbuild_predicted.log" 2>&1

$PY scripts/build_vessel_baseline_masks.py --boxes-dir "$CT_BOXES_DIR" \
  --vessel-mask-dir "/home/jovyan/rtx4claude-datavol-1/data_topaneu26/vessel_masks" \
  --output-dir "$ROOT/artifacts/ct_baseline_masks_oracle_gt" \
  --held-out-center center2 --held-out-center center4 \
  >> "$ROOT/logs/chain_ct_maskbuild_oracle.log" 2>&1

log "CT masks built for all 109 cases (predicted_ta36 + oracle_gt)"

# DETECTION diagnostics: 10-case clean val ONLY (small-N, wide error bars)
$PY scripts/m1_lesion_diagnostics.py --boxes-dir "$CT_BOXES_DIR" --case-ids-json "$CT_VAL_IDS_JSON" \
  --output "$ROOT/reports/ct_baseline_lesion_diagnostics_val10.json" \
  >> "$ROOT/logs/chain_ct_lesion_diag.log" 2>&1

# CLASSIFICATION + official 6 metrics + coverage: all 109 (detector-optimistic, classifier-clean)
$PY scripts/score_m1.py --predictions-dir "$ROOT/artifacts/ct_baseline_masks_predicted_ta36" \
  --case-ids-json "$CT_ALL_IDS_JSON" \
  --output "$ROOT/reports/ct_baseline_score_predicted_ta36_all109.json" \
  >> "$ROOT/logs/chain_ct_score_predicted.log" 2>&1

$PY scripts/score_m1.py --predictions-dir "$ROOT/artifacts/ct_baseline_masks_oracle_gt" \
  --case-ids-json "$CT_ALL_IDS_JSON" \
  --output "$ROOT/reports/ct_baseline_score_oracle_gt_all109.json" \
  >> "$ROOT/logs/chain_ct_score_oracle.log" 2>&1

# Class-assignment accuracy with single/junction/position group breakdown, all 109
$PY scripts/class_assignment_diagnostics.py --boxes-dir "$CT_BOXES_DIR" \
  --vessel-mask-dir "$TA36_CT_OUT" --case-ids-json "$CT_ALL_IDS_JSON" \
  --held-out-center center2 --held-out-center center4 --label predicted_ta36 \
  --output "$ROOT/reports/ct_class_assignment_predicted_ta36_all109.json" \
  >> "$ROOT/logs/chain_ct_class_assign_predicted.log" 2>&1

$PY scripts/class_assignment_diagnostics.py --boxes-dir "$CT_BOXES_DIR" \
  --vessel-mask-dir "/home/jovyan/rtx4claude-datavol-1/data_topaneu26/vessel_masks" \
  --case-ids-json "$CT_ALL_IDS_JSON" \
  --held-out-center center2 --held-out-center center4 --label oracle_gt \
  --output "$ROOT/reports/ct_class_assignment_oracle_gt_all109.json" \
  >> "$ROOT/logs/chain_ct_class_assign_oracle.log" 2>&1

log "CT honest baseline complete: detection diagnostics on val-10 (small-N caveat), classification+official6+coverage+group-breakdown on all 109"
{
  echo ""
  echo "## CT baseline auto-chain: stage 1 complete ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
  echo "**Detector is optimistic** on this set (fold2 trained on 99/109 CT cases) -- detection metrics (sensitivity/FN/FP/candidates) reported ONLY on the clean 10-case val subset (reports/ct_baseline_lesion_diagnostics_val10.json), n=10 lesions, wide error bars, direction-only."
  echo "**Classifier is rule-based, has seen 0/109 CT cases** regardless of detector training -- classification accuracy (reports/ct_class_assignment_{predicted_ta36,oracle_gt}_all109.json, with single/junction/position group breakdown), coverage, and the official 6 metrics (reports/ct_baseline_score_{predicted_ta36,oracle_gt}_all109.json) reported on all 109 cases."
  echo "Do not compare ct_class_assignment_*.json numbers directly against phase2b_geometric_attachment.py's geometric_attachment_results.md -- different measurement (novel class guess from a DETECTED box vs. correctness-check against a KNOWN class from the GT lesion mask), see class_assignment_diagnostics.py's docstring."
} >> "$HANDOFF"
echo "CT_BASELINE_DONE" >> "$ROOT/logs/chain_ct_baseline.marker"

# --- launch MR TA36 on center2's 40 cases (same GPU3, sequential, not parallel with CT) ---
log "=== launching MR TA36 (center2, 40 cases, GPU3) ==="
mkdir -p "$ROOT/artifacts/ta36_mr_center2_input" "$ROOT/artifacts/ta36_mr_center2_output"
$PY -c "
import json, shutil
m = json.loads(open('/home/jovyan/rtx4claude-datavol-1/topaneu2026_task1/artifacts/phase0/case_manifest.json').read())
cases_by_id = {c['case_id']: c for c in m['cases']}
ids = json.loads(open('$ROOT/artifacts/m1_center2_mr_case_ids.json').read())
for cid in ids:
    shutil.copy(cases_by_id[cid]['image'], f'$ROOT/artifacts/ta36_mr_center2_input/{cid}_0000.nii.gz')
print(f'copied {len(ids)} MR center2 files for TA36')
" >> "$LOG" 2>&1

R=/home/jovyan/ta36_rootfs
SP=$R/app/nnUNet/.venv/lib/python3.12/site-packages
cd "$R/app/nnUNet"
setsid env PYTHONPATH="$SP:$R/app/nnUNet" \
  CUDA_VISIBLE_DEVICES=3 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nnUNet_n_proc_DA=1 \
  "$R/usr/local/bin/python3.12" run_inference.py \
  -i "$ROOT/artifacts/ta36_mr_center2_input" \
  -o "$ROOT/artifacts/ta36_mr_center2_output" \
  --suffix _0000.nii.gz --n_gpus 1 \
  > "$ROOT/logs/ta36_mr_center2_gpu3.log" 2>&1 < /dev/null &
disown
log "MR TA36 launched, pid $!"

while true; do
  n=$(ls "$ROOT/artifacts/ta36_mr_center2_output"/*.nii.gz 2>/dev/null | wc -l)
  if [ "$n" -ge 40 ]; then break; fi
  if ! pgrep -f "run_inference.py.*ta36_mr_center2_input" >/dev/null 2>&1; then
    n=$(ls "$ROOT/artifacts/ta36_mr_center2_output"/*.nii.gz 2>/dev/null | wc -l)
    if [ "$n" -lt 40 ]; then
      log "*** MR TA36 PROCESS DIED EARLY: only $n/40 outputs -- ABORTING, needs human attention ***"
      echo "MR_TA36_FAILED n=$n" >> "$ROOT/logs/mr_ta36.marker"
      exit 1
    fi
    break
  fi
  sleep 180
done
log "MR TA36 done: $n/40 vessel masks"
echo "MR_TA36_DONE n=$n" >> "$ROOT/logs/mr_ta36.marker"
{
  echo ""
  echo "## MR TA36 vessel-mask inference complete ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
  echo "40/40 center2 MR cases -> $ROOT/artifacts/ta36_mr_center2_output/. Waiting on MR fold1 detector boxes (separate chain) to build the MR baseline."
} >> "$HANDOFF"

log "=== Phase A complete ==="
