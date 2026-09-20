#!/usr/bin/env bash
# Polls fold1's train.log for new epoch checkpoints; every 5 epochs, snapshots
# the checkpoint, runs inference on center2 (GPU2), and computes pure
# box-vs-GT recall -- builds a real-model learning curve automatically so
# nobody has to remember to trigger it. 2026-08-14, peer-requested.
set -uo pipefail

TRAIN_LOG="/home/jovyan/rtx4claude-datavol-1/nndet_models/Task030FG_TopAneuMR/RetinaUNetV001_D3V001_3d/fold1/train.log"
LAST_EVALED_EPOCH=-1

count_completed_epochs() {
  grep -c "validation_epoch_end:268" "$TRAIN_LOG" 2>/dev/null || echo 0
}

while true; do
  n_epochs=$(count_completed_epochs)
  current_epoch=$((n_epochs - 1))
  if [ "$current_epoch" -ge 0 ] && [ $((current_epoch % 5)) -eq 0 ] && [ "$current_epoch" -ne "$LAST_EVALED_EPOCH" ]; then
    echo "[$(date -u +%H:%M:%S)] triggering eval at epoch $current_epoch" \
      >> /home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/logs/fold1_eval_loop.log
    for CENTER in center2 center1; do
      # center2: fold1's held-out (clean generalization estimate)
      # center1: IN fold1's training set (quantifies train-vs-held-out gap, i.e. overfitting)
      bash /home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/scripts/eval_fold1_checkpoint_on_center2.sh \
        "epoch${current_epoch}" "$CENTER" \
        >> /home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/logs/fold1_eval_loop.log 2>&1
      /home/jovyan/rtx4claude-datavol-1/conda_envs/topcow_claim_official/bin/python \
        /home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/scripts/probe_pure_box_recall.py \
        --center "$CENTER" \
        --boxes-dir "/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/artifacts/fold1_eval_${CENTER}_epoch${current_epoch}" \
        --label "fold1_epoch${current_epoch}_${CENTER}" \
        >> /home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/logs/fold1_eval_loop.log 2>&1
    done
    echo "[$(date -u +%H:%M:%S)] eval at epoch $current_epoch done (center2 + center1)" \
      >> /home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/logs/fold1_eval_loop.log
    LAST_EVALED_EPOCH=$current_epoch

    # 2026-08-15, peer-requested falsifiable checkpoint: by epoch 20-25,
    # center1 (fold1's OWN training set) K=1 must clearly exceed the
    # zero-shot Task020FG baseline (47.8%) -- if training data itself
    # can't beat transfer learning by then, that's not "needs more time",
    # it's a real problem (LR, data conversion, label format) worth
    # stopping to investigate, not epoch 50.
    if [ "$current_epoch" -ge 20 ] && [ "$current_epoch" -le 25 ]; then
      RECALL_JSON="/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/reports/pure_box_recall_fold1_epoch${current_epoch}_center1.json"
      if [ -f "$RECALL_JSON" ]; then
        K1=$(/home/jovyan/rtx4claude-datavol-1/conda_envs/topcow_claim_official/bin/python -c "
import json
d = json.loads(open('$RECALL_JSON').read())
print(d['by_k']['1']['recall'])
")
        ABOVE=$(/home/jovyan/rtx4claude-datavol-1/conda_envs/topcow_claim_official/bin/python -c "print(1 if $K1 > 0.478 else 0)")
        if [ "$ABOVE" -eq 1 ]; then
          echo "[$(date -u +%H:%M:%S)] FALSIFIABLE CHECKPOINT epoch $current_epoch: center1 K=1 = $K1 > 0.478 (zero-shot baseline) -- PASS, training progressing normally" \
            >> /home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/logs/fold1_eval_loop.log
        else
          echo "[$(date -u +%H:%M:%S)] *** FALSIFIABLE CHECKPOINT ALERT *** epoch $current_epoch: center1 K=1 = $K1 <= 0.478 (zero-shot baseline) -- training set STILL not beating transfer baseline. Investigate LR/data/labels, do not assume 'needs more epochs'." \
            | tee -a /home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/logs/fold1_eval_loop.log \
            /home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/logs/FALSIFIABLE_CHECKPOINT_ALERT.log
        fi
      fi
    fi
  fi
  if ! kill -0 "$(pgrep -f 'nndet_train Task030FG_TopAneuMR -o exp.fold=1' | head -1)" 2>/dev/null; then
    echo "[$(date -u +%H:%M:%S)] fold1 training process no longer found, stopping poll loop" \
      >> /home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/logs/fold1_eval_loop.log
    break
  fi
  sleep 300
done
