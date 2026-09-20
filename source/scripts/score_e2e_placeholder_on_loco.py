"""Scores the placeholder end-to-end pipeline's output on a LOCO held-out center.

2026-08-14. First real 6-metric TopAneu Task 2 score, using the crude
detect-only + ellipsoid + placeholder-class pipeline
(build_e2e_placeholder_pipeline.py). Wraps local_scoring_arena.py (the
validated wrapper around the OFFICIAL evaluate.py) directly -- no metric
reimplementation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, "/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/scripts")
from local_scoring_arena import aggregate, score_case  # noqa: E402

LOCO_SPLITS = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/configs/loco_splits.json")
OUT_DIR = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/reports")


def load_prediction(path: Path) -> np.ndarray:
    """local_scoring_arena expects the SAME (z,y,x) SimpleITK array order load_gt() uses."""
    import SimpleITK as sitk
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--held-out-center", required=True, choices=("center1", "center2", "center4", "center5"))
    parser.add_argument("--label", default="e2e_placeholder")
    args = parser.parse_args()

    loco = json.loads(LOCO_SPLITS.read_text())
    val_ids = loco["folds"][args.held_out_center]["val"]

    results = []
    missing = []
    for case_id in val_ids:
        pred_path = args.predictions_dir / f"{case_id}.nii.gz"
        if not pred_path.is_file():
            missing.append(case_id)
            continue
        pred = load_prediction(pred_path)
        results.append(score_case(pred, case_id))

    print(f"scored {len(results)}/{len(val_ids)} held-out cases ({len(missing)} missing predictions)",
         flush=True)
    if missing:
        print(f"missing (first 10): {missing[:10]}", flush=True)

    if not results:
        print("no predictions scored, aborting", flush=True)
        return 1

    agg = aggregate(results)
    print("overall:", json.dumps(agg["overall"], indent=2))

    out_path = OUT_DIR / f"{args.label}_{args.held_out_center}_score.json"
    out_path.write_text(json.dumps({
        "held_out_center": args.held_out_center,
        "n_scored": len(results), "n_missing": len(missing),
        "overall": agg["overall"], "per_class": agg["per_class"],
    }, indent=2, default=float) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
