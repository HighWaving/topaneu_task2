"""Pure geometric box-vs-GT recall by top-K, no pipeline/mask/scoring involved.

2026-08-14, peer-requested fast check to answer "threshold problem (a) vs
detector-genuinely-misses (b)" in seconds rather than waiting on the full
pipeline sweep. any-overlap hit rule, matching the competition's own rule --
for each GT lesion, is there any top-K box whose voxel footprint overlaps it
at all.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, "/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/scripts")
from build_e2e_placeholder_pipeline import box_to_native_bounds  # noqa: E402

BOXES_DIR = Path("/home/jovyan/rtx4claude-datavol-1/new_aneurysms/artifacts/nndet_task020_zeroshot_topaneu_mr")
MANIFEST = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task1/artifacts/phase0/case_manifest.json")
LOCO_SPLITS = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/configs/loco_splits.json")
K_VALUES = (1, 3, 5, 10, 20, 50, 200)


def box_overlaps_lesion(low: np.ndarray, high: np.ndarray, lesion_voxels: np.ndarray) -> bool:
    """any-overlap: does any lesion voxel fall inside [low, high]?"""
    inside = np.all((lesion_voxels >= low) & (lesion_voxels <= high), axis=1)
    return bool(inside.any())


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--center", default="center5",
                        help="filter directly by case center -- the zero-shot detector never "
                             "trained on ANY TopAneu data, so every center is equally held-out "
                             "from its perspective (2026-08-14, peer-requested multi-center check, "
                             "not going through the LOCO val split since that's a different question)")
    parser.add_argument("--boxes-dir", type=Path, default=BOXES_DIR,
                        help="override to point at a different model's box predictions, e.g. "
                             "a real-model checkpoint's eval output instead of the zero-shot "
                             "Task020FG default")
    parser.add_argument("--label", default=None,
                        help="output filename tag; defaults to --center")
    args = parser.parse_args()
    boxes_dir = args.boxes_dir
    label = args.label or args.center

    manifest = json.loads(MANIFEST.read_text())
    cases_by_id = {c["case_id"]: c for c in manifest["cases"]}
    val_ids = [c["case_id"] for c in manifest["cases"]
              if c["center"] == args.center and c["modality"] == "mr"]

    hits_by_k = {k: 0 for k in K_VALUES}
    total_lesions = 0
    max_k = max(K_VALUES)

    for i, case_id in enumerate(val_ids):
        print(f"progress: {i}/{len(val_ids)} ({case_id})", flush=True)
        case = cases_by_id[case_id]
        boxes_pkl = boxes_dir / f"{case_id}_boxes.pkl"
        if not boxes_pkl.is_file():
            continue
        loc_img = nib.load(case["location_mask"])
        loc_arr = np.asarray(loc_img.dataobj)
        if not loc_arr.any():
            continue
        labeled, n = ndimage.label(loc_arr > 0, structure=np.ones((3, 3, 3), dtype=np.uint8))
        lesion_voxel_sets = [np.argwhere(labeled == cid) for cid in range(1, n + 1)]
        total_lesions += len(lesion_voxel_sets)

        with boxes_pkl.open("rb") as handle:
            prediction = pickle.load(handle)
        boxes = np.asarray(prediction["pred_boxes"], dtype=float)
        scores = np.asarray(prediction["pred_scores"], dtype=float)
        order = np.argsort(-scores)[:max_k]

        # Precompute each top-max_k box's low/high once, not per (lesion, K).
        box_bounds = [box_to_native_bounds(boxes[idx]) for idx in order]

        for lesion_voxels in lesion_voxel_sets:
            first_hit_rank = None
            for rank, (low, high) in enumerate(box_bounds):
                if box_overlaps_lesion(low, high, lesion_voxels):
                    first_hit_rank = rank
                    break
            if first_hit_rank is None:
                continue
            for k in K_VALUES:
                if first_hit_rank < k:
                    hits_by_k[k] += 1

    print(f"total lesions in {args.center} (MR): {total_lesions}", flush=True)
    print(f"{'K':>5} {'hits':>6} {'recall':>8}")
    for k in K_VALUES:
        recall = hits_by_k[k] / total_lesions if total_lesions else 0.0
        print(f"{k:>5} {hits_by_k[k]:>6} {recall:>8.1%}")

    out = {"center": args.center, "boxes_dir": str(boxes_dir), "total_lesions": total_lesions,
          "by_k": {k: {"hits": hits_by_k[k], "recall": hits_by_k[k] / total_lesions}
                   for k in K_VALUES}}
    out_path = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/reports/"
                    f"pure_box_recall_{label}.json")
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
