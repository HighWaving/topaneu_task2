"""PATCH P2 (Experiment 1, 2026-08-20): pure geometric box-vs-GT recall by
top-K, generalized from probe_pure_box_recall.py to any modality (that
script hardcodes `c["modality"] == "mr"`) and with K_VALUES extended to 200.
CT's candidate-recall curve does not exist anywhere on disk; this produces
it from the cached artifacts/ct_fold2_all109_boxes, no re-inference.

Guards the known stale-manifest crash (case_manifest.json lists
topaneu_center1_mr_150, which was Discarded and has no files on disk) by
skipping and COUNTING any case whose location_mask is missing, instead of
crashing -- this is why the center1 epoch-60 probe died at 30/200 in the
prior run of this script.

New file; scripts/probe_pure_box_recall.py is not modified.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent.parent))
from build_e2e_placeholder_pipeline import box_to_native_bounds  # noqa: E402

MANIFEST = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task1/artifacts/phase0/case_manifest.json")
K_VALUES = (1, 2, 3, 5, 10, 20, 50, 200)


def box_overlaps_lesion(low: np.ndarray, high: np.ndarray, lesion_voxels: np.ndarray) -> bool:
    inside = np.all((lesion_voxels >= low) & (lesion_voxels <= high), axis=1)
    return bool(inside.any())


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", required=True, choices=["mr", "ct"])
    parser.add_argument("--center", default=None,
                        help="optional: restrict to one center. Default: every center of "
                             "--modality that has a boxes.pkl in --boxes-dir (e.g. CT's "
                             "109-case set spans center2+center4).")
    parser.add_argument("--boxes-dir", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.case_manifest.read_text())
    cases_by_id = {c["case_id"]: c for c in manifest["cases"]}
    available_boxes = {p.stem.removesuffix("_boxes") for p in args.boxes_dir.glob("*_boxes.pkl")}
    val_ids = [c["case_id"] for c in manifest["cases"]
              if c["modality"] == args.modality
              and (args.center is None or c["center"] == args.center)
              and c["case_id"] in available_boxes]

    hits_by_k = {k: 0 for k in K_VALUES}
    total_lesions = 0
    max_k = max(K_VALUES)
    n_skipped_missing_location = 0

    for i, case_id in enumerate(val_ids):
        if i % 20 == 0:
            print(f"progress: {i}/{len(val_ids)} ({case_id})", flush=True)
        case = cases_by_id[case_id]
        boxes_pkl = args.boxes_dir / f"{case_id}_boxes.pkl"
        if not boxes_pkl.is_file():
            continue
        loc_path = Path(case["location_mask"])
        if not loc_path.is_file():
            n_skipped_missing_location += 1
            continue
        loc_arr = np.asarray(nib.load(str(loc_path)).dataobj)
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

    print(f"total lesions, modality={args.modality} center={args.center}: {total_lesions}", flush=True)
    print(f"skipped (missing location_mask): {n_skipped_missing_location}", flush=True)
    print(f"{'K':>5} {'hits':>6} {'recall':>8}")
    for k in K_VALUES:
        recall = hits_by_k[k] / total_lesions if total_lesions else 0.0
        print(f"{k:>5} {hits_by_k[k]:>6} {recall:>8.1%}")

    out = {"modality": args.modality, "center": args.center, "boxes_dir": str(args.boxes_dir),
          "n_cases_considered": len(val_ids),
          "n_skipped_missing_location_mask": n_skipped_missing_location,
          "total_lesions": total_lesions,
          "by_k": {k: {"hits": hits_by_k[k], "recall": hits_by_k[k] / total_lesions if total_lesions else None}
                   for k in K_VALUES}}
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
