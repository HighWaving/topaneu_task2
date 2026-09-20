"""PATCH P5 (Experiment 1, 2026-08-20): Dice | (detected AND correctly
classified), by k. Nothing on disk measures this -- the official DICE_i is
summed over TPs and divided by TP+FN+FP per class, so a low official Dice is
produced by missed lesions and wrong classes long before shape matters.
This isolates the shape term: for lesions where
scripts/analysis/assignment_confusion.py (P3) recorded rank < k AND
correct==True, computes Dice between that lesion's predicted ellipsoid
(ellipsoid_bool_mask over the SAME box bounds P3 matched, imported from
build_vessel_baseline_masks.py, not re-derived) and the GT lesion's own
voxel mask. Reports the distribution (median, IQR), not just the mean,
since lesion Dice is heavily size-skewed.

Input: an assignment_confusion.py output file (one row per lesion, with
box_low/box_high/rank/correct/lesion_centroid_vox already computed). New
file; no existing script is modified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent.parent))
from build_vessel_baseline_masks import ellipsoid_bool_mask  # noqa: E402

K_VALUES = (1, 2, 3, 5, 10)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return float(2 * inter / denom) if denom else 1.0


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignment-confusion-json", type=Path, required=True,
                        help="output of scripts/analysis/assignment_confusion.py")
    parser.add_argument("--case-manifest", type=Path,
                        default=Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task1/"
                                    "artifacts/phase0/case_manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.assignment_confusion_json.read_text())
    rows = data["rows"]
    manifest = json.loads(args.case_manifest.read_text())
    cases_by_id = {c["case_id"]: c for c in manifest["cases"]}

    by_case: dict[str, list[dict]] = {}
    for r in rows:
        by_case.setdefault(r["case_id"], []).append(r)

    dice_by_k: dict[int, list[float]] = {k: [] for k in K_VALUES}
    n_cases_missing = 0

    for case_id, case_rows in by_case.items():
        case = cases_by_id.get(case_id)
        if case is None:
            n_cases_missing += 1
            continue
        loc_path = Path(case["location_mask"])
        if not loc_path.is_file():
            n_cases_missing += 1
            continue
        shape = nib.load(case["image"]).shape
        loc_arr = np.asarray(nib.load(str(loc_path)).dataobj)

        # GT lesion voxel masks are expensive (ndimage.label per class per
        # case) -- cache the labeled volume per true_class_id within this
        # case, since assignment_confusion.py's rows are already grouped by
        # (case, true_class) and re-labeling per row would be redundant.
        components_by_class: dict[int, tuple[np.ndarray, int]] = {}

        for r in case_rows:
            if r["rank"] is None or not r["correct"]:
                continue
            cls = r["true_class_id"]
            if cls not in components_by_class:
                labeled, n = ndimage.label(
                    loc_arr == cls, structure=np.ones((3, 3, 3), dtype=np.uint8))
                components_by_class[cls] = (labeled, n)
            labeled, n = components_by_class[cls]
            centroid = np.asarray(r["lesion_centroid_vox"])
            best_cid, best_dist = None, None
            for cid in range(1, n + 1):
                voxels = np.argwhere(labeled == cid)
                d = float(np.linalg.norm(voxels.mean(axis=0) - centroid))
                if best_dist is None or d < best_dist:
                    best_cid, best_dist = cid, d
            if best_cid is None:
                continue
            gt_mask = labeled == best_cid

            low = np.asarray(r["box_low"])
            high = np.asarray(r["box_high"])
            pred_mask = ellipsoid_bool_mask(shape, low, high)
            d = dice(pred_mask, gt_mask)

            for k in K_VALUES:
                if r["rank"] < k:
                    dice_by_k[k].append(d)

    def stats(values: list[float]) -> dict:
        if not values:
            return {"n": 0, "median": None, "q1": None, "q3": None, "mean": None, "min": None, "max": None}
        arr = np.asarray(values)
        return {"n": len(arr), "median": float(np.median(arr)),
                "q1": float(np.percentile(arr, 25)), "q3": float(np.percentile(arr, 75)),
                "mean": float(np.mean(arr)), "min": float(arr.min()), "max": float(arr.max())}

    report = {
        "source": str(args.assignment_confusion_json),
        "label": data.get("label"),
        "n_cases_missing_location_mask": n_cases_missing,
        "by_k": {k: stats(v) for k, v in dice_by_k.items()},
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
