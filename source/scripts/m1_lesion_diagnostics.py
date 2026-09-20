"""M1 lesion-level diagnostics (2026-08-15): sensitivity, FN, FP/case, candidates/case.

At the SAME operating point build_m1_masks.py actually used (score>=0.3, max
20 boxes/case) -- not a top-K sweep, the real selected-box set. any-overlap
hit rule, matching the competition. Diagnostic only, not one of the official
6 metrics (peer's own instruction: FP/case is informative, never an
optimization target).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from build_e2e_placeholder_pipeline import box_to_native_bounds  # noqa: E402
from build_m1_masks import SCORE_THRESHOLD, MAX_BOXES_PER_CASE  # noqa: E402


def box_overlaps_lesion(low: np.ndarray, high: np.ndarray, lesion_voxels: np.ndarray) -> bool:
    inside = np.all((lesion_voxels >= low) & (lesion_voxels <= high), axis=1)
    return bool(inside.any())


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--boxes-dir", type=Path, required=True)
    parser.add_argument("--case-ids-json", type=Path, required=True,
                        help="JSON list of case ids to evaluate")
    parser.add_argument("--case-manifest", type=Path,
                        default=Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task1/"
                                    "artifacts/phase0/case_manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.case_manifest.read_text())
    cases_by_id = {c["case_id"]: c for c in manifest["cases"]}
    case_ids = json.loads(args.case_ids_json.read_text())

    total_lesions = 0
    total_hit_lesions = 0
    total_candidates = 0
    total_fp_candidates = 0
    candidates_per_case = {}
    per_case_detail = {}

    for case_id in case_ids:
        case = cases_by_id[case_id]
        loc_arr = np.asarray(nib.load(case["location_mask"]).dataobj)
        labeled, n = ndimage.label(loc_arr > 0, structure=np.ones((3, 3, 3), dtype=np.uint8))
        lesion_voxel_sets = [np.argwhere(labeled == cid) for cid in range(1, n + 1)]
        n_lesions = len(lesion_voxel_sets)
        total_lesions += n_lesions

        boxes_pkl = args.boxes_dir / f"{case_id}_boxes.pkl"
        if not boxes_pkl.is_file():
            candidates_per_case[case_id] = 0
            per_case_detail[case_id] = {"n_lesions": n_lesions, "n_hit": 0, "n_candidates": 0, "n_fp": 0}
            continue
        with boxes_pkl.open("rb") as handle:
            prediction = pickle.load(handle)
        boxes = np.asarray(prediction["pred_boxes"], dtype=float)
        scores = np.asarray(prediction["pred_scores"], dtype=float)
        order = np.argsort(-scores)
        selected = []
        for index in order:
            if len(selected) >= MAX_BOXES_PER_CASE:
                break
            if scores[index] < SCORE_THRESHOLD:
                break
            selected.append(index)

        bounds = [box_to_native_bounds(boxes[i]) for i in selected]
        candidates_per_case[case_id] = len(bounds)
        total_candidates += len(bounds)

        n_hit_lesions = 0
        for lesion_voxels in lesion_voxel_sets:
            if any(box_overlaps_lesion(low, high, lesion_voxels) for low, high in bounds):
                n_hit_lesions += 1
        total_hit_lesions += n_hit_lesions

        n_fp = 0
        for low, high in bounds:
            hit_any = any(box_overlaps_lesion(low, high, lv) for lv in lesion_voxel_sets)
            if not hit_any:
                n_fp += 1
        total_fp_candidates += n_fp

        per_case_detail[case_id] = {
            "n_lesions": n_lesions, "n_hit": n_hit_lesions,
            "n_candidates": len(bounds), "n_fp": n_fp,
        }

    n_cases = len(case_ids)
    counts = list(candidates_per_case.values())
    report = {
        "n_cases": n_cases,
        "total_lesions": total_lesions,
        "total_hit_lesions": total_hit_lesions,
        "lesion_sensitivity": total_hit_lesions / total_lesions if total_lesions else None,
        "total_fn": total_lesions - total_hit_lesions,
        "total_candidates": total_candidates,
        "total_fp_candidates": total_fp_candidates,
        "fp_per_case_mean": total_fp_candidates / n_cases if n_cases else None,
        "candidates_per_case": {
            "mean": float(np.mean(counts)), "median": float(np.median(counts)),
            "min": int(np.min(counts)), "max": int(np.max(counts)),
            "zero_candidate_cases": sum(1 for c in counts if c == 0),
        },
        "score_threshold": SCORE_THRESHOLD, "max_boxes_per_case": MAX_BOXES_PER_CASE,
        "per_case": per_case_detail,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_case"}, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
