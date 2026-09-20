"""PATCH P1 (Experiment 1, 2026-08-20): lesion sensitivity, candidates/case and
FP/case swept over k in {1,2,3,5,10} and score-threshold in {0.1,0.2,0.3}.

m1_lesion_diagnostics.py only measures the single realised operating point
(score>=0.3, cap 20, imported from build_m1_masks.py) -- it has no k or
threshold flag. This quantifies how many real lesions the 0.3 threshold
discards relative to the top-k candidate ceiling (epoch-49 prior: ceiling
57/58 = 98.3% vs realised 51/58 = 87.9%, ~6 lesions lost to the threshold).
Same any-overlap hit rule as m1_lesion_diagnostics.py and the official
scorer. New file; scripts/m1_lesion_diagnostics.py is not modified.
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

K_VALUES = (1, 2, 3, 5, 10)
SCORE_THRESHOLDS = (0.1, 0.2, 0.3)


def box_overlaps_lesion(low: np.ndarray, high: np.ndarray, lesion_voxels: np.ndarray) -> bool:
    inside = np.all((lesion_voxels >= low) & (lesion_voxels <= high), axis=1)
    return bool(inside.any())


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--boxes-dir", type=Path, required=True)
    parser.add_argument("--case-ids-json", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path,
                        default=Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task1/"
                                    "artifacts/phase0/case_manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.case_manifest.read_text())
    cases_by_id = {c["case_id"]: c for c in manifest["cases"]}
    case_ids = json.loads(args.case_ids_json.read_text())

    # Load every case's lesion voxel sets and every box's (bounds, score) ONCE;
    # the (threshold, k) grid below is then pure re-selection over cached
    # arrays, not repeated file I/O per grid point.
    per_case_lesions: dict[str, list[np.ndarray]] = {}
    per_case_boxes: dict[str, tuple[list, np.ndarray]] = {}
    n_skipped_missing_location = 0
    total_lesions = 0

    for case_id in case_ids:
        case = cases_by_id.get(case_id)
        if case is None:
            n_skipped_missing_location += 1
            continue
        loc_path = Path(case["location_mask"])
        if not loc_path.is_file():
            n_skipped_missing_location += 1
            continue
        loc_arr = np.asarray(nib.load(str(loc_path)).dataobj)
        labeled, n = ndimage.label(loc_arr > 0, structure=np.ones((3, 3, 3), dtype=np.uint8))
        lesion_voxel_sets = [np.argwhere(labeled == cid) for cid in range(1, n + 1)]
        per_case_lesions[case_id] = lesion_voxel_sets
        total_lesions += len(lesion_voxel_sets)

        boxes_pkl = args.boxes_dir / f"{case_id}_boxes.pkl"
        if not boxes_pkl.is_file():
            per_case_boxes[case_id] = ([], np.zeros(0))
            continue
        with boxes_pkl.open("rb") as handle:
            prediction = pickle.load(handle)
        boxes = np.asarray(prediction["pred_boxes"], dtype=float)
        scores = np.asarray(prediction["pred_scores"], dtype=float)
        order = np.argsort(-scores)
        bounds_sorted = [box_to_native_bounds(boxes[i]) for i in order]
        scores_sorted = scores[order]
        per_case_boxes[case_id] = (bounds_sorted, scores_sorted)

    print(f"loaded {len(per_case_lesions)} cases, {total_lesions} lesions, "
         f"{n_skipped_missing_location} skipped (missing location_mask)", flush=True)

    grid: dict[float, dict[int, dict]] = {}
    for threshold in SCORE_THRESHOLDS:
        grid[threshold] = {}
        for k in K_VALUES:
            hit_lesions = 0
            total_candidates = 0
            total_fp = 0
            candidates_per_case = {}
            for case_id, lesion_voxel_sets in per_case_lesions.items():
                bounds_sorted, scores_sorted = per_case_boxes[case_id]
                selected = []
                for (low, high), sc in zip(bounds_sorted, scores_sorted):
                    if len(selected) >= k:
                        break
                    if sc < threshold:
                        break
                    selected.append((low, high))
                candidates_per_case[case_id] = len(selected)
                total_candidates += len(selected)

                for lesion_voxels in lesion_voxel_sets:
                    if any(box_overlaps_lesion(low, high, lesion_voxels) for low, high in selected):
                        hit_lesions += 1
                for low, high in selected:
                    if not any(box_overlaps_lesion(low, high, lv) for lv in lesion_voxel_sets):
                        total_fp += 1

            n_cases = len(per_case_lesions)
            counts = list(candidates_per_case.values())
            grid[threshold][k] = {
                "lesion_sensitivity": hit_lesions / total_lesions if total_lesions else None,
                "hit_lesions": hit_lesions, "total_lesions": total_lesions,
                "fn": total_lesions - hit_lesions,
                "total_candidates": total_candidates,
                "total_fp": total_fp,
                "fp_per_case_mean": total_fp / n_cases if n_cases else None,
                "candidates_per_case_mean": float(np.mean(counts)) if counts else None,
            }
        line = ", ".join(f"k={k} sens={grid[threshold][k]['lesion_sensitivity']:.3f}" for k in K_VALUES)
        print(f"threshold={threshold}: {line}", flush=True)

    report = {
        "boxes_dir": str(args.boxes_dir),
        "n_cases": len(per_case_lesions),
        "n_skipped_missing_location_mask": n_skipped_missing_location,
        "total_lesions": total_lesions,
        "k_values": list(K_VALUES), "score_thresholds": list(SCORE_THRESHOLDS),
        "by_threshold": {str(t): {str(k): v for k, v in by_k.items()} for t, by_k in grid.items()},
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
