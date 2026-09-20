"""PATCH P3+P4 (Experiment 1, 2026-08-20): per-lesion assignment confusion.

class_assignment_diagnostics.py only emits aggregate tallies (overall / by
group / per class), fixed at the cap-20 score>=0.3 operating point -- there
is no per-lesion output, so predicted-vs-oracle cannot be diffed and error
*patterns* cannot be seen, and accuracy cannot be recomputed at any k other
than 20 without rerunning the classifier.

This emits one row per GT lesion (case_id, true class, predicted class,
vessel source, group, territory, the matched box's rank/score/bounds) using
the SAME classifier call and SAME any-overlap hit rule as
class_assignment_diagnostics.py (imported, not re-derived), at
--score-threshold 0.3 (unchanged) up to --max-k candidates per case. Because
the rank of each lesion's best-matching box is recorded, accuracy and
detection at ANY k <= --max-k are exact post-hoc slices of one run (see
`accuracy_by_k` below) rather than needing separate runs per k.

Each non-correct row is classified, in this precedence order: wrong_side
(L<->R, same LOCATION_TO_VESSEL segment set), wrong_adjacent_branch (segment
sets intersect), wrong_territory (score_m1.py's own posterior{1,2}/
anterior{3,4,5} territory split disagrees), junction_single_confusion
(group differs and is exactly {single, junction}), else other. This
precedence is a judgement call, stated here rather than hidden, in the style
of this codebase's other documented judgement calls (see
phase2b_location_class_to_vessel_map.py's own docstring).

New file; scripts/class_assignment_diagnostics.py is not modified.
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent.parent))
from build_e2e_placeholder_pipeline import box_to_native_bounds  # noqa: E402
from build_vessel_baseline_masks import SCORE_THRESHOLD, ellipsoid_bool_mask  # noqa: E402
from vessel_geometric_classifier import VesselGeometricClassifier, nearest_label_lookup  # noqa: E402
from phase2b_location_class_to_vessel_map import LOCATION_TO_VESSEL  # noqa: E402
from score_m1 import territory, POSTERIOR, ANTERIOR  # noqa: E402

K_SUMMARY_VALUES = (1, 2, 3, 5, 10, 20)


def box_overlaps_lesion(low: np.ndarray, high: np.ndarray, lesion_voxels: np.ndarray) -> bool:
    inside = np.all((lesion_voxels >= low) & (lesion_voxels <= high), axis=1)
    return bool(inside.any())


def side_and_bare(name: str) -> tuple[str | None, str]:
    if name.startswith("R-"):
        return "R", name[2:]
    if name.startswith("L-"):
        return "L", name[2:]
    return None, name


def territory_bucket(name: str | None) -> str | None:
    if not name:
        return None
    terr = territory(name)
    return "posterior" if terr in POSTERIOR else "anterior" if terr in ANTERIOR else "unmapped"


def classify_error(true_name: str, pred_name: str | None) -> str:
    if pred_name is None:
        return "unclassified"
    if pred_name == true_name:
        return "correct"
    true_side, true_bare = side_and_bare(true_name)
    pred_side, pred_bare = side_and_bare(pred_name)
    if true_bare == pred_bare and true_side != pred_side:
        return "wrong_side"
    true_segs = LOCATION_TO_VESSEL.get(true_name, (None, frozenset()))[1]
    pred_segs = LOCATION_TO_VESSEL.get(pred_name, (None, frozenset()))[1]
    if true_segs & pred_segs:
        return "wrong_adjacent_branch"
    if territory_bucket(true_name) != territory_bucket(pred_name):
        return "wrong_territory"
    true_group = LOCATION_TO_VESSEL.get(true_name, (None, None))[0]
    pred_group = LOCATION_TO_VESSEL.get(pred_name, (None, None))[0]
    if true_group != pred_group and {true_group, pred_group} == {"single", "junction"}:
        return "junction_single_confusion"
    return "other"


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--boxes-dir", type=Path, required=True)
    parser.add_argument("--vessel-mask-dir", type=Path, required=True)
    parser.add_argument("--case-ids-json", type=Path, required=True)
    parser.add_argument("--held-out-center", action="append", required=True)
    parser.add_argument("--label", required=True, help="e.g. predicted_ta36 or oracle_gt")
    parser.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD)
    parser.add_argument("--max-k", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path,
                        default=Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task1/"
                                    "artifacts/phase0/case_manifest.json"))
    args = parser.parse_args()

    manifest = json.loads(args.case_manifest.read_text())
    cases_by_id = {c["case_id"]: c for c in manifest["cases"]}
    case_ids = json.loads(args.case_ids_json.read_text())
    location_names = {v: k for k, v in
                      json.loads(Path("/home/jovyan/rtx4claude-datavol-1/data_topaneu26/"
                                      "location_mapping.json").read_text())["labels"].items()}
    group_by_class = {name: grp for name, (grp, _) in LOCATION_TO_VESSEL.items()}

    print(f"building classifier, excluding {args.held_out_center}...", flush=True)
    classifier = VesselGeometricClassifier(exclude_centers=set(args.held_out_center))

    rows = []
    n_no_vessel_mask = 0
    n_skipped_missing_location = 0

    for i, case_id in enumerate(case_ids):
        if i % 20 == 0:
            print(f"progress: {i}/{len(case_ids)}", flush=True)
        case = cases_by_id.get(case_id)
        if case is None:
            n_skipped_missing_location += 1
            continue
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
        vessel_mask_path = args.vessel_mask_dir / f"{case_id}.nii.gz"
        if not vessel_mask_path.is_file():
            n_no_vessel_mask += 1
            continue
        vessel_arr = nib.load(str(vessel_mask_path)).get_fdata()
        ref_shape = nib.load(case["image"]).shape

        with boxes_pkl.open("rb") as handle:
            prediction = pickle.load(handle)
        boxes = np.asarray(prediction["pred_boxes"], dtype=float)
        scores = np.asarray(prediction["pred_scores"], dtype=float)
        order = np.argsort(-scores)
        selected = []
        for index in order:
            if len(selected) >= args.max_k:
                break
            if scores[index] < args.score_threshold:
                break
            selected.append(index)
        candidates = [(box_to_native_bounds(boxes[idx]), float(scores[idx])) for idx in selected]

        nearest_label_volume = nearest_label_lookup(vessel_arr) if vessel_arr.shape == ref_shape else None

        for cls in np.unique(loc_arr):
            if cls == 0:
                continue
            true_name = location_names.get(int(cls), f"class_{int(cls)}")
            labeled, n_components = ndimage.label(
                loc_arr == cls, structure=np.ones((3, 3, 3), dtype=np.uint8))
            for component_id in range(1, n_components + 1):
                lesion_voxels = np.argwhere(labeled == component_id)
                centroid = lesion_voxels.mean(axis=0).tolist()

                best_rank, best_low, best_high, best_score = None, None, None, None
                for rank, ((low, high), sc) in enumerate(candidates):
                    if box_overlaps_lesion(low, high, lesion_voxels):
                        best_rank, best_low, best_high, best_score = rank, low, high, sc
                        break  # candidates is score-descending: first hit = best

                row = {
                    "case_id": case_id,
                    "true_class_id": int(cls), "true_class_name": true_name,
                    "true_group": group_by_class.get(true_name),
                    "true_territory": territory_bucket(true_name),
                    "lesion_centroid_vox": centroid, "lesion_n_voxels": int(len(lesion_voxels)),
                    "vessel_source": args.label,
                    "detected": best_rank is not None,
                    "rank": best_rank, "score": best_score,
                    "box_low": best_low.tolist() if best_low is not None else None,
                    "box_high": best_high.tolist() if best_high is not None else None,
                }
                if best_rank is None:
                    row.update(predicted_class_id=None, predicted_class_name=None,
                              predicted_group=None, predicted_territory=None,
                              correct=False, error_type="not_detected")
                else:
                    lesion_mask = ellipsoid_bool_mask(ref_shape, best_low, best_high)
                    if nearest_label_volume is not None:
                        pred_class_id = classifier.classify_from_nearest_label_volume(
                            lesion_mask, nearest_label_volume)
                    else:
                        pred_class_id = classifier.class_id_by_name.get(classifier.most_common_class)
                    pred_name = classifier.location_names.get(pred_class_id) if pred_class_id else None
                    row.update(
                        predicted_class_id=pred_class_id, predicted_class_name=pred_name,
                        predicted_group=group_by_class.get(pred_name) if pred_name else None,
                        predicted_territory=territory_bucket(pred_name),
                        correct=(pred_name == true_name),
                        error_type=classify_error(true_name, pred_name),
                    )
                rows.append(row)

    total_lesions = len(rows)
    accuracy_by_k = {}
    for k in K_SUMMARY_VALUES:
        if k > args.max_k:
            continue
        detected = [r for r in rows if r["rank"] is not None and r["rank"] < k]
        correct = [r for r in detected if r["correct"]]
        accuracy_by_k[k] = {
            "n_detected": len(detected),
            "detection_rate": len(detected) / total_lesions if total_lesions else None,
            "n_correct": len(correct),
            "assignment_accuracy_given_detected": len(correct) / len(detected) if detected else None,
        }

    per_class_tally = defaultdict(lambda: {"n": 0, "correct": 0, "predicted_instead": defaultdict(int)})
    for r in rows:
        if not r["detected"]:
            continue
        t = per_class_tally[r["true_class_name"]]
        t["n"] += 1
        t["correct"] += int(r["correct"])
        if not r["correct"]:
            t["predicted_instead"][r["predicted_class_name"] or "UNCLASSIFIED"] += 1

    error_type_tally: dict[str, int] = defaultdict(int)
    for r in rows:
        error_type_tally[r["error_type"]] += 1

    zero_accuracy_classes = {
        name: {"n": v["n"], "predicted_instead": dict(v["predicted_instead"])}
        for name, v in per_class_tally.items() if v["n"] >= 1 and v["correct"] == 0
    }

    report = {
        "label": args.label, "boxes_dir": str(args.boxes_dir),
        "vessel_mask_dir": str(args.vessel_mask_dir),
        "score_threshold": args.score_threshold, "max_k": args.max_k,
        "n_cases": len(case_ids), "n_no_vessel_mask": n_no_vessel_mask,
        "n_skipped_missing_location_mask": n_skipped_missing_location,
        "total_lesions": total_lesions,
        "accuracy_by_k": accuracy_by_k,
        "error_type_tally": dict(error_type_tally),
        "zero_accuracy_classes_at_max_k": zero_accuracy_classes,
        "per_class_at_max_k": {
            name: {"n": v["n"], "correct": v["correct"],
                  "accuracy": round(v["correct"] / v["n"], 4) if v["n"] else None,
                  "predicted_instead": dict(v["predicted_instead"])}
            for name, v in per_class_tally.items()},
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "per_class_at_max_k")}, indent=2))
    print(f"\nwrote {args.output} ({len(rows)} lesion rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
