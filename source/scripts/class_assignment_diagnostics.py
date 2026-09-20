"""Class-assignment accuracy diagnostic, with single/junction/position group
breakdown. 2026-08-17, peer-directed correction: `vessel_geometric_classifier.py`
turns a vessel-segment observation into a class GUESS (the forward direction
`phase2b_geometric_attachment.py` never built -- that script only checks
whether a guess overlaps the KNOWN true segment set). The two are NOT the
same measurement and must not be tabled side by side without this caveat.

This measures: for each GT lesion that a detected box any-overlaps (a TP by
the competition's own hit rule), does the DETECTED box's ellipsoid, classified
by `vessel_geometric_classifier.py` against a vessel mask (oracle GT or
predicted), name the correct one of 52 classes? Unlike phase2b's oracle
measurement (which classifies the GT lesion's own mask), this also carries
box-localization noise from the detector -- a strictly harder, more honest
number for what a deployed pipeline would actually produce.

Grouped by the TRUE class's `phase2b_location_class_to_vessel_map.py` group
(single/junction/position) -- peer-requested, since the classifier's tie-break
(prefer smaller segment-set on ambiguous segment) could systematically favor
single-segment classes over the 20 junction classes, and this is the check
for whether that's actually happening.
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

sys.path.insert(0, str(Path(__file__).parent))
from build_e2e_placeholder_pipeline import box_to_native_bounds  # noqa: E402
from build_vessel_baseline_masks import SCORE_THRESHOLD, MAX_BOXES_PER_CASE, ellipsoid_bool_mask  # noqa: E402
from vessel_geometric_classifier import VesselGeometricClassifier  # noqa: E402
from phase2b_location_class_to_vessel_map import LOCATION_TO_VESSEL  # noqa: E402


def box_overlaps_lesion(low: np.ndarray, high: np.ndarray, lesion_voxels: np.ndarray) -> bool:
    inside = np.all((lesion_voxels >= low) & (lesion_voxels <= high), axis=1)
    return bool(inside.any())


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--boxes-dir", type=Path, required=True)
    parser.add_argument("--vessel-mask-dir", type=Path, required=True)
    parser.add_argument("--case-ids-json", type=Path, required=True)
    parser.add_argument("--held-out-center", action="append", required=True)
    parser.add_argument("--label", required=True, help="e.g. predicted_ta36 or oracle_gt")
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

    n_tp = 0
    n_correct = 0
    group_tally = defaultdict(lambda: {"n": 0, "correct": 0})
    per_class_tally = defaultdict(lambda: {"n": 0, "correct": 0})
    n_no_vessel_mask = 0

    for i, case_id in enumerate(case_ids):
        if i % 20 == 0:
            print(f"progress: {i}/{len(case_ids)}", flush=True)
        case = cases_by_id.get(case_id)
        if case is None:
            continue
        boxes_pkl = args.boxes_dir / f"{case_id}_boxes.pkl"
        if not boxes_pkl.is_file():
            continue
        loc_arr = np.asarray(nib.load(case["location_mask"]).dataobj)
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
            if len(selected) >= MAX_BOXES_PER_CASE:
                break
            if scores[index] < SCORE_THRESHOLD:
                break
            selected.append(index)
        bounds = [(box_to_native_bounds(boxes[idx]), scores[idx]) for idx in selected]

        for cls in np.unique(loc_arr):
            if cls == 0:
                continue
            true_name = location_names.get(int(cls), f"class_{int(cls)}")
            labeled, n_components = ndimage.label(
                loc_arr == cls, structure=np.ones((3, 3, 3), dtype=np.uint8))
            for component_id in range(1, n_components + 1):
                lesion_voxels = np.argwhere(labeled == component_id)
                matches = [(low, high, sc) for (low, high), sc in bounds
                          if box_overlaps_lesion(low, high, lesion_voxels)]
                if not matches:
                    continue  # FN, not part of this diagnostic (see m1_lesion_diagnostics.py)
                low, high, _ = max(matches, key=lambda m: m[2])  # highest-scoring match
                n_tp += 1
                lesion_mask = ellipsoid_bool_mask(ref_shape, low, high)
                pred_class_id = classifier.classify_lesion_mask(lesion_mask, vessel_arr)
                pred_name = classifier.location_names.get(pred_class_id, None) if pred_class_id else None
                hit = pred_name == true_name
                n_correct += hit
                grp = group_by_class.get(true_name, "unmapped")
                group_tally[grp]["n"] += 1
                group_tally[grp]["correct"] += hit
                per_class_tally[true_name]["n"] += 1
                per_class_tally[true_name]["correct"] += hit

    report = {
        "label": args.label,
        "caveat": ("This classifies the DETECTED box's ellipsoid (via vessel_geometric_classifier.py), "
                  "NOT the GT lesion mask -- carries detector localization noise on top of classifier "
                  "noise. Do NOT compare directly against phase2b_geometric_attachment.py's "
                  "geometric_attachment_results.md numbers (those check segment-set overlap against a "
                  "KNOWN true class, a different and easier task; this produces a novel class GUESS)."),
        "n_cases": len(case_ids), "n_no_vessel_mask": n_no_vessel_mask,
        "n_tp_lesions_classified": n_tp,
        "overall_accuracy": round(n_correct / n_tp, 4) if n_tp else None,
        "by_group": {
            grp: {**v, "accuracy": round(v["correct"] / v["n"], 4) if v["n"] else None}
            for grp, v in group_tally.items()},
        "per_class": {
            cls: {**v, "accuracy": round(v["correct"] / v["n"], 4) if v["n"] else None}
            for cls, v in per_class_tally.items()},
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_class"}, indent=2))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
