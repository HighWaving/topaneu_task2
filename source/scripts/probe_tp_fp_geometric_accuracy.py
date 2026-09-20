"""TP/FP-split geometric class-assignment accuracy on DETECTED boxes (not GT locations).

2026-08-14, peer's decisive branch point. Earlier oracle geometric-attachment
accuracy (76.3% pooled, phase2b_geometric_attachment.py) was measured by
querying geometry at the TRUE lesion's own location -- that number cannot
tell us whether the pipeline's much lower observed class accuracy (11.3%
official RECALL at N=1, vs a 45.2% pure box-vs-GT recall at the same K=1)
comes from (a) FP candidates polluting the output, or (b) the geometric
method itself being sensitive to the offset between a detected box's center
and the true lesion's centroid.

This splits center5's zero-shot candidates into:
  TP: any-overlap with a real GT lesion -- for these, geometric accuracy is
      computed against the TRUE lesion's class, querying geometry at the
      CANDIDATE BOX's center/region (not the GT location) -- the first real
      test of whether method C (nearest-centroid) or method B (shell
      histogram, aggregates a region rather than one point) is more robust
      to localization error.
  FP: no overlap with any GT lesion -- no true class to score against,
      reported as which classes they get assigned to (this is where the
      "attractor class" pollution actually enters the pipeline's output).

Reuses phase2b_geometric_attachment.py's method_b_shell_histogram/
method_c_nearest_centroid/correct() and phase2b_location_class_to_vessel_map's
LOCATION_TO_VESSEL directly -- not reimplemented.
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

sys.path.insert(0, "/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/scripts")
from phase2b_geometric_attachment import (  # noqa: E402
    load_labels, method_b_shell_histogram, method_c_nearest_centroid,
    nearest_label_lookup, correct, SHELL_DILATION_VOXELS,
)
from phase2b_location_class_to_vessel_map import LOCATION_TO_VESSEL  # noqa: E402
from build_e2e_placeholder_pipeline import box_to_native_bounds  # noqa: E402

BOXES_DIR = Path("/home/jovyan/rtx4claude-datavol-1/new_aneurysms/artifacts/nndet_task020_zeroshot_topaneu_mr")
MANIFEST = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task1/artifacts/phase0/case_manifest.json")
LOCO_SPLITS = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/configs/loco_splits.json")
DATA_ROOT = Path("/home/jovyan/rtx4claude-datavol-1/data_topaneu26")
GLOBAL_ZERO_CLASSES = {3, 9, 10, 12, 13, 14, 18, 20, 52}
TOP_K_CANDIDATES = 20  # per case, enough to get a real TP/FP mixture without full-200 cost


def box_ellipsoid_mask(shape, low, high) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    centre = (low + high) / 2.0
    radii = np.maximum((high - low) / 2.0, 0.5)
    lo_clip = np.maximum(np.floor(low).astype(int), 0)
    hi_clip = np.minimum(np.ceil(high).astype(int), np.asarray(shape))
    if np.any(lo_clip >= hi_clip):
        return mask
    xs = np.arange(lo_clip[0], hi_clip[0])
    ys = np.arange(lo_clip[1], hi_clip[1])
    zs = np.arange(lo_clip[2], hi_clip[2])
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    r2 = (((xx - centre[0]) / radii[0]) ** 2 + ((yy - centre[1]) / radii[1]) ** 2 +
         ((zz - centre[2]) / radii[2]) ** 2)
    sub = mask[lo_clip[0]:hi_clip[0], lo_clip[1]:hi_clip[1], lo_clip[2]:hi_clip[2]]
    sub[r2 <= 1.0] = True
    mask[lo_clip[0]:hi_clip[0], lo_clip[1]:hi_clip[1], lo_clip[2]:hi_clip[2]] = sub
    return mask


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    cases_by_id = {c["case_id"]: c for c in manifest["cases"]}
    loco = json.loads(LOCO_SPLITS.read_text())
    val_ids = [c for c in loco["folds"]["center5"]["val"]
              if cases_by_id.get(c, {}).get("modality") == "mr"]

    location_names = load_labels(DATA_ROOT / "location_mapping.json")
    vessel_names = load_labels(DATA_ROOT / "vessel_mapping.json")

    tp_results = {"B": {"correct": 0, "total": 0}, "C": {"correct": 0, "total": 0}}
    fp_assigned_classes = Counter()
    fp_total = 0
    tp_total_candidates = 0

    for i, case_id in enumerate(val_ids):
        if i % 10 == 0:
            print(f"progress: {i}/{len(val_ids)}", flush=True)
        case = cases_by_id[case_id]
        boxes_pkl = BOXES_DIR / f"{case_id}_boxes.pkl"
        if not boxes_pkl.is_file():
            continue

        loc_img = nib.load(case["location_mask"])
        loc_arr = np.asarray(loc_img.dataobj)
        shape = loc_arr.shape
        if not loc_arr.any():
            continue
        labeled, n = ndimage.label(loc_arr > 0, structure=np.ones((3, 3, 3), dtype=np.uint8))
        gt_lesions = []
        for cid in range(1, n + 1):
            voxels = np.argwhere(labeled == cid)
            true_class_id = int(loc_arr[tuple(voxels[0])])
            true_class_name = location_names.get(true_class_id, None)
            gt_lesions.append((voxels, true_class_name))

        vessel_arr_zyx = sitk.GetArrayFromImage(sitk.ReadImage(str(case["vessel_mask"])))
        vessel_arr = np.transpose(vessel_arr_zyx, (2, 1, 0))
        nearest_label_volume = nearest_label_lookup(vessel_arr)

        with boxes_pkl.open("rb") as handle:
            prediction = pickle.load(handle)
        boxes = np.asarray(prediction["pred_boxes"], dtype=float)
        scores = np.asarray(prediction["pred_scores"], dtype=float)
        order = np.argsort(-scores)[:TOP_K_CANDIDATES]

        for idx in order:
            low, high = box_to_native_bounds(boxes[idx])
            # any-overlap: does any GT lesion voxel fall inside [low, high]?
            # O(n_lesion_voxels), not O(n_box_voxels) -- same approach as
            # probe_pure_box_recall.py, much cheaper than enumerating the box.
            true_class_name = None
            for gt_voxels, cls_name in gt_lesions:
                inside = np.all((gt_voxels >= low) & (gt_voxels <= high), axis=1)
                if inside.any():
                    true_class_name = cls_name
                    break

            if true_class_name is None:
                # FP candidate: tabulate what it gets assigned via method C only (cheap).
                fp_total += 1
                centre = np.rint((low + high) / 2.0).astype(int)
                centre = np.clip(centre, 0, np.asarray(shape) - 1)
                vessel_label = int(nearest_label_volume[tuple(centre)])
                if vessel_label != 0:
                    seg_name = vessel_names.get(vessel_label, f"vessel_{vessel_label}")
                    fp_assigned_classes[seg_name] += 1
                continue

            if true_class_name not in LOCATION_TO_VESSEL:
                continue
            group, truth_segments = LOCATION_TO_VESSEL[true_class_name]
            tp_total_candidates += 1
            lesion_mask = box_ellipsoid_mask(shape, low, high)

            pred_c = method_c_nearest_centroid(lesion_mask, vessel_names, nearest_label_volume)
            tp_results["C"]["total"] += 1
            if correct(pred_c, truth_segments):
                tp_results["C"]["correct"] += 1

            pred_b = method_b_shell_histogram(lesion_mask, vessel_arr, vessel_names,
                                              SHELL_DILATION_VOXELS[1])  # k=2, mid-range
            tp_results["B"]["total"] += 1
            if correct(pred_b, truth_segments):
                tp_results["B"]["correct"] += 1

    print(f"\nTP candidates scored: {tp_total_candidates}, FP candidates: {fp_total}", flush=True)
    for method in ("B", "C"):
        r = tp_results[method]
        acc = r["correct"] / r["total"] if r["total"] else None
        print(f"method {method}: {r['correct']}/{r['total']} = {acc}", flush=True)

    print("\nFP candidates' assigned vessel segments (top 15):", flush=True)
    for seg, n in fp_assigned_classes.most_common(15):
        print(f"  {seg}: {n}", flush=True)

    out = {
        "tp_candidates_scored": tp_total_candidates, "fp_candidates": fp_total,
        "tp_accuracy_by_method": {m: {**tp_results[m],
                                      "accuracy": (tp_results[m]["correct"] / tp_results[m]["total"]
                                                  if tp_results[m]["total"] else None)}
                                  for m in ("B", "C")},
        "fp_assigned_segments_top15": fp_assigned_classes.most_common(15),
        "oracle_reference_accuracy": 0.763,
    }
    out_path = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/reports/"
                    "tp_fp_geometric_accuracy_center5.json")
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
