"""Vessel-based end-to-end mask builder: detector boxes -> ellipsoid -> vessel-
geometric 52-class label -> 0-52 uint8 volume. 2026-08-17, peer-directed CT/MR
honest-baseline work (successor to build_m1_masks.py, which used the
registration-free centroid-kNN floor from Experiment B; this uses a real or
predicted vessel mask instead, once one is available).

Same box-fill mechanics as `build_e2e_placeholder_pipeline.py` (imported, not
re-derived). The only change from `build_m1_masks.py`: classification comes
from `vessel_geometric_classifier.py` (nearest-vessel-segment -> class,
method C) instead of position-only k-NN, and needs a per-case vessel mask
(oracle GT or TA36-predicted, selected via --vessel-mask-dir).

For each detected box, the ellipsoid-filled region itself is used as the
"lesion mask" fed to the classifier (not a GT lesion mask -- there is none at
test time), consistent with how a real deployed pipeline would have to work.

2026-08-17, peer-caught operating-point issue: nnDetection scores are not
calibrated probabilities, so a fixed --score-threshold does not transfer
across models -- CT at 0.3 hit the MAX_BOXES_PER_CASE=20 cap on the median
case (~16x over-prediction vs ~1.2 true lesions/case), badly diluting the
official metrics' TP+FN+FP denominator. Fix: --k-values writes one snapshot
subdirectory per cutoff (output-dir/k{K:02d}/), sharing the expensive parts
(reference image load, vessel mask load, per-case nearest_label_volume
distance transform, per-box classification) across every k in one pass --
only the trivial ellipsoid redraw repeats per k. --score-threshold is now a
CLI flag (default 0.3, unchanged from before if not given) instead of a
hardcoded module constant that couldn't be swept without editing code.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from build_e2e_placeholder_pipeline import box_to_native_bounds, fill_ellipsoid  # noqa: E402
from vessel_geometric_classifier import VesselGeometricClassifier, nearest_label_lookup  # noqa: E402

SCORE_THRESHOLD = 0.3  # default, overridable via --score-threshold
MAX_BOXES_PER_CASE = 20  # default cap, also the ceiling for --k-values
K_VALUES_DEFAULT = (20,)  # single snapshot at the old default, backward compatible


def ellipsoid_bool_mask(shape: tuple, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Same ellipsoid geometry as fill_ellipsoid, but returned as a boolean
    mask over the FULL volume so it can be fed straight to the vessel
    classifier (which expects a lesion_mask the same shape as vessel_arr)."""
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
    normalized = (((xx - centre[0]) / radii[0]) ** 2 +
                 ((yy - centre[1]) / radii[1]) ** 2 +
                 ((zz - centre[2]) / radii[2]) ** 2)
    inside = normalized <= 1.0
    sub = mask[lo_clip[0]:hi_clip[0], lo_clip[1]:hi_clip[1], lo_clip[2]:hi_clip[2]]
    sub[inside] = True
    mask[lo_clip[0]:hi_clip[0], lo_clip[1]:hi_clip[1], lo_clip[2]:hi_clip[2]] = sub
    return mask


def classify_and_bound_boxes(boxes_pkl: Path, score_threshold: float, max_k: int,
                             vessel_arr: np.ndarray | None, ref_shape: tuple,
                             classifier: VesselGeometricClassifier) -> list[tuple]:
    """Returns up to max_k (low, high, score, class_id) tuples, score-descending,
    for boxes at/above score_threshold. Classification and the expensive
    per-case nearest_label_volume are each computed ONCE here, shared by
    every k-snapshot the caller builds from this list."""
    with boxes_pkl.open("rb") as handle:
        prediction = pickle.load(handle)
    boxes = np.asarray(prediction["pred_boxes"], dtype=float)
    scores = np.asarray(prediction["pred_scores"], dtype=float)
    if not prediction.get("restore", False):
        raise ValueError(f"{boxes_pkl}: boxes are not restored to original image space")

    order = np.argsort(-scores)
    selected = []
    for index in order:
        if len(selected) >= max_k:
            break
        if scores[index] < score_threshold:
            break
        selected.append(index)

    nearest_label_volume = None
    if vessel_arr is not None and vessel_arr.shape == ref_shape:
        nearest_label_volume = nearest_label_lookup(vessel_arr)  # once per case

    fallback_class = classifier.class_id_by_name.get(classifier.most_common_class)
    results = []
    for index in selected:
        low, high = box_to_native_bounds(boxes[index])
        if nearest_label_volume is not None:
            lesion_mask = ellipsoid_bool_mask(ref_shape, low, high)
            class_id = classifier.classify_from_nearest_label_volume(lesion_mask, nearest_label_volume)
        else:
            class_id = None
        if class_id is None:
            class_id = fallback_class
        results.append((low, high, float(scores[index]), class_id))
    return results  # score-descending order


def build_case_masks(boxes_pkl: Path, reference_image: Path, vessel_mask_path: Path,
                     classifier: VesselGeometricClassifier, score_threshold: float,
                     k_values: tuple) -> dict:
    """Returns {k: (mask_array, n_boxes_used)} for every k in k_values, sharing
    all expensive per-case work (image/vessel load, nearest_label_volume,
    per-box classification) across the whole k-sweep."""
    ref = nib.load(reference_image)
    shape = ref.shape
    vessel_arr = nib.load(vessel_mask_path).get_fdata() if vessel_mask_path.is_file() else None

    max_k = max(k_values)
    classified = classify_and_bound_boxes(boxes_pkl, score_threshold, max_k, vessel_arr, shape, classifier)

    out = {}
    for k in k_values:
        mask = np.zeros(shape, dtype=np.uint8)
        top_k = classified[:k]
        # lowest-score-first draw order so the highest-confidence box ends up
        # on top at any overlap (same convention as build_e2e_placeholder_pipeline.py)
        for low, high, _score, class_id in reversed(top_k):
            fill_ellipsoid(mask, low, high, class_id)
        out[k] = (mask, len(top_k))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boxes-dir", type=Path, required=True)
    parser.add_argument("--vessel-mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--held-out-center", action="append", required=True,
                        help="center(s) to exclude from the classifier's training pool; "
                             "repeat for multiple (e.g. CT fold2's val mixes center2+center4)")
    parser.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD)
    parser.add_argument("--k-values", type=str, default=None,
                        help="comma-separated box-count cutoffs, e.g. '1,2,3,5,10,20'. "
                             "Each writes its own subdirectory output-dir/k{K:02d}/. "
                             "Default: single snapshot at k=20 written directly to "
                             "output-dir (old single-point behavior, unchanged).")
    parser.add_argument("--case-manifest", type=Path,
                        default=Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task1/"
                                    "artifacts/phase0/case_manifest.json"))
    parser.add_argument("--case-id", action="append", default=None)
    args = parser.parse_args()

    single_mode = args.k_values is None
    k_values = (MAX_BOXES_PER_CASE,) if single_mode else tuple(
        sorted(int(k) for k in args.k_values.split(",")))

    manifest = json.loads(args.case_manifest.read_text())
    cases_by_id = {c["case_id"]: c for c in manifest["cases"]}

    print(f"building vessel-geometric classifier, excluding {args.held_out_center}...", flush=True)
    classifier = VesselGeometricClassifier(exclude_centers=set(args.held_out_center))

    if single_mode:
        out_dirs = {k_values[0]: args.output_dir}
    else:
        out_dirs = {k: args.output_dir / f"k{k:02d}" for k in k_values}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    available_boxes = sorted(p.stem.removesuffix("_boxes") for p in args.boxes_dir.glob("*_boxes.pkl"))
    case_ids = args.case_id if args.case_id else available_boxes
    print(f"building masks for {len(case_ids)} cases, k_values={k_values}, "
         f"score_threshold={args.score_threshold}", flush=True)

    written = {k: 0 for k in k_values}
    n_boxes_per_case = {k: {} for k in k_values}
    n_no_vessel_mask = 0
    for i, case_id in enumerate(case_ids):
        if i % 10 == 0:
            print(f"progress: {i}/{len(case_ids)}", flush=True)
        boxes_pkl = args.boxes_dir / f"{case_id}_boxes.pkl"
        if not boxes_pkl.is_file():
            continue
        case = cases_by_id.get(case_id)
        if case is None:
            continue
        vessel_mask_path = args.vessel_mask_dir / f"{case_id}.nii.gz"
        if not vessel_mask_path.is_file():
            n_no_vessel_mask += 1
        masks_by_k = build_case_masks(boxes_pkl, Path(case["image"]), vessel_mask_path,
                                      classifier, args.score_threshold, k_values)
        ref = nib.load(case["image"])
        for k, (mask, n_boxes) in masks_by_k.items():
            n_boxes_per_case[k][case_id] = n_boxes
            out_img = nib.Nifti1Image(mask, ref.affine, ref.header)
            out_img.set_data_dtype(np.uint8)
            out_img.to_filename(out_dirs[k] / f"{case_id}.nii.gz")
            written[k] += 1

    for k in k_values:
        print(f"k={k}: wrote {written[k]} masks to {out_dirs[k]}", flush=True)
        counts = list(n_boxes_per_case[k].values())
        if counts:
            print(f"  n_boxes_per_case: mean={np.mean(counts):.2f} median={np.median(counts):.1f} "
                 f"min={np.min(counts)} max={np.max(counts)} "
                 f"zero_box_cases={sum(1 for n in counts if n == 0)}/{len(counts)}", flush=True)
        (out_dirs[k] / "n_boxes_per_case.json").write_text(
            json.dumps(n_boxes_per_case[k], indent=2) + "\n")
    print(f"cases with no vessel mask file (fell back to most-common class): "
         f"{n_no_vessel_mask}/{len(case_ids)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
