"""Minimal end-to-end pipeline: detector boxes -> ellipsoid mask -> placeholder class -> 0-52 uint8 volume.

2026-08-14, peer-directed priority pivot: build the crude full pipeline NOW
rather than polish any one stage first. Every stage below is deliberately
the cheapest thing that produces a real, scoreable output -- the point is to
get a first real 6-metric TopAneu score and eliminate "the pipeline doesn't
actually fit together" as an unknown, before investing in quality anywhere.

Detection: whatever detector is available (first this session: the zero-shot
Task020FG model on TopAneu-MR; later: TopAneu-MR's own trained checkpoint,
even from an early, unconverged epoch -- interface is identical).

Segmentation: NONE. Task 2's hit rule is any-overlap (IoU > 0), and only
DICE/VS/HD95 of the 6 metrics care about shape -- the user's own priority
order puts Dice last. So this fills an axis-aligned ELLIPSOID inscribed in
each detected box directly into the output volume. That's enough to get real
PRECISION/RECALL/MCC numbers; DICE will be poor, which is expected and fine
at this stage.

Classification (52-class): every detected lesion gets the SAME fixed
placeholder class, chosen from the 9 classes confirmed (via
fp_marginal_cost_probe.py and loco_coverage_ceiling.py, checked in EVERY
LOCO held-out center) to have zero true instances anywhere in the 417-case
population -- misassigning to one of these costs a one-time saturating HD95
penalty and nothing on the other 5 metrics, unlike guessing a real/common
class (see reports/fp_marginal_cost_probe.md).

Box coordinate handling copies build_adam90_comparison.py's `box_candidates`
exactly (SimpleITK (z,y,x) box layout -> nibabel (x,y,z) voxel low/high),
since that conversion is already proven correct in this codebase -- not
re-derived here.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import nibabel as nib
import numpy as np

PLACEHOLDER_CLASS = 3  # one of the 9 globally-zero-instance classes; arbitrary pick among them
SCORE_THRESHOLD = 0.3  # keep boxes at least this confident; a placeholder operating point
MAX_BOXES_PER_CASE = 20  # generous cap, not a tuned operating point


def box_to_native_bounds(box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SimpleITK (z,y,x) box layout -> nibabel (x,y,z) voxel low/high.

    Identical to build_adam90_comparison.py's box_candidates -- same source
    format (nnDetection's restored pred_boxes), same conversion, not
    re-derived.
    """
    low = np.asarray([box[4], box[1], box[0]], dtype=float)
    high = np.asarray([box[5], box[3], box[2]], dtype=float)
    return low, high


def fill_ellipsoid(mask: np.ndarray, low: np.ndarray, high: np.ndarray, value: int) -> None:
    """Fills the axis-aligned ellipsoid inscribed in [low, high] with `value`.

    Voxel grid coordinates, in-place. Clips to the volume bounds.
    """
    centre = (low + high) / 2.0
    radii = np.maximum((high - low) / 2.0, 0.5)  # avoid zero-radius for degenerate boxes

    lo_clip = np.maximum(np.floor(low).astype(int), 0)
    hi_clip = np.minimum(np.ceil(high).astype(int), np.asarray(mask.shape))
    if np.any(lo_clip >= hi_clip):
        return

    xs = np.arange(lo_clip[0], hi_clip[0])
    ys = np.arange(lo_clip[1], hi_clip[1])
    zs = np.arange(lo_clip[2], hi_clip[2])
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    normalized = (((xx - centre[0]) / radii[0]) ** 2 +
                 ((yy - centre[1]) / radii[1]) ** 2 +
                 ((zz - centre[2]) / radii[2]) ** 2)
    inside = normalized <= 1.0
    sub = mask[lo_clip[0]:hi_clip[0], lo_clip[1]:hi_clip[1], lo_clip[2]:hi_clip[2]]
    sub[inside] = value
    mask[lo_clip[0]:hi_clip[0], lo_clip[1]:hi_clip[1], lo_clip[2]:hi_clip[2]] = sub


def build_case_mask(boxes_pkl: Path, reference_image: Path) -> np.ndarray:
    """One case's 0-52 uint8 prediction volume, matching reference_image's own grid."""
    ref = nib.load(reference_image)
    shape = ref.shape
    mask = np.zeros(shape, dtype=np.uint8)

    with boxes_pkl.open("rb") as handle:
        prediction = pickle.load(handle)
    boxes = np.asarray(prediction["pred_boxes"], dtype=float)
    scores = np.asarray(prediction["pred_scores"], dtype=float)
    if not prediction.get("restore", False):
        raise ValueError(f"{boxes_pkl}: boxes are not restored to original image space")

    order = np.argsort(-scores)
    selected = []
    for index in order:
        if len(selected) >= MAX_BOXES_PER_CASE:
            break
        if scores[index] < SCORE_THRESHOLD:
            break
        selected.append(index)
    # Draw lowest-score-first, highest-score-last: the output is a single-
    # channel mask where a later fill overwrites an earlier one at any
    # overlap, so drawing in ascending score order keeps the most confident
    # box on top instead of letting weaker boxes clobber it (2026-08-14,
    # peer-caught bug -- the original loop drew descending-score-first,
    # which is backwards).
    for index in reversed(selected):
        low, high = box_to_native_bounds(boxes[index])
        fill_ellipsoid(mask, low, high, PLACEHOLDER_CLASS)
    return mask


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boxes-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path,
                        default=Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task1/"
                                    "artifacts/phase0/case_manifest.json"))
    parser.add_argument("--case-id", action="append", default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.case_manifest.read_text())
    cases_by_id = {c["case_id"]: c for c in manifest["cases"]}

    available_boxes = sorted(p.stem.removesuffix("_boxes") for p in args.boxes_dir.glob("*_boxes.pkl"))
    case_ids = args.case_id if args.case_id else available_boxes
    print(f"building masks for {len(case_ids)} cases", flush=True)

    written = 0
    for i, case_id in enumerate(case_ids):
        if i % 50 == 0:
            print(f"progress: {i}/{len(case_ids)}", flush=True)
        boxes_pkl = args.boxes_dir / f"{case_id}_boxes.pkl"
        if not boxes_pkl.is_file():
            continue
        case = cases_by_id.get(case_id)
        if case is None:
            continue
        mask = build_case_mask(boxes_pkl, Path(case["image"]))
        ref = nib.load(case["image"])
        out_img = nib.Nifti1Image(mask, ref.affine, ref.header)
        out_img.set_data_dtype(np.uint8)
        out_img.to_filename(args.output_dir / f"{case_id}.nii.gz")
        written += 1

    print(f"wrote {written} placeholder prediction masks to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
