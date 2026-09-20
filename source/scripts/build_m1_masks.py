"""M1 mask builder: detector boxes -> ellipsoid -> HONEST 52-class label -> 0-52 uint8 volume.

2026-08-15, peer-directed. Same box-fill mechanics as
`build_e2e_placeholder_pipeline.py` (imported, not re-derived -- box
coordinate conversion and ellipsoid fill are already validated there). The
only change: instead of a fixed placeholder class, each detected lesion's
centroid is classified via `location_classifier_m1.py`'s axis-corrected
position-only k-NN (trained on centers != the held-out center, so this is
still a fair LOCO evaluation, no leakage from the held-out center into the
classifier's training pool).

Never reads vessel_masks/ -- classifier trains from location_masks/ only,
detector reads only preprocessed image data. No vessel_mask path appears
anywhere in this script or its imports.
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
from location_classifier_m1 import LocationClassifier  # noqa: E402

SCORE_THRESHOLD = 0.3
MAX_BOXES_PER_CASE = 20


def build_case_mask(boxes_pkl: Path, reference_image: Path, classifier: LocationClassifier) -> np.ndarray:
    ref = nib.load(reference_image)
    shape = ref.shape
    shape_arr = np.asarray(shape, dtype=float)
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

    n_boxes = len(selected)
    for index in reversed(selected):  # lowest score first, so highest ends up on top
        low, high = box_to_native_bounds(boxes[index])
        centroid = (low + high) / 2.0
        class_id = classifier.classify(centroid, shape_arr)
        fill_ellipsoid(mask, low, high, class_id)
    return mask, n_boxes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boxes-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--held-out-center", required=True)
    parser.add_argument("--case-manifest", type=Path,
                        default=Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task1/"
                                    "artifacts/phase0/case_manifest.json"))
    parser.add_argument("--case-id", action="append", default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.case_manifest.read_text())
    cases_by_id = {c["case_id"]: c for c in manifest["cases"]}

    print(f"building LOCO position classifier, excluding {args.held_out_center}...", flush=True)
    classifier = LocationClassifier(exclude_centers={args.held_out_center})

    available_boxes = sorted(p.stem.removesuffix("_boxes") for p in args.boxes_dir.glob("*_boxes.pkl"))
    case_ids = args.case_id if args.case_id else available_boxes
    print(f"building masks for {len(case_ids)} cases", flush=True)

    written = 0
    n_boxes_per_case = []
    for i, case_id in enumerate(case_ids):
        if i % 10 == 0:
            print(f"progress: {i}/{len(case_ids)}", flush=True)
        boxes_pkl = args.boxes_dir / f"{case_id}_boxes.pkl"
        if not boxes_pkl.is_file():
            continue
        case = cases_by_id.get(case_id)
        if case is None:
            continue
        mask, n_boxes = build_case_mask(boxes_pkl, Path(case["image"]), classifier)
        n_boxes_per_case.append(n_boxes)
        ref = nib.load(case["image"])
        out_img = nib.Nifti1Image(mask, ref.affine, ref.header)
        out_img.set_data_dtype(np.uint8)
        out_img.to_filename(args.output_dir / f"{case_id}.nii.gz")
        written += 1

    print(f"wrote {written} M1 prediction masks to {args.output_dir}", flush=True)
    print(f"n_boxes_per_case: mean={np.mean(n_boxes_per_case):.2f} "
         f"median={np.median(n_boxes_per_case):.1f} "
         f"min={np.min(n_boxes_per_case)} max={np.max(n_boxes_per_case)} "
         f"zero_box_cases={sum(1 for n in n_boxes_per_case if n == 0)}/{len(n_boxes_per_case)}",
         flush=True)
    (args.output_dir / "n_boxes_per_case.json").write_text(
        json.dumps(dict(zip(case_ids, n_boxes_per_case)), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
