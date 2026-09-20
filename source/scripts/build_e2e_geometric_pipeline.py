"""Variant B: detected boxes -> ellipsoid -> GEOMETRIC class assignment (oracle vessel masks).

2026-08-14, peer-directed. Variant A (build_e2e_placeholder_pipeline.py) uses
a fixed never-true class for every detection -- necessarily scores near-zero
since it can never produce a TP on any real class, and is a plumbing sanity
check only, not an informative baseline (the peer's correction, made before
seeing Variant A's actual score, to head off misreading it as one).

This variant instead assigns each DETECTED box's class via the already-
validated geometric method (C_nearest_centroid from
phase2b_geometric_attachment.py, 76.3% pooled top-1 when the true class is
known and geometry is queried at the TRUE lesion location) against the
case's own GROUND-TRUTH vessel_mask -- TopAneu ships vessel_masks for all
417 training cases, so this is available for LOCO-local evaluation without
waiting on an external vessel model. The key methodological difference from
the original phase2b measurement: geometry here is queried at the DETECTED
box's centroid, not the GT lesion's own location -- this is the first real
test of whether geometric attachment still works when anchored to imperfect
detection rather than oracle localization.

Reverse mapping (vessel segment -> class) is necessarily approximate:
LOCATION_TO_VESSEL maps class -> segment set, and multiple classes can
share a segment (junction/position groups by design -- segment identity
alone can't disambiguate them, documented in phase2b_geometric_attachment's
own module docstring). Where a segment has multiple candidate classes, this
picks the GLOBALLY MOST COMMON one (from case_manifest.json's
location_class_counts) as a plausible tie-break, not a resolved answer --
flagged in the output, not hidden.

Ceilings that apply to this variant's score, to state alongside it (not
optional context, required for it not to be misread later):
  1. The held-out center's own coverage ceiling (n_distinct_classes/52) --
     no system can exceed it, oracle or not.
  2. The zero-shot detector's own resolution penalty (trained at 0.70mm,
     TopAneu-MR's native spacing is ~0.50mm) depresses detection recall.
  3. Vessel masks here are ORACLE (ground truth) -- this is an upper bound
     on what any real vessel model would deliver, particularly on the
     small distal vessels the long-tail classes tend to sit on.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

sys.path.insert(0, "/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/scripts")
from phase2b_location_class_to_vessel_map import LOCATION_TO_VESSEL  # noqa: E402
from build_e2e_placeholder_pipeline import box_to_native_bounds, fill_ellipsoid  # noqa: E402

DATA_ROOT = Path("/home/jovyan/rtx4claude-datavol-1/data_topaneu26")
MANIFEST = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task1/artifacts/phase0/case_manifest.json")
# 2026-08-14, peer-requested top-K operating-point sweep: no score threshold,
# pure per-scan rank cutoff, so each K answers "keep this many candidates per
# scan" cleanly without a confounding score cutoff mixed in.
SCORE_THRESHOLD = 0.0
MAX_BOXES_PER_CASE = 20  # overridden by --top-k


def load_labels(mapping_path: Path) -> dict[int, str]:
    labels = json.loads(mapping_path.read_text())["labels"]
    return {v: k for k, v in labels.items()}


def build_segment_to_class(class_counts: dict[str, int], class_name_to_id: dict[str, int]) -> dict[str, str]:
    """segment name -> most-globally-common class anchored there (single or junction).

    2026-08-14, two bugs found and fixed here, not just one:

    (1) PRE-EXISTING bug (predates the peer's exclusion request): manifest's
    `location_class_counts` is keyed by class ID as a string ("1", "2", ...),
    but this function was looking it up by class NAME ("R-1.1 VA trunk", ...)
    -- every lookup silently returned the 0 default, so the documented
    "most globally common" tie-break was never actually comparing real
    counts; `max()` over an all-zero list just returns whichever class
    happened to be listed first for that segment. Fixed by converting
    cls_name -> id via `class_name_to_id` before the count lookup.

    (2) Peer-requested exclusion: the 9 classes with zero instances across
    all 417 training cases AND all 4 LOCO held-out centers
    (fp_marginal_cost_probe.py, loco_coverage_ceiling.py) are dropped from
    candidacy entirely, not just deprioritized -- class 52 (L-5.3
    Distal-M2M3) was found in the top-10 most-assigned classes in a real
    run despite having 0 real instances anywhere, because bug (1) meant it
    was never actually losing a tie-break to begin with.
    """
    candidates: dict[str, list[tuple[str, int]]] = {}
    for cls_name, (group, segments) in LOCATION_TO_VESSEL.items():
        cls_id = str(class_name_to_id[cls_name])
        n = class_counts.get(cls_id, 0)
        for seg in segments:
            candidates.setdefault(seg, []).append((cls_name, n))
    result = {}
    for seg, opts in candidates.items():
        non_zero_global = [(name, n) for name, n in opts if n > 0]
        if not non_zero_global:
            continue  # every candidate for this segment is a zero-instance class -- exclude, no assignment
        result[seg] = max(non_zero_global, key=lambda kv: kv[1])[0]
    return result


def nearest_label_lookup(vessel_arr: np.ndarray) -> np.ndarray:
    background = vessel_arr == 0
    _, indices = ndimage.distance_transform_edt(background, return_indices=True)
    return vessel_arr[tuple(indices)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boxes-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--held-out-center", required=True)
    parser.add_argument("--mode", choices=("topk", "one-per-class"), default="topk")
    parser.add_argument("--top-k", type=int, default=MAX_BOXES_PER_CASE,
                        help="[topk mode] keep this many highest-score candidates per scan")
    parser.add_argument("--class-score-threshold", type=float, default=0.0,
                        help="[one-per-class mode] minimum score for a class's single best "
                             "candidate to be drawn at all")
    parser.add_argument("--max-classes-per-scan", type=int, default=None,
                        help="[one-per-class mode] keep only the N classes whose best candidate "
                             "scores highest, per scan -- a directly interpretable alternative "
                             "to --class-score-threshold (peer-requested, 2026-08-14: threshold "
                             "is data-dependent and opaque, N-classes sweeps the actually "
                             "relevant range directly). Applied after the threshold filter.")
    args = parser.parse_args()
    top_k = args.top_k
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST.read_text())
    cases_by_id = {c["case_id"]: c for c in manifest["cases"]}
    class_counts = manifest["location_class_counts"]

    location_names = load_labels(DATA_ROOT / "location_mapping.json")
    vessel_names = load_labels(DATA_ROOT / "vessel_mapping.json")
    vessel_name_to_id = {v: k for k, v in vessel_names.items()}
    class_name_to_id = {v: k for k, v in location_names.items()}
    segment_to_class = build_segment_to_class(class_counts, class_name_to_id)

    loco = json.loads(Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/"
                           "configs/loco_splits.json").read_text())
    val_ids = [c for c in loco["folds"][args.held_out_center]["val"]
              if cases_by_id.get(c, {}).get("modality") == "mr"]
    print(f"{len(val_ids)} MR cases in held-out {args.held_out_center}", flush=True)

    written = 0
    skipped_no_boxes = 0
    unresolved_segments = set()
    for i, case_id in enumerate(val_ids):
        if i % 20 == 0:
            print(f"progress: {i}/{len(val_ids)}", flush=True)
        boxes_pkl = args.boxes_dir / f"{case_id}_boxes.pkl"
        if not boxes_pkl.is_file():
            skipped_no_boxes += 1
            continue
        case = cases_by_id[case_id]
        ref = nib.load(case["image"])
        shape = ref.shape
        mask = np.zeros(shape, dtype=np.uint8)

        vessel_path = Path(case["vessel_mask"])
        vessel_arr_zyx = sitk.GetArrayFromImage(sitk.ReadImage(str(vessel_path)))
        # location_masks/images are read via nibabel (x,y,z); vessel_masks share
        # that same file's geometry, so transpose zyx -> xyz to match `mask`.
        vessel_arr = np.transpose(vessel_arr_zyx, (2, 1, 0))
        nearest_label_volume = nearest_label_lookup(vessel_arr)

        with boxes_pkl.open("rb") as handle:
            prediction = pickle.load(handle)
        boxes = np.asarray(prediction["pred_boxes"], dtype=float)
        scores = np.asarray(prediction["pred_scores"], dtype=float)
        order = np.argsort(-scores)

        # Compute every candidate's geometric class assignment ONCE,
        # independent of selection mode -- topk and one-per-class both
        # consume the same (index, low, high, class_id) triples, just with
        # different selection logic on top.
        classified: list[tuple[int, np.ndarray, np.ndarray, int]] = []
        for index in order:
            low, high = box_to_native_bounds(boxes[index])
            centre = np.rint((low + high) / 2.0).astype(int)
            centre = np.clip(centre, 0, np.asarray(shape) - 1)
            vessel_label = int(nearest_label_volume[tuple(centre)])
            if vessel_label == 0:
                continue
            segment_name = vessel_names.get(vessel_label, None)
            predicted_class_name = segment_to_class.get(segment_name) if segment_name else None
            if predicted_class_name is None:
                unresolved_segments.add(segment_name)
                continue
            classified.append((index, low, high, class_name_to_id[predicted_class_name]))

        if args.mode == "topk":
            selected = [(low, high, cid) for idx, low, high, cid in classified
                       if idx in set(order[:top_k]) and scores[idx] >= SCORE_THRESHOLD]
        else:  # one-per-class: highest-scoring candidate per predicted class wins
            best_per_class: dict[int, tuple[float, np.ndarray, np.ndarray]] = {}
            for idx, low, high, cid in classified:
                score = scores[idx]
                if score < args.class_score_threshold:
                    continue
                if cid not in best_per_class or score > best_per_class[cid][0]:
                    best_per_class[cid] = (score, low, high)
            # ascending score so the highest-confidence class ends up drawn last
            ordered = sorted(best_per_class.items(), key=lambda kv: kv[1][0])
            if args.max_classes_per_scan is not None:
                ordered = ordered[-args.max_classes_per_scan:]  # keep the N highest-scoring
            selected = [(low, high, cid) for cid, (score, low, high) in ordered]

        # Draw lowest-score-first, highest-score-last (2026-08-14, peer-caught
        # bug): a single-channel mask means later fills overwrite earlier
        # ones, so the most confident box must be drawn last to survive
        # overlaps intact -- the original loop drew descending-score-first,
        # backwards from what a single-channel output needs. `selected` is
        # already in ascending-score order for both modes.
        for low, high, class_id in selected:
            fill_ellipsoid(mask, low, high, class_id)

        out_img = nib.Nifti1Image(mask, ref.affine, ref.header)
        out_img.set_data_dtype(np.uint8)
        out_img.to_filename(args.output_dir / f"{case_id}.nii.gz")
        written += 1

    print(f"wrote {written} geometric-assignment masks, {skipped_no_boxes} cases missing boxes",
         flush=True)
    if unresolved_segments:
        print(f"segments with no mapped class (left as background): {unresolved_segments}",
             flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
