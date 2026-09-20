"""Decisive, training-free experiment: does a geometric rerank signal close the K=1 vs K=200 recall gap?

2026-08-14, peer-directed. K=1 pure box-vs-GT recall is 45.2%, K=200
("everything's in the list somewhere") is 89.3% -- a 44-point gap that is
purely about RANKING, not detection or classification. This is exactly what
stage 2 rerank exists to fix (the architecture the user specified: detect
-> rerank -> segment), and we already have a training-free rerank signal
available: vessel geometry.

geo_score(candidate) = f(distance to nearest vessel) x g(nearest vessel
segment's aneurysm prior probability), where g is estimated purely from the
417-case training population's location_class_counts (no model, no GPU) --
aneurysms cluster heavily on a handful of segments (MCA bifurcation, ICA
terminus/PComm region, ACom, basilar tip), matching clinical literature.

Test: rerank each scan's candidates by geo_score alone, and separately by
detector_score + geo_score, then recompute pure box-vs-GT recall@{1,3,5} on
the reranked order. Compare against the baseline (detector-score-only)
45.2%/60.7%/66.7%. No masks, no HD95, no arena -- this only needs
candidates, scores, GT, and vessel masks, all already on disk.
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
from phase2b_location_class_to_vessel_map import LOCATION_TO_VESSEL  # noqa: E402
from phase2b_geometric_attachment import load_labels  # noqa: E402
from build_e2e_placeholder_pipeline import box_to_native_bounds  # noqa: E402

BOXES_DIR = Path("/home/jovyan/rtx4claude-datavol-1/new_aneurysms/artifacts/nndet_task020_zeroshot_topaneu_mr")
MANIFEST = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task1/artifacts/phase0/case_manifest.json")
LOCO_SPLITS = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/configs/loco_splits.json")
DATA_ROOT = Path("/home/jovyan/rtx4claude-datavol-1/data_topaneu26")
K_VALUES = (1, 3, 5)
CANDIDATES_PER_SCAN = 20
DISTANCE_SCALE_MM = 3.0  # f(d) = exp(-d/scale); aneurysms are typically within a few mm of the parent vessel
GEO_WEIGHT = 1.0  # overridden by driver script for the weight sweep
COMBINE_MODE = "symmetric"  # "symmetric": det_z + w*geo_z; "asymmetric": det_z + min(0, w*geo_z) -- geo can only penalize


def build_segment_prior(manifest: dict) -> dict[str, float]:
    """segment -> P(aneurysm here | training population), purely from counts, no model."""
    id_to_name = load_labels(DATA_ROOT / "location_mapping.json")
    cc = manifest["location_class_counts"]
    segment_counts = Counter()
    for cid_str, n in cc.items():
        name = id_to_name.get(int(cid_str))
        if name is None or name not in LOCATION_TO_VESSEL:
            continue
        _, segments = LOCATION_TO_VESSEL[name]
        for seg in segments:
            segment_counts[seg] += n
    total = sum(segment_counts.values())
    return {seg: n / total for seg, n in segment_counts.items()} if total else {}


def nearest_label_and_distance(vessel_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    background = vessel_arr == 0
    distances, indices = ndimage.distance_transform_edt(background, return_indices=True, return_distances=True)
    return vessel_arr[tuple(indices)], distances


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    cases_by_id = {c["case_id"]: c for c in manifest["cases"]}
    loco = json.loads(LOCO_SPLITS.read_text())
    val_ids = [c for c in loco["folds"]["center5"]["val"]
              if cases_by_id.get(c, {}).get("modality") == "mr"]

    vessel_names = load_labels(DATA_ROOT / "vessel_mapping.json")
    segment_prior = build_segment_prior(manifest)
    print(f"segment prior built from {sum(manifest['location_class_counts'].values())} total training "
         f"lesion-instances, {len(segment_prior)} segments", flush=True)

    hits_baseline = {k: 0 for k in K_VALUES}
    hits_geo_only = {k: 0 for k in K_VALUES}
    hits_combined = {k: 0 for k in K_VALUES}
    total_lesions = 0
    max_k = max(K_VALUES)

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
        lesion_voxel_sets = [np.argwhere(labeled == cid) for cid in range(1, n + 1)]
        total_lesions += len(lesion_voxel_sets)

        vessel_arr_zyx = sitk.GetArrayFromImage(sitk.ReadImage(str(case["vessel_mask"])))
        vessel_arr = np.transpose(vessel_arr_zyx, (2, 1, 0))
        nearest_label_volume, distance_volume = nearest_label_and_distance(vessel_arr)

        with boxes_pkl.open("rb") as handle:
            prediction = pickle.load(handle)
        boxes = np.asarray(prediction["pred_boxes"], dtype=float)
        scores = np.asarray(prediction["pred_scores"], dtype=float)
        order = np.argsort(-scores)[:CANDIDATES_PER_SCAN]

        candidates = []  # (idx, low, high, detector_score, geo_score)
        for idx in order:
            low, high = box_to_native_bounds(boxes[idx])
            centre = np.rint((low + high) / 2.0).astype(int)
            centre = np.clip(centre, 0, np.asarray(shape) - 1)
            dist = float(distance_volume[tuple(centre)])
            vessel_label = int(nearest_label_volume[tuple(centre)])
            seg_name = vessel_names.get(vessel_label, None) if vessel_label != 0 else None
            prior = segment_prior.get(seg_name, 0.0) if seg_name else 0.0
            geo_score = float(np.exp(-dist / DISTANCE_SCALE_MM)) * prior
            candidates.append((idx, low, high, float(scores[idx]), geo_score))

        def first_hit_rank_per_lesion(ordered_candidates, lesion_voxels) -> int | None:
            for rank, (idx, low, high, det_s, geo_s) in enumerate(ordered_candidates):
                inside = np.all((lesion_voxels >= low) & (lesion_voxels <= high), axis=1)
                if inside.any():
                    return rank
            return None

        # geo-only rerank
        geo_ordered = sorted(candidates, key=lambda c: -c[4])
        # combined: z-normalize each within-scan, sum
        det_scores = np.array([c[3] for c in candidates])
        geo_scores = np.array([c[4] for c in candidates])
        det_z = (det_scores - det_scores.mean()) / (det_scores.std() + 1e-9)
        geo_z = (geo_scores - geo_scores.mean()) / (geo_scores.std() + 1e-9)
        if COMBINE_MODE == "asymmetric":
            # geo can only ever penalize a candidate (pull an anatomically
            # implausible one down), never boost one above the detector's
            # own top pick -- peer's hypothesis: geo is good at excluding
            # the impossible, not at picking the most likely.
            adjustment = np.minimum(0.0, GEO_WEIGHT * geo_z)
        else:
            adjustment = GEO_WEIGHT * geo_z
        combined_order_idx = sorted(range(len(candidates)), key=lambda i: -(det_z[i] + adjustment[i]))
        combined_ordered = [candidates[i] for i in combined_order_idx]

        # PER-LESION recall (matches probe_pure_box_recall.py's convention
        # exactly, so these numbers are directly comparable to the earlier
        # 45.2%/60.7%/66.7% baseline) -- each lesion gets its own first-hit
        # rank under each of the three orderings.
        for lesion_voxels in lesion_voxel_sets:
            rank_baseline = first_hit_rank_per_lesion(candidates, lesion_voxels)
            rank_geo = first_hit_rank_per_lesion(geo_ordered, lesion_voxels)
            rank_combined = first_hit_rank_per_lesion(combined_ordered, lesion_voxels)
            for k in K_VALUES:
                if rank_baseline is not None and rank_baseline < k:
                    hits_baseline[k] += 1
                if rank_geo is not None and rank_geo < k:
                    hits_geo_only[k] += 1
                if rank_combined is not None and rank_combined < k:
                    hits_combined[k] += 1

    print(f"\ntotal lesions: {total_lesions}", flush=True)
    print(f"{'K':>5} {'baseline':>12} {'geo-only':>12} {'combined':>12}")
    for k in K_VALUES:
        b = hits_baseline[k] / total_lesions if total_lesions else 0.0
        g = hits_geo_only[k] / total_lesions if total_lesions else 0.0
        c = hits_combined[k] / total_lesions if total_lesions else 0.0
        print(f"{k:>5} {b:>11.1%} {g:>11.1%} {c:>11.1%}  "
             f"(hits: {hits_baseline[k]}/{hits_geo_only[k]}/{hits_combined[k]})")

    out = {"total_lesions": total_lesions,
          "hits_by_k": {str(k): {"baseline": hits_baseline[k], "geo_only": hits_geo_only[k],
                                 "combined": hits_combined[k],
                                 "recall_baseline": hits_baseline[k] / total_lesions if total_lesions else None,
                                 "recall_geo_only": hits_geo_only[k] / total_lesions if total_lesions else None,
                                 "recall_combined": hits_combined[k] / total_lesions if total_lesions else None}
                       for k in K_VALUES}}
    out_path = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/reports/"
                    "geometric_rerank_probe_center5.json")
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
