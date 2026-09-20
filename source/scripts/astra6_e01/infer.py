from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .e01_common import (BASELINE_DIR, BOX_DIR, DATA, N_CLASSES, P, TA36_DIR, TOP_K,
                         THRESHOLD, compute_feature, fill_ellipsoid, load_boxes, load_nifti,
                         load_nifti_geometry,
                         nifti_output, select_candidates, sha256_tree, sitk_array, write_json,
                         vessel_geometry)


def predict_case(case_id: str, model: Any, run: Path, loc_names: dict[int, str], ves_names: dict[int, str]) -> dict[str, Any]:
    image_path = DATA / "images" / f"{case_id}_0000.nii.gz"
    vessel_path = TA36_DIR / f"{case_id}.nii.gz"
    boxes, scores, box_obj = load_boxes(BOX_DIR / f"{case_id}_boxes.pkl")
    affine, shape = load_nifti_geometry(image_path)
    vessel_aff, vessel_shape = load_nifti_geometry(vessel_path)
    if vessel_shape != shape or not np.allclose(vessel_aff, affine, atol=1e-4):
        raise ValueError(f"evaluation geometry mismatch {case_id}")
    geometry = vessel_geometry(vessel_path, shape, affine)
    modality = "MR"
    selected = select_candidates(boxes, scores)
    mask = np.zeros(shape, dtype=np.uint8)
    ledger = []
    for rank, (index, score, low, high) in enumerate(reversed(selected)):
        # Build all probabilities in fixed 1..52 ID order, filling unseen classes with zero.
        feature = compute_feature(geometry, affine, low, high, modality, False, None)
        p = np.zeros(N_CLASSES, dtype=np.float64)
        model_p = model.predict_proba(feature.reshape(1, -1))[0]
        for cls, prob in zip(model.classes_, model_p):
            p[int(cls) - 1] = float(prob)
        predicted = int(np.argmax(p) + 1)
        fill_ellipsoid(mask, low, high, predicted)
        ledger.append({"case_id": case_id, "original_index": index,
                      "detector_rank_desc": len(selected) - 1 - rank,
                      "score": score, "low": low.tolist(), "high": high.tolist(),
                      "predicted_class_id": predicted, "probabilities": p.tolist()})
    # Ledger is written in detector score descending order, while painting remains low-to-high.
    ledger.sort(key=lambda x: x["detector_rank_desc"])
    baseline = load_nifti(BASELINE_DIR / f"{case_id}.nii.gz")[0]
    if not np.array_equal(mask > 0, baseline > 0):
        raise RuntimeError(f"binary foreground invariant failed for {case_id}")
    out = run / "predictions/mr_center2_k05" / f"{case_id}.nii.gz"
    nifti_output(mask, image_path, out)
    return {"case_id": case_id, "selected": len(selected), "mask_path": str(out),
            "ledger": ledger, "foreground_voxels": int(np.count_nonzero(mask)),
            "baseline_foreground_voxels": int(np.count_nonzero(baseline)),
            "box_restore": bool(box_obj["restore"])}


def run_inference(run: Path, model: Any, eval_case_ids: list[str], loc_names: dict[int, str], ves_names: dict[int, str]) -> dict[str, Any]:
    run.mkdir(parents=True, exist_ok=True)
    ledger_path = run / "candidate_predictions.jsonl"
    rows = []
    all_info = []
    for i, case_id in enumerate(eval_case_ids):
        info = predict_case(case_id, model, run, loc_names, ves_names)
        all_info.append({k: v for k, v in info.items() if k != "ledger"})
        rows.extend(info["ledger"])
        if (i + 1) % 5 == 0:
            print(f"inference progress {i + 1}/{len(eval_case_ids)}", flush=True)
    ledger_path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
    prediction_hash = sha256_tree(run / "predictions")
    write_json(run / "prediction_manifest.json", {"case_ids": eval_case_ids, "n_cases": len(all_info),
                                                   "n_candidates": len(rows), "prediction_tree_sha256": prediction_hash,
                                                   "binary_foreground_checked": True,
                                                   "candidate_selection": {"threshold": THRESHOLD, "top_k": TOP_K}})
    write_json(run / "prediction_case_info.json", all_info)
    return {"rows": rows, "case_info": all_info, "prediction_tree_sha256": prediction_hash}
