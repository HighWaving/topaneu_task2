from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .e01_common import (DATA, N_CLASSES, SEED, V, center_for_case, case_id_from_image,
                         component_boxes, compute_feature, feature_schema, load_nifti,
                         load_nifti_geometry,
                         lr_pair_map, modality_for_case, vessel_geometry, write_json)


def build_manifest(run: Path) -> dict[str, Any]:
    images = sorted(DATA.joinpath("images").glob("*_0000.nii.gz"))
    loc_dir = DATA / "location_masks"
    ves_dir = DATA / "vessel_masks"
    cases = []
    excluded = []
    for image in images:
        case_id = case_id_from_image(image)
        center = center_for_case(case_id)
        loc = loc_dir / f"{case_id}.nii.gz"
        vessel = ves_dir / f"{case_id}.nii.gz"
        if not loc.is_file() or not vessel.is_file():
            raise FileNotFoundError(f"missing current location/vessel for {case_id}")
        entry = {"case_id": case_id, "image": str(image), "location_mask": str(loc),
                 "vessel_mask": str(vessel), "center": center,
                 "modality": modality_for_case(case_id)}
        if center == 2 or case_id == "topaneu_center1_mr_150":
            excluded.append({"case_id": case_id, "reason": "center2" if center == 2 else "discarded_MR150"})
        elif center in (1, 4, 5):
            arr, aff, shape = load_nifti(loc)
            if arr.ndim != 3:
                raise ValueError(f"location not 3d: {case_id}")
            comps = component_boxes(arr)
            entry["n_components"] = len(comps)
            entry["positive"] = bool(comps)
            cases.append(entry)
        else:
            excluded.append({"case_id": case_id, "reason": f"center{center}_not_allowed"})
    train = [c for c in cases if c["positive"]]
    neg = [c for c in cases if not c["positive"]]
    manifest = {"spec": "ASTRA6-E01", "seed": SEED, "allowed_centers": [1, 4, 5],
                "cases": cases, "train_case_ids": [c["case_id"] for c in train],
                "negative_case_ids": [c["case_id"] for c in neg], "excluded": excluded,
                "counts": {"all_images": len(images), "allowed": len(cases),
                           "positive": len(train), "negative": len(neg),
                           "excluded": len(excluded)}}
    write_json(run / "source_manifest.json", manifest)
    write_json(run / "train_case_ids.json", manifest["train_case_ids"])
    return manifest


def jitter_boxes(comp: dict[str, Any], affine: np.ndarray, shape: tuple[int, ...], rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
    low = comp["low"].copy(); high = comp["high"].copy()
    base = [(low.copy(), high.copy())]
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    physical_edge = (high - low) * spacing
    coords = comp["coords"]
    for _ in range(8):
        chosen = None
        for _attempt in range(32):
            centre = (low + high) / 2.0
            displacement_mm = rng.uniform(-1.0, 1.0, 3) * np.minimum(0.2 * physical_edge, 2.0)
            centre_j = centre + displacement_mm / np.maximum(spacing, 1e-9)
            size = (high - low) * rng.uniform(0.8, 1.2, 3)
            lo = np.maximum(centre_j - size / 2.0, 0.0)
            hi = np.minimum(centre_j + size / 2.0, np.asarray(shape, dtype=float))
            ilo = np.maximum(np.floor(lo).astype(int), 0)
            ihi = np.minimum(np.ceil(hi).astype(int), np.asarray(shape))
            if np.any(ilo >= ihi):
                continue
            if np.any(np.all((coords >= ilo) & (coords < ihi), axis=1)):
                chosen = (lo, hi); break
        base.append(chosen if chosen is not None else (low.copy(), high.copy()))
    return base


def build_features(run: Path, manifest: dict[str, Any], loc_names: dict[int, str], ves_names: dict[int, str]) -> dict[str, Any]:
    loc_pair = lr_pair_map(loc_names)
    ves_pair = lr_pair_map(ves_names)
    rng = np.random.default_rng(SEED)
    schema = feature_schema(loc_names, ves_names)
    records: list[dict[str, Any]] = []
    rows: list[np.ndarray] = []
    labels: list[int] = []
    views: list[str] = []
    for case_i, case in enumerate(manifest["cases"]):
        if not case["positive"]:
            continue
        image_aff, image_shape = load_nifti_geometry(Path(case["image"]))
        loc_arr, loc_aff, loc_shape = load_nifti(Path(case["location_mask"]))
        if image_shape != loc_shape or not np.allclose(image_aff, loc_aff, atol=1e-4):
            raise ValueError(f"image/location geometry mismatch {case['case_id']}")
        geometry = vessel_geometry(Path(case["vessel_mask"]), image_shape, image_aff)
        for comp in component_boxes(loc_arr):
            jitter = jitter_boxes(comp, image_aff, image_shape, rng)
            for mirror in (False, True):
                label_id = loc_pair[comp["class_id"]] if mirror else comp["class_id"]
                view = "mirror" if mirror else "original"
                for sample_i, (low, high) in enumerate(jitter):
                    rows.append(compute_feature(geometry, image_aff, low, high, case["modality"], mirror, ves_pair))
                    labels.append(label_id); views.append(view)
                    records.append({"case_id": case["case_id"], "component_id": comp["component_id"],
                                    "class_id": label_id, "source_class_id": comp["class_id"],
                                    "view": view, "sample_index": sample_i,
                                    "low": low.tolist(), "high": high.tolist()})
        if (case_i + 1) % 25 == 0:
            print(f"feature progress {case_i + 1}/{len(manifest['cases'])}", flush=True)
    x = np.asarray(rows, dtype=np.float32); y = np.asarray(labels, dtype=np.int16)
    if x.ndim != 2 or x.shape[1] != 295 or not np.isfinite(x).all():
        raise AssertionError(f"training feature invariant {x.shape}")
    counts = {str(i): int(np.sum(y == i)) for i in range(1, N_CLASSES + 1)}
    base_counts = {str(i): int(sum(1 for r in records if r["class_id"] == i)) for i in range(1, N_CLASSES + 1)}
    sample_weights = np.asarray([1.0 / (9.0 * np.sqrt(base_counts[str(int(label))])) for label in y], dtype=np.float64)
    sample_weights /= sample_weights.mean()
    feature_dir = run / "features"; feature_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(feature_dir / "train.npz", X=x, y=y, sample_weight=sample_weights)
    (feature_dir / "train_records.jsonl").write_text("\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n")
    write_json(run / "feature_schema.json", {"n_features": 295, "features": schema,
                                               "label_ids": list(range(1, N_CLASSES + 1)),
                                               "location_names": loc_names, "vessel_names": ves_names,
                                               "location_lr_pair": loc_pair, "vessel_lr_pair": ves_pair,
                                               "augmentation": {"base_plus_jitter": 9, "mirror_views": 2, "seed": SEED}})
    write_json(run / "train_support.json", {"row_counts": counts, "view_component_counts": base_counts,
                                             "n_rows": int(len(y)), "n_components": int(len(y) // 18)})
    return {"X": x, "y": y, "sample_weight": sample_weights, "records": records,
            "loc_pair": loc_pair, "ves_pair": ves_pair, "schema": schema,
            "train_support": base_counts}
