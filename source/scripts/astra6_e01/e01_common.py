from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy.spatial import cKDTree

N_CLASSES = 52
N_VESSELS = 36
SEED = 20260905
THRESHOLD = 0.3
TOP_K = 5

P = Path(__file__).resolve().parents[2]
V = P.parent
DATA = V / "data_topaneu26"
BOX_DIR = P / "artifacts/fold1_eval_center2_epoch60"
TA36_DIR = P / "artifacts/ta36_mr_center2_output"
BASELINE_DIR = P / "artifacts/mr_epoch60_masks_predicted_ta36_sweep/k05"
COHORT_PATH = P / "artifacts/m1_center2_mr_case_ids.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_tree(path: Path) -> str:
    h = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file())
    for p in files:
        h.update(str(p.relative_to(path)).encode())
        h.update(sha256_file(p).encode())
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def case_id_from_image(path: Path) -> str:
    return path.name.removesuffix("_0000.nii.gz")


def modality_for_case(case_id: str) -> str:
    if "_mr_" in case_id:
        return "MR"
    if "_ct_" in case_id:
        return "CT"
    raise ValueError(f"cannot infer modality: {case_id}")


def center_for_case(case_id: str) -> int:
    m = re.search(r"center(\d+)", case_id)
    if not m:
        raise ValueError(case_id)
    return int(m.group(1))


def load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)
    return arr, np.asarray(img.affine, dtype=float), tuple(int(x) for x in img.shape)


def load_nifti_geometry(path: Path) -> tuple[np.ndarray, tuple[int, ...]]:
    img = nib.load(str(path))
    return np.asarray(img.affine, dtype=float), tuple(int(x) for x in img.shape)


def box_to_native_bounds(box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Frozen helper semantics: restored boxes are SITK z,y,x; native NIfTI is x,y,z.
    low = np.asarray([box[4], box[1], box[0]], dtype=float)
    high = np.asarray([box[5], box[3], box[2]], dtype=float)
    return low, high


def fill_ellipsoid(mask: np.ndarray, low: np.ndarray, high: np.ndarray, value: int) -> None:
    centre = (low + high) / 2.0
    radii = np.maximum((high - low) / 2.0, 0.5)
    lo_clip = np.maximum(np.floor(low).astype(int), 0)
    hi_clip = np.minimum(np.ceil(high).astype(int), np.asarray(mask.shape))
    if np.any(lo_clip >= hi_clip):
        return
    xs = np.arange(lo_clip[0], hi_clip[0])
    ys = np.arange(lo_clip[1], hi_clip[1])
    zs = np.arange(lo_clip[2], hi_clip[2])
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    normalized = (((xx - centre[0]) / radii[0]) ** 2
                   + ((yy - centre[1]) / radii[1]) ** 2
                   + ((zz - centre[2]) / radii[2]) ** 2)
    sub = mask[lo_clip[0]:hi_clip[0], lo_clip[1]:hi_clip[1], lo_clip[2]:hi_clip[2]]
    sub[normalized <= 1.0] = value
    mask[lo_clip[0]:hi_clip[0], lo_clip[1]:hi_clip[1], lo_clip[2]:hi_clip[2]] = sub


def component_boxes(mask: np.ndarray) -> list[dict[str, Any]]:
    # Work on the sparse foreground coordinate list.  Repeated full-volume
    # label/argwhere calls are needlessly expensive on the native grids.
    foreground = np.argwhere(mask > 0)
    if foreground.size == 0:
        return []
    foreground_labels = mask[tuple(foreground.T)]
    out: list[dict[str, Any]] = []
    for cls in range(1, N_CLASSES + 1):
        coords_cls = foreground[foreground_labels == cls]
        if not len(coords_cls):
            continue
        # Exact 26-connectivity with a sparse integer-key BFS.  A radius
        # query-pairs graph is tempting but can materialize millions of
        # edges for a dense component, while this visits each voxel once.
        sx, sy, sz = mask.shape
        stride_y, stride_z = sy * sz, sz
        keys = (coords_cls[:, 0].astype(np.int64) * stride_y
                + coords_cls[:, 1].astype(np.int64) * stride_z
                + coords_cls[:, 2].astype(np.int64))
        unvisited = set(int(k) for k in keys)
        groups: list[list[int]] = []
        offsets = [(dx, dy, dz) for dx in (-1, 0, 1)
                   for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                   if (dx, dy, dz) != (0, 0, 0)]
        while unvisited:
            start = unvisited.pop()
            queue = [start]
            group = [start]
            while queue:
                key = queue.pop()
                x, rem = divmod(key, stride_y)
                y, z = divmod(rem, stride_z)
                for dx, dy, dz in offsets:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < sx and 0 <= ny < sy and 0 <= nz < sz:
                        nk = int(nx * stride_y + ny * stride_z + nz)
                        if nk in unvisited:
                            unvisited.remove(nk)
                            queue.append(nk)
                            group.append(nk)
            groups.append(group)
        for component_id, group_keys in enumerate(groups, start=1):
            flat = np.asarray(group_keys, dtype=np.int64)
            coords = np.column_stack(np.unravel_index(flat, mask.shape)).astype(np.int64)
            lo = coords.min(axis=0).astype(float) - 0.5
            hi = coords.max(axis=0).astype(float) + 0.5
            out.append({"class_id": cls, "component_id": component_id,
                        "coords": coords, "low": lo, "high": hi,
                        "n_voxels": int(coords.shape[0])})
    return out


def label_mappings() -> tuple[dict[int, str], dict[int, str]]:
    loc = json.loads((DATA / "location_mapping.json").read_text())["labels"]
    ves = json.loads((DATA / "vessel_mapping.json").read_text())["labels"]
    return ({int(v): k for k, v in loc.items() if int(v) > 0},
            {int(v): k for k, v in ves.items() if int(v) > 0})


def lr_pair_map(names: dict[int, str]) -> dict[int, int]:
    by_name = {name: idx for idx, name in names.items()}
    pair: dict[int, int] = {}
    for idx, name in names.items():
        if name.startswith("R-"):
            pair[idx] = by_name.get("L-" + name[2:], idx)
        elif name.startswith("L-"):
            pair[idx] = by_name.get("R-" + name[2:], idx)
        else:
            pair[idx] = idx
    return pair


def affine_world(affine: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=float)
    return xyz @ affine[:3, :3].T + affine[:3, 3]


def native_box_world(affine: np.ndarray, low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    corners = np.array(np.meshgrid(*[(low[i], high[i]) for i in range(3)], indexing="ij")).reshape(3, -1).T
    world = affine_world(affine, corners)
    return affine_world(affine, (low + high) / 2.0), world.min(axis=0), world.max(axis=0)


def vessel_geometry(vessel_path: Path, image_shape: tuple[int, ...], image_affine: np.ndarray) -> dict[str, Any]:
    arr, aff, shape = load_nifti(vessel_path)
    if shape != image_shape or not np.allclose(aff, image_affine, atol=1e-4):
        raise ValueError(f"vessel/image geometry mismatch: {vessel_path}")
    union = arr > 0
    if not np.any(union):
        raise ValueError(f"empty vessel union: {vessel_path}")
    union_world = affine_world(aff, np.argwhere(union))
    p01, p99 = np.percentile(union_world, [1, 99], axis=0)
    gm = (p01 + p99) / 2.0
    gs = np.maximum(p99 - p01, 1.0)
    trees: dict[int, cKDTree | None] = {}
    segment_m: dict[int, np.ndarray] = {}
    segment_s: dict[int, np.ndarray] = {}
    for vessel_id in range(1, N_VESSELS + 1):
        coords = np.argwhere(arr == vessel_id)
        if coords.size:
            pts = affine_world(aff, coords)
            trees[vessel_id] = cKDTree(pts)
            p05, p95 = np.percentile(pts, [5, 95], axis=0)
            segment_m[vessel_id] = (p05 + p95) / 2.0
            segment_s[vessel_id] = np.maximum(p95 - p05, 1.0)
        else:
            trees[vessel_id] = None
            segment_m[vessel_id] = np.zeros(3, dtype=float)
            segment_s[vessel_id] = np.ones(3, dtype=float)
    return {"global_m": gm, "global_s": gs, "trees": trees,
            "segment_m": segment_m, "segment_s": segment_s}


def feature_schema(loc_names: dict[int, str], ves_names: dict[int, str]) -> list[str]:
    names = ["global_ras_x", "global_ras_y", "global_ras_z"]
    for v in range(1, N_VESSELS + 1):
        names += [f"v{v:02d}_presence", f"v{v:02d}_distance50"]
        names += [f"v{v:02d}_nearest_offset_{a}" for a in "xyz"]
        names += [f"v{v:02d}_segment_relative_{a}" for a in "xyz"]
    names += ["box_extent_x", "box_extent_y", "box_extent_z", "modality"]
    if len(names) != 295:
        raise AssertionError(len(names))
    return names


def compute_feature(geometry: dict[str, Any], affine: np.ndarray,
                    low: np.ndarray, high: np.ndarray, modality: str,
                    mirror: bool = False, vessel_pair: dict[int, int] | None = None) -> np.ndarray:
    q, box_lo, box_hi = native_box_world(affine, low, high)
    g = np.clip((q - geometry["global_m"]) / geometry["global_s"], -1.5, 1.5)
    ext = np.log1p(np.minimum(box_hi - box_lo, 100.0)) / math.log(101.0)
    x: list[float] = [float(v) for v in g]
    order = list(range(1, N_VESSELS + 1))
    if mirror:
        if vessel_pair is None:
            raise ValueError("vessel_pair required for mirrored features")
        order = [vessel_pair[v] for v in order]
    for vessel_id in order:
        tree = geometry["trees"][vessel_id]
        if tree is None:
            x.extend([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        distance, index = tree.query(q)
        nearest = np.asarray(tree.data[int(index)], dtype=float)
        offset = np.clip(nearest - q, -50.0, 50.0) / 50.0
        sm = geometry["segment_m"][vessel_id]
        ss = geometry["segment_s"][vessel_id]
        rel = np.clip((nearest - sm) / ss, -1.5, 1.5)
        x.extend([1.0, float(min(distance, 50.0) / 50.0), *map(float, offset), *map(float, rel)])
    if mirror:
        # Reflection is applied in the declared RAS coordinate feature space.
        x[0] *= -1.0
        for base in range(3, 3 + 36 * 8, 8):
            x[base + 2] *= -1.0  # nearest-offset x
            x[base + 5] *= -1.0  # segment-relative x
    x.extend(map(float, ext))
    x.append(0.0 if modality == "MR" else 1.0)
    arr = np.asarray(x, dtype=np.float32)
    if arr.shape != (295,) or not np.isfinite(arr).all():
        raise AssertionError(f"bad feature shape/finite: {arr.shape}")
    return arr


def load_boxes(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    boxes = np.asarray(obj["pred_boxes"], dtype=float)
    scores = np.asarray(obj["pred_scores"], dtype=float)
    if not obj.get("restore", False) or boxes.ndim != 2 or boxes.shape[1] != 6 or len(boxes) != len(scores):
        raise ValueError(f"bad box pickle {path}")
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError(f"nonfinite box pickle {path}")
    return boxes, scores, obj


def select_candidates(boxes: np.ndarray, scores: np.ndarray) -> list[tuple[int, float, np.ndarray, np.ndarray]]:
    selected: list[tuple[int, float, np.ndarray, np.ndarray]] = []
    for index in np.argsort(-scores):
        if len(selected) >= TOP_K or scores[index] < THRESHOLD:
            break
        low, high = box_to_native_bounds(boxes[index])
        selected.append((int(index), float(scores[index]), low, high))
    return selected


def sitk_array(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def nifti_output(mask_xyz: np.ndarray, reference: Path, path: Path) -> None:
    img = nib.load(str(reference))
    out = nib.Nifti1Image(mask_xyz.astype(np.uint8), img.affine, img.header.copy())
    out.set_data_dtype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(path))
