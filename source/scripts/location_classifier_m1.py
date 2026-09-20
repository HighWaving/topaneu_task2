"""M1 production 52-class position classifier (2026-08-15, peer-directed fix).

Bug fixed: `probe_centroid_only_baseline.py` (Experiment B) built its k-NN
pool via `sitk.GetArrayFromImage`, which returns arrays in (z,y,x) order
(confirmed: for topaneu_center2_mr_002, nib.load(...).shape = (490,583,200)
= (x,y,z) but sitk.GetArrayFromImage(...).shape = (200,583,490) = (z,y,x) --
an exact axis reversal, not a guess). `build_e2e_placeholder_pipeline.py`'s
box-to-mask conversion works entirely in nibabel (x,y,z) voxel order. Wiring
Experiment B's sitk-trained classifier directly to nibabel-order box
centroids would silently swap the z and x axes for every query -- exactly
the mechanism the peer flagged as the likely cause of Experiment B's
anomalous 32.8% territory-error / 0.8% lr_mirror rates.

Fix: this module rebuilds the training pool with nibabel (matching the mask
builder's own convention) instead of sitk, so there is exactly one
coordinate convention in the M1 pipeline, not two. Nothing else about
Experiment B's method changes (still per-case-shape-normalized centroid
k-NN, still k=1, the empirically-best k from Experiment B's sweep).

Scope discipline (explicit, peer-directed): this fixes the coordinate bug
only. It does not attempt to improve accuracy, add registration, or chase
the territory/lr_mirror anomaly further -- if the self-check below still
shows something odd, that gets recorded, not chased, and M1 proceeds anyway.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

DATA_ROOT = Path("/home/jovyan/rtx4claude-datavol-1/data_topaneu26")
LOCATION_MASKS = DATA_ROOT / "location_masks"
CASE_RE = re.compile(r"^topaneu_(center\d)_(mr|ct)_")
K_DEFAULT = 1  # empirically best-coverage k from Experiment B's sweep over {1,3,5,7,9,15,21}


def load_labels(mapping_path: Path) -> dict[int, str]:
    labels = json.loads(mapping_path.read_text())["labels"]
    return {v: k for k, v in labels.items()}


def mirror_name(name: str) -> str | None:
    if name.startswith("L-"):
        return "R-" + name[2:]
    if name.startswith("R-"):
        return "L-" + name[2:]
    return None


class LocationClassifier:
    """k-NN, nibabel (x,y,z) voxel-centroid-normalized-by-own-shape, LOCO."""

    def __init__(self, exclude_centers: set[str], k: int = K_DEFAULT):
        self.k = k
        self.class_id_by_name: dict[str, int] = {}
        location_names = load_labels(DATA_ROOT / "location_mapping.json")
        self.name_by_class_id = location_names
        self.class_id_by_name = {v: k_ for k_, v in location_names.items()}

        pts, labels = [], []
        files = sorted(f for f in LOCATION_MASKS.iterdir() if f.suffix == ".gz")
        for f in files:
            m = CASE_RE.match(f.name)
            if not m or m.group(1) in exclude_centers:
                continue
            arr = nib.load(str(f)).get_fdata()
            if not arr.any():
                continue
            shape = np.asarray(arr.shape, dtype=float)
            for cls in np.unique(arr):
                if cls == 0:
                    continue
                cls_name = location_names.get(int(cls), f"class_{cls}")
                labeled, n_components = ndimage.label(
                    arr == cls, structure=np.ones((3, 3, 3), dtype=np.uint8))
                for component_id in range(1, n_components + 1):
                    centroid = np.asarray(ndimage.center_of_mass(labeled == component_id))
                    pts.append(centroid / shape)
                    labels.append(cls_name)

        self.train_pts = np.asarray(pts)
        self.train_labels = labels
        self._self_check()

    def _self_check(self) -> None:
        """Peer-specified diagnostic (2026-08-15): empirically determine which
        axis is L/R from the 24 self-labeled L-/R- class pairs, rather than
        assuming nibabel/sitk convention. Prints and stores the result; does
        NOT gate M1 on the outcome (recorded, not chased, per explicit scope
        limit)."""
        l_pts = np.asarray([p for p, lbl in zip(self.train_pts, self.train_labels)
                            if lbl.startswith("L-")])
        r_pts = np.asarray([p for p, lbl in zip(self.train_pts, self.train_labels)
                            if lbl.startswith("R-")])
        mid_pts = np.asarray([p for p, lbl in zip(self.train_pts, self.train_labels)
                              if not lbl.startswith(("L-", "R-"))])
        diff = l_pts.mean(axis=0) - r_pts.mean(axis=0)
        lr_axis = int(np.argmax(np.abs(diff)))
        self.self_check = {
            "n_left": len(l_pts), "n_right": len(r_pts), "n_midline": len(mid_pts),
            "mean_L_minus_R_per_axis": diff.tolist(),
            "inferred_lr_axis": lr_axis,
            "inferred_lr_axis_diff": float(diff[lr_axis]),
            "midline_mean_on_lr_axis": float(mid_pts[:, lr_axis].mean()) if len(mid_pts) else None,
            "midline_std_on_lr_axis": float(mid_pts[:, lr_axis].std()) if len(mid_pts) else None,
        }
        print(f"[location_classifier_m1] self-check: L-R diff per axis (nibabel x,y,z order) = "
              f"{[round(x, 4) for x in diff.tolist()]}", flush=True)
        print(f"[location_classifier_m1] inferred L/R axis = {lr_axis} "
              f"(diff={diff[lr_axis]:+.4f}), midline classes on that axis: "
              f"mean={self.self_check['midline_mean_on_lr_axis']:.4f} "
              f"std={self.self_check['midline_std_on_lr_axis']:.4f}", flush=True)

    def classify(self, centroid_xyz_voxel: np.ndarray, case_shape_xyz: np.ndarray) -> int:
        """centroid_xyz_voxel, case_shape_xyz: both nibabel (x,y,z) voxel order."""
        query = np.asarray(centroid_xyz_voxel) / np.asarray(case_shape_xyz, dtype=float)
        dists = np.linalg.norm(self.train_pts - query, axis=1)
        order = np.argsort(dists)[:self.k]
        votes = Counter(self.train_labels[i] for i in order)
        top_count = max(votes.values())
        tied = [lbl for lbl, c in votes.items() if c == top_count]
        if len(tied) == 1:
            pred_name = tied[0]
        else:
            pred_name = next(self.train_labels[i] for i in order if self.train_labels[i] in tied)
        return self.class_id_by_name[pred_name]


if __name__ == "__main__":
    clf = LocationClassifier(exclude_centers={"center2"})
    print(json.dumps(clf.self_check, indent=2))
