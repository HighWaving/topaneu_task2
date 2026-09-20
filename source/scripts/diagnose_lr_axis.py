"""One-shot diagnostic (2026-08-15, peer-requested): which normalized-centroid
axis actually encodes L/R, determined empirically from the 24 L-/R- paired
class names, not assumed from SimpleITK/nibabel convention. Also checks
whether per-case shape normalization is even sane across centers (large
FOV variance would break it independent of axis order).

Not a general tool -- single use, prints results and exits.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

DATA_ROOT = Path("/home/jovyan/rtx4claude-datavol-1/data_topaneu26")
LOCATION_MASKS = DATA_ROOT / "location_masks"
CASE_RE = re.compile(r"^topaneu_(center\d)_(mr|ct)_")


def load_labels(mapping_path: Path) -> dict[int, str]:
    labels = json.loads(mapping_path.read_text())["labels"]
    return {v: k for k, v in labels.items()}


def main() -> int:
    location_names = load_labels(DATA_ROOT / "location_mapping.json")
    files = sorted(f for f in LOCATION_MASKS.iterdir() if f.suffix == ".gz")

    l_centroids, r_centroids, midline_centroids = [], [], []
    shapes_by_center = defaultdict(list)

    for i, loc_file in enumerate(files):
        if i % 40 == 0:
            print(f"progress: {i}/{len(files)}", flush=True)
        m = CASE_RE.match(loc_file.name)
        if not m:
            continue
        center = m.group(1)
        img = sitk.ReadImage(str(loc_file))
        arr = sitk.GetArrayFromImage(img)
        shapes_by_center[center].append(arr.shape)
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
                norm = centroid / shape
                if cls_name.startswith("L-"):
                    l_centroids.append(norm)
                elif cls_name.startswith("R-"):
                    r_centroids.append(norm)
                else:
                    midline_centroids.append(norm)

    l_arr = np.asarray(l_centroids)
    r_arr = np.asarray(r_centroids)
    mid_arr = np.asarray(midline_centroids)

    print(f"\nn L-lesions: {len(l_arr)}  n R-lesions: {len(r_arr)}  n midline: {len(mid_arr)}")
    print("\nmean(L) - mean(R) per axis (SimpleITK array axes: 0=z(S/I) 1=y(A/P) 2=x(L/R) nominal):")
    diff = l_arr.mean(axis=0) - r_arr.mean(axis=0)
    for ax in range(3):
        print(f"  axis {ax}: mean(L)={l_arr[:,ax].mean():.4f} mean(R)={r_arr[:,ax].mean():.4f} "
              f"diff={diff[ax]:+.4f}")
    lr_axis = int(np.argmax(np.abs(diff)))
    print(f"\n  => largest |diff| is axis {lr_axis} (diff={diff[lr_axis]:+.4f}) "
          f"-- this is the true L/R axis")
    sign_note = "L has LARGER coord" if diff[lr_axis] > 0 else "L has SMALLER coord"
    print(f"  => {sign_note} on axis {lr_axis}")

    print("\nmidline classes (should cluster near 0.5 on the L/R axis):")
    if len(mid_arr):
        print(f"  axis {lr_axis} midline mean={mid_arr[:,lr_axis].mean():.4f} "
              f"std={mid_arr[:,lr_axis].std():.4f}  (n={len(mid_arr)})")
        for ax in range(3):
            print(f"  axis {ax}: mean={mid_arr[:,ax].mean():.4f} std={mid_arr[:,ax].std():.4f}")
    else:
        print("  (none found)")

    print("\nper-center array shape variance (large spread => FOV normalization unreliable):")
    for center, shapes in sorted(shapes_by_center.items()):
        arr = np.asarray(shapes)
        print(f"  {center}: n={len(arr)} mean={arr.mean(axis=0)} std={arr.std(axis=0)} "
              f"min={arr.min(axis=0)} max={arr.max(axis=0)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
