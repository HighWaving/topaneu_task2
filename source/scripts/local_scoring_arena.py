"""Local wrapper around the OFFICIAL TopAneu-26 Task 2 evaluate.py -- not a reimplementation.

2026-08-14, user-directed strategy pivot. `TopAneu-26/eval/task2/evaluate.py`
is the exact scoring code grand-challenge will run. Reimplementing the
metrics locally would risk a subtle mismatch (rounding, edge cases, the
TN-avoids-DSC-bias convention) that only shows up on the real leaderboard --
too late to fix. Instead this imports `evaluate.py`'s functions directly and
only replaces the one thing that's container-specific:
`load_gt()`'s hardcoded `/opt/ml/input/data/ground_truth/location_masks/`
path, monkey-patched to point at this session's own local copy of the same
files (`data_topaneu26/location_masks/`).

Validated (see `validate_local_scoring_arena.py`) against per-class
behaviour on 5 real positive cases: all-zero predictions correctly score
HD95=1.0 on every class present in that case's GT; predictions identical
to GT correctly score precision=recall=dice=volsim=1.0, hd95=0.0 on every
present class. (The *overall*, 52-class-averaged metrics are NOT checked
directly against the official `test_evaluations/outputs-*.json` fixture --
those were generated from a broad synthetic set touching most/all 52
classes, and `evaluation_average()` divides by N_CLASSES=52 unweighted, so
the overall average on a small real sample is legitimately diluted by
class coverage, not comparable to the fixture. Per-class checks are the
correct invariant regardless of sample size.)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

EVAL_DIR = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task1/TopAneu-26/eval/task2")
DATA_ROOT = Path("/home/jovyan/rtx4claude-datavol-1/data_topaneu26")

# 2026-08-14, peer-diagnosed perf fix: the official get_surface() runs
# binary_erosion on the FULL volume (~43M voxels) per class per case (52 x
# 68 = 3536 calls for one center's worth of scoring), which is the entire
# cost of a scoring run (~50min). Masks here are tiny (one small lesion, one
# small ellipsoid) against a huge mostly-zero volume, so this crops to the
# union bounding box (+2 voxel margin, enough for 3x3x3-connectivity erosion
# to see the same neighborhood it would in the full volume) before calling
# the OFFICIAL, UNMODIFIED get_surface() on the cropped arrays. `diag` is
# still computed from the ORIGINAL (uncropped) shape, since it normalizes
# hd_fn's output and cropping must not change that scale.
# NEVER edit evaluate.py itself -- see module docstring. This is validated
# byte-for-byte against the unmodified hd_fn in
# validate_fast_hd_fn.py before use; do not enable without rerunning that.
USE_FAST_HD95 = True  # validate_fast_hd_fn.py passed 7/7 cases at exactly 0.00e+00 diff, 2026-08-14


def fast_hd_fn(module, img1: np.ndarray, img2: np.ndarray, perc: int = 95, norm: bool = True) -> float:
    """Same math as evaluate.py's hd_fn, cropped to the union bbox for speed.

    Calls the OFFICIAL get_surface() (not reimplemented) on cropped arrays;
    only the erosion's input size shrinks, not its semantics.
    """
    diag = np.linalg.norm(img1.shape)  # from the ORIGINAL shape, not the crop
    if not np.any(img1) and not np.any(img2):
        return 0
    elif not np.any(img1) or not np.any(img2):
        return 1 if norm else diag

    nz = np.argwhere(img1 | img2)
    lo = np.maximum(nz.min(axis=0) - 2, 0)
    hi = np.minimum(nz.max(axis=0) + 3, np.asarray(img1.shape))
    sl = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
    sub1, sub2 = img1[sl], img2[sl]

    from scipy.spatial import cKDTree

    coords_a = np.argwhere(module.get_surface(sub1))
    coords_b = np.argwhere(module.get_surface(sub2))
    tree_b = cKDTree(coords_b)
    tree_a = cKDTree(coords_a)
    d_ab, _ = tree_b.query(coords_a)
    d_ba, _ = tree_a.query(coords_b)
    hd = max(np.percentile(d_ab, perc), np.percentile(d_ba, perc))
    return hd / diag if norm else hd


def _load_evaluate_module():
    """Imports evaluate.py directly from disk, with load_gt patched to a local path.

    evaluate.py does ``import numpy as np`` etc. at module level with no
    package structure, so this loads it as a standalone module rather than
    fighting sys.path against the TopAneu-26 checkout's own layout.
    """
    spec = importlib.util.spec_from_file_location("topaneu_evaluate", EVAL_DIR / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(EVAL_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)

    import SimpleITK as sitk

    def local_load_gt(fn: str) -> np.ndarray:
        # Same filename convention as the original: strip "_0000" suffix,
        # but point at our own local .nii.gz copy instead of a mounted .mha.
        case_id = fn.replace("_0000.mha", "").replace("_0000.nii.gz", "")
        path = DATA_ROOT / "location_masks" / f"{case_id}.nii.gz"
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))

    module.load_gt = local_load_gt

    if USE_FAST_HD95:
        original_hd_fn = module.hd_fn

        def patched_hd_fn(img1, img2, perc=95, norm=True):
            return fast_hd_fn(module, img1, img2, perc, norm)

        module.hd_fn = patched_hd_fn
        module._original_hd_fn = original_hd_fn  # kept for validation scripts

    return module


_EVALUATE = None


def evaluate_module():
    global _EVALUATE
    if _EVALUATE is None:
        _EVALUATE = _load_evaluate_module()
    return _EVALUATE


def score_case(predictions: np.ndarray, case_id: str) -> dict:
    """Score one case's prediction array against its ground truth.

    ``case_id`` is the bare case id (e.g. "topaneu_center1_mr_001"), NOT the
    "_0000"-suffixed input filename evaluate.py's own load_gt() strips --
    this wrapper's local_load_gt() accepts either form.
    """
    module = evaluate_module()
    return module.evaluation_function(predictions, case_id)


def aggregate(results: list[dict]) -> dict:
    module = evaluate_module()
    per_class = module.evaluation_aggregation(results)
    overall = module.evaluation_average(per_class)
    return {"per_class": per_class, "overall": overall}
