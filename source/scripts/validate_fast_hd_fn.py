"""Validates fast_hd_fn against the OFFICIAL, unmodified hd_fn byte-for-byte.

2026-08-14. Required to pass before local_scoring_arena.USE_FAST_HD95 is
flipped to True. Never trust the crop optimization without this -- the
scoring harness is the ground truth for every downstream number this
session has produced; a silent mismatch here would be worse than the slow
path, not just slower.

Cases covered: both empty, only one empty (both directions), a small
synthetic lesion-like blob with partial overlap between the two masks, and
a blob touching the volume's edge (tests the crop's boundary clamping
against get_surface's own border_value=0 erosion behavior).
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/scripts")
from local_scoring_arena import fast_hd_fn, _load_evaluate_module  # noqa: E402


def make_blob(shape, center, radius) -> np.ndarray:
    arr = np.zeros(shape, dtype=bool)
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    c = np.asarray(center)
    r2 = ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2)
    arr[r2 <= radius ** 2] = True
    return arr


def main() -> int:
    module = _load_evaluate_module()  # USE_FAST_HD95 is still False at this point
    original_hd_fn = module.hd_fn
    shape = (60, 80, 70)

    empty = np.zeros(shape, dtype=bool)
    blob_a = make_blob(shape, (30, 40, 35), 5)
    blob_b_overlap = make_blob(shape, (32, 41, 36), 5)  # shifted, partial overlap
    blob_b_far = make_blob(shape, (10, 10, 10), 3)      # disjoint, no overlap
    blob_edge = make_blob(shape, (2, 2, 2), 4)           # touches volume edge
    blob_edge2 = make_blob(shape, (3, 3, 3), 3)

    cases = [
        ("both_empty", empty, empty),
        ("only_a_empty", empty, blob_a),
        ("only_b_empty", blob_a, empty),
        ("partial_overlap", blob_a, blob_b_overlap),
        ("disjoint_nonempty", blob_a, blob_b_far),
        ("identical", blob_a, blob_a),
        ("edge_touching", blob_edge, blob_edge2),
    ]

    max_abs_diff = 0.0
    all_pass = True
    for name, img1, img2 in cases:
        expected = original_hd_fn(img1, img2, 95, True)
        actual = fast_hd_fn(module, img1, img2, 95, True)
        diff = abs(expected - actual)
        max_abs_diff = max(max_abs_diff, diff)
        ok = diff < 1e-9
        all_pass &= ok
        print(f"{name:<20} expected={expected:.12f} actual={actual:.12f} diff={diff:.2e} "
             f"{'PASS' if ok else 'FAIL'}", flush=True)

    print(f"\nmax abs diff across all cases: {max_abs_diff:.2e}")
    if all_pass:
        print("ALL CASES PASS -- fast_hd_fn is byte-for-byte equivalent to the official hd_fn.")
    else:
        print("FAILURE -- fast_hd_fn diverges from the official hd_fn. DO NOT enable USE_FAST_HD95.")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
