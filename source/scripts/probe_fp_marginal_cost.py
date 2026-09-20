"""Decisive experiment: what is the marginal cost of a false-positive class prediction?

2026-08-14, peer-requested. Peer's hypothesis: since TN classes score
DICE=0 (not the "correct" 1) by evaluate.py's own explicit design "to
avoid biasing the evaluation", maybe over-predicting (extra FP classes)
costs ~nothing, which would argue for a very permissive operating point
(sensitivity-first, don't worry about FP).

Before running anything, read evaluation_function/evaluation_aggregation
in full (evaluate.py) to derive the mechanism directly, then verify
empirically. Two distinct scenarios must be told apart -- collapsing them
would give a wrong answer to the peer's actual question:

  (i)  spurious class that is NEVER true anywhere in the sample (e.g. the
       placeholder pipeline picks a class that happens to have zero real
       GT instances in this fold)
  (ii) spurious class that IS sometimes true elsewhere in the sample (e.g.
       the placeholder pipeline always emits the single most common class
       49 -- which genuinely has real GT instances, so wrong assignments
       collide with real TP/FN for that same class)

Code-derived prediction (to check against the run):
  (i)  PRECISION/RECALL/DICE/VOLSIM/MCC for that class stay ~0 in BOTH the
       correct and the over-predicted case (numerator floored at 0
       regardless of denominator), EXCEPT HD95, which jumps from ~0
       (never triggered, TP+FN+FP=0 -> aggregate 0/eps) to ~1.0 (every
       spurious case scores hd_fn's one-side-empty penalty =1, divided by
       a denominator now equal to the number of spurious cases). So for a
       genuinely-never-true class, only HD95 for that one class is hit --
       not "free" but narrowly confined.
  (ii) For a class with real TP/FN elsewhere, extra FP dilutes an
       already-nonzero DICE/VOLSIM/PRECISION/MCC numerator ratio downward
       for real, and pushes HD95 upward for real. NOT free. RECALL for
       that class is the only metric genuinely unaffected by FP (it only
       depends on TP/FN).
  Missing a real class (FN) always costs on that class's own DICE->0,
  HD95->1, VOLSIM->0, RECALL down -- there is no free-FN case analogous
  to (i), since FN can only happen for classes that ARE truly present.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent))
from local_scoring_arena import aggregate, score_case  # noqa: E402

MANIFEST = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task1/artifacts/phase0/case_manifest.json")
OUT_DIR = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/reports")

NEVER_TRUE_CLASS = 1     # chosen below to genuinely never appear in the sample
COMMON_REAL_CLASS = None  # chosen below: the class that appears most often in the sample


def build_scenario(gt: np.ndarray, mode: str, extra_class: int, n_extra: int, drop_one: bool) -> np.ndarray:
    pred = gt.copy()
    if drop_one:
        present = [c for c in np.unique(gt) if c != 0]
        if present:
            pred[pred == present[0]] = 0
    if mode == "spurious":
        # Adds n_extra voxels of extra_class at fixed background corner
        # locations far from any real lesion, so it never overlaps GT.
        for i in range(n_extra):
            z, y, x = 2 + i, 2, 2
            if z < pred.shape[0]:
                pred[z, y, x] = extra_class + i  # each extra voxel a DIFFERENT spurious class
    return pred


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    sample = []
    for case in manifest["cases"]:
        gt = sitk.GetArrayFromImage(sitk.ReadImage(case["location_mask"]))
        if gt.any():
            sample.append((case["case_id"], gt))
        if len(sample) >= 6:
            break
    print(f"sample: {len(sample)} cases", flush=True)

    from collections import Counter
    class_counts = Counter()
    for _, gt in sample:
        for c in np.unique(gt):
            if c != 0:
                class_counts[int(c)] += 1
    print("class occurrence (n cases containing class):", dict(class_counts))
    common_class = class_counts.most_common(1)[0][0]
    never_true = next(c for c in range(1, 53) if c not in class_counts)
    print(f"common_class (real GT elsewhere) = {common_class}, never_true class = {never_true}")

    def score(preds_by_case: list[tuple[str, np.ndarray]]) -> dict:
        results = [score_case(pred, case_id) for case_id, pred in preds_by_case]
        return aggregate(results)

    # (a) perfect
    scen_a = [(cid, gt.copy()) for cid, gt in sample]
    agg_a = score(scen_a)

    # (i-b/c) perfect + 1 / +5 spurious NEVER-true classes per case
    def add_spurious(gt, cls_list):
        pred = gt.copy()
        for i, cls in enumerate(cls_list):
            pred[1 + i, 1, 1] = cls  # fixed background voxel, never overlaps real lesions
        return pred

    scen_i_b = [(cid, add_spurious(gt, [never_true])) for cid, gt in sample]
    scen_i_c = [(cid, add_spurious(gt, list(range(never_true, never_true + 5)))) for cid, gt in sample]
    agg_i_b = score(scen_i_b)
    agg_i_c = score(scen_i_c)

    # (ii) perfect but replace one real class's voxels with common_class
    # everywhere it ISN'T already common_class -- simulates "placeholder
    # always predicts class 49" when the true label is something else.
    def misassign_to_common(gt, common):
        pred = gt.copy()
        present = [c for c in np.unique(gt) if c != 0]
        for c in present:
            if c != common:
                pred[pred == c] = common
        return pred

    scen_ii = [(cid, misassign_to_common(gt, common_class)) for cid, gt in sample]
    agg_ii = score(scen_ii)

    # (d) perfect minus one real class (an FN)
    def drop_one_class(gt):
        pred = gt.copy()
        present = [c for c in np.unique(gt) if c != 0]
        if present:
            pred[pred == present[0]] = 0
        return pred

    scen_d = [(cid, drop_one_class(gt)) for cid, gt in sample]
    agg_d = score(scen_d)

    def per_class_row(agg, cls):
        pc = agg["per_class"]
        c = int(cls)
        return {k: round(float(pc[f"{k}_{c}"]), 4) for k in ("PRECISION", "RECALL", "DICE", "HD95", "VOLSIM", "MCC")}

    report = {
        "sample_n": len(sample),
        "class_occurrence": dict(class_counts),
        "common_class": common_class,
        "never_true_class": never_true,
        "scenario_a_perfect__overall": agg_a["overall"],
        "scenario_i_b_plus1_never_true__overall": agg_i_b["overall"],
        "scenario_i_c_plus5_never_true__overall": agg_i_c["overall"],
        "scenario_ii_misassign_to_common__overall": agg_ii["overall"],
        "scenario_d_drop_one_real_class__overall": agg_d["overall"],
        "never_true_class_per_class__a_vs_b_vs_c": {
            "a": per_class_row(agg_a, never_true),
            "b_plus1": per_class_row(agg_i_b, never_true),
            "c_plus5": per_class_row(agg_i_c, never_true),
        },
        "common_class_per_class__a_vs_ii_misassign": {
            "a": per_class_row(agg_a, common_class),
            "ii_misassign": per_class_row(agg_ii, common_class),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "fp_marginal_cost_probe.json"
    out_path.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(json.dumps(report, indent=2, default=float))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
