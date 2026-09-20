"""Decisive experiment: does speculative over-prediction of a real (non-globally-zero) class have positive expected value?

2026-08-14, peer hypothesis, checked against the real evaluate.py before
trusting the algebra. Peer's derivation: aggregated DICE_c (and similarly
HD95/VOLSIM/PRECISION/RECALL) = sum(per-case value) / (TP_c+FN_c+FP_c+eps).
For a class c that truly exists somewhere in the held-out set (not one of
the 9 globally-zero classes), a speculative extra prediction of c:
  - if it MISSES every true instance of c (unlucky): numerator unchanged
    (adds 0 either way), denominator grows -- these 5 metrics stay flat at
    the "don't predict at all" baseline, no worse.
  - if it happens to overlap a true instance (lucky): numerator gains a
    positive term -- strict improvement over baseline.
So broad/speculative guessing may have positive expected value even at low
precision, for classes known (or suspected) to exist somewhere.

I rechecked this by hand for DICE/HD95/PRECISION/RECALL/VOLSIM and it
holds. But I flagged a concern the peer didn't raise: MCC's per-case
TN_i = n_aneu - (tp+fn), where n_aneu is THAT CASE's distinct GT class
count (not a fixed pool) -- my hand derivation suggested MCC might get
WORSE (more negative) from added FP even with zero lucky hits, unlike the
other 5 metrics. Rather than assert an algebra-derived claim I'm not fully
confident in, this is checked directly against the real evaluate.py.

Uses a real class from a real LOCO fold (center4, class 4, which occurs in
exactly 1 of center4's 62 held-out cases -- an "=1 instance" class, the
exact bucket the peer's derivation is about). Two localization scenarios:
  - "lucky": one guess is placed as a JITTERED (not exact) copy of the true
    lesion mask in the one true case, simulating imperfect but overlapping
    localization -- plus k-1 further guesses in OTHER cases where class 4
    is absent (spurious).
  - "unlucky": all k guesses are spurious (placed in cases where class 4 is
    genuinely absent), zero overlap with the one true instance.
k in {1, 3, 6, 10}. Respects the single-channel uint8 constraint the peer
flagged: every guess occupies distinct voxels, never overwriting a case's
real (perfect, for every OTHER class) prediction -- this experiment only
perturbs class 4's own channel; every other class in every case keeps its
ground-truth-perfect prediction throughout, isolating class 4's own
aggregate metrics from confounds elsewhere.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from local_scoring_arena import aggregate, score_case  # noqa: E402

MANIFEST = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task1/artifacts/phase0/case_manifest.json")
LOCO_SPLITS = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/configs/loco_splits.json")
OUT_DIR = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/reports")

HELD_OUT_CENTER = "center4"
TARGET_CLASS = 4
K_VALUES = (1, 3, 6, 10)


def jitter_mask(mask: np.ndarray, shift: tuple[int, int, int]) -> np.ndarray:
    """A deliberately imperfect copy of the true lesion -- shifted a few
    voxels so it overlaps but is not identical, simulating realistic
    (not oracle) localization for the 'lucky hit' case.
    """
    shifted = np.zeros_like(mask)
    coords = np.argwhere(mask)
    for z, y, x in coords:
        nz, ny, nx = z + shift[0], y + shift[1], x + shift[2]
        if 0 <= nz < mask.shape[0] and 0 <= ny < mask.shape[1] and 0 <= nx < mask.shape[2]:
            shifted[nz, ny, nx] = 1
    return shifted.astype(bool)


def place_spurious_stub(pred: np.ndarray, case_id: str, index: int) -> np.ndarray:
    """A small 3x3x3 stub of TARGET_CLASS at a case-varying but fixed
    background location, far from real lesion locations -- respects the
    single-channel constraint: only touches voxels currently 0 (background),
    never overwrites another class's real prediction.
    """
    z0 = 5 + (hash(case_id) % 20)
    y0, x0 = 5, 5
    for dz in range(3):
        for dy in range(3):
            for dx in range(3):
                zz, yy, xx = z0 + dz, y0 + dy, x0 + dx
                if zz < pred.shape[0] and yy < pred.shape[1] and xx < pred.shape[2] and pred[zz, yy, xx] == 0:
                    pred[zz, yy, xx] = TARGET_CLASS
    return pred


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    cases_by_id = {c["case_id"]: c for c in manifest["cases"]}
    loco = json.loads(LOCO_SPLITS.read_text())
    val_ids = loco["folds"][HELD_OUT_CENTER]["val"]

    true_case_id = None
    for case_id in val_ids:
        if TARGET_CLASS in cases_by_id[case_id]["locations"]:
            true_case_id = case_id
            break
    assert true_case_id is not None, f"class {TARGET_CLASS} not found in {HELD_OUT_CENTER}"
    print(f"target class {TARGET_CLASS}, true instance in case {true_case_id}", flush=True)

    # A manageable sub-sample: the true case + enough absent-class cases to
    # supply spurious placements for the largest k. This keeps runtime sane
    # (HD95's Hausdorff cost is the bottleneck, ~5-20s/case) while still
    # being real evaluate.py end to end.
    absent_cases = [c for c in val_ids if TARGET_CLASS not in cases_by_id[c]["locations"]][:max(K_VALUES)]
    sample_ids = [true_case_id] + absent_cases
    print(f"sample: {len(sample_ids)} cases ({true_case_id} + {len(absent_cases)} class-{TARGET_CLASS}-absent)",
          flush=True)

    gts = {}
    for case_id in sample_ids:
        gt = sitk.GetArrayFromImage(sitk.ReadImage(cases_by_id[case_id]["location_mask"]))
        gts[case_id] = gt
    print("loaded GT volumes", flush=True)

    def perfect_prediction(case_id: str) -> np.ndarray:
        return gts[case_id].copy()

    def baseline_no_guess() -> list[tuple[str, np.ndarray]]:
        preds = []
        for case_id in sample_ids:
            pred = perfect_prediction(case_id)
            if case_id == true_case_id:
                pred[pred == TARGET_CLASS] = 0  # don't predict it even where it's true
            preds.append((case_id, pred))
        return preds

    def scenario(k: int, lucky: bool) -> list[tuple[str, np.ndarray]]:
        preds = []
        n_spurious_needed = k - 1 if lucky else k
        spurious_cases = absent_cases[:n_spurious_needed]
        for case_id in sample_ids:
            pred = perfect_prediction(case_id)
            if case_id == true_case_id:
                pred[pred == TARGET_CLASS] = 0
                if lucky:
                    jittered = jitter_mask(gts[case_id] == TARGET_CLASS, shift=(2, 1, 1))
                    pred[jittered] = TARGET_CLASS
            elif case_id in spurious_cases:
                pred = place_spurious_stub(pred, case_id, spurious_cases.index(case_id))
            preds.append((case_id, pred))
        return preds

    def score(preds: list[tuple[str, np.ndarray]]) -> dict:
        results = [score_case(pred, cid) for cid, pred in preds]
        return aggregate(results)

    def per_class_row(agg: dict, cls: int) -> dict:
        pc = agg["per_class"]
        return {k: round(float(pc[f"{k}_{cls}"]), 4) for k in
               ("PRECISION", "RECALL", "DICE", "HD95", "VOLSIM", "MCC")}

    print("scoring baseline (no guess)...", flush=True)
    agg_baseline = score(baseline_no_guess())
    baseline_row = per_class_row(agg_baseline, TARGET_CLASS)
    print(f"baseline (class {TARGET_CLASS}): {baseline_row}", flush=True)

    report = {"held_out_center": HELD_OUT_CENTER, "target_class": TARGET_CLASS,
             "true_case": true_case_id, "baseline_no_guess": baseline_row,
             "lucky": {}, "unlucky": {}}

    for k in K_VALUES:
        for lucky in (True, False):
            label = "lucky" if lucky else "unlucky"
            print(f"scoring k={k} {label}...", flush=True)
            agg_k = score(scenario(k, lucky))
            row = per_class_row(agg_k, TARGET_CLASS)
            report[label][k] = row
            print(f"  k={k} {label}: {row}", flush=True)

    out_json = OUT_DIR / "speculative_guessing_probe.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n")

    lines = [f"# Speculative over-prediction: class {TARGET_CLASS} in {HELD_OUT_CENTER} "
            f"(true instance in {true_case_id}, an '=1 instance' class)", "",
            f"Baseline (never predict class {TARGET_CLASS} anywhere): "
            f"`{json.dumps(baseline_row)}`", "",
            "## Lucky (1 guess overlaps the true lesion, imperfectly; k-1 further spurious guesses)", "",
            "| k | PRECISION | RECALL | DICE | HD95 | VOLSIM | MCC |", "|---|---:|---:|---:|---:|---:|---:|"]
    for k in K_VALUES:
        r = report["lucky"][k]
        lines.append(f"| {k} | {r['PRECISION']} | {r['RECALL']} | {r['DICE']} | {r['HD95']} | "
                     f"{r['VOLSIM']} | {r['MCC']} |")
    lines += ["", "## Unlucky (all k guesses spurious, zero overlap with the true lesion)", "",
             "| k | PRECISION | RECALL | DICE | HD95 | VOLSIM | MCC |", "|---|---:|---:|---:|---:|---:|---:|"]
    for k in K_VALUES:
        r = report["unlucky"][k]
        lines.append(f"| {k} | {r['PRECISION']} | {r['RECALL']} | {r['DICE']} | {r['HD95']} | "
                     f"{r['VOLSIM']} | {r['MCC']} |")

    out_md = OUT_DIR / "speculative_guessing_probe.md"
    out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out_json} and {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
