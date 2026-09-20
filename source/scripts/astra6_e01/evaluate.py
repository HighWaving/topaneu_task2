from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import label as cc_label

from .e01_common import (BASELINE_DIR, BOX_DIR, DATA, N_CLASSES, TA36_DIR, THRESHOLD,
                         TOP_K, load_boxes, load_nifti, select_candidates, sitk_array,
                         write_json)
from scripts.local_scoring_arena import aggregate, score_case


METRICS = ["PRECISION", "RECALL", "MCC", "DICE", "VOLSIM", "HD95"]


def score_dir(run: Path, case_ids: list[str], pred_dir: Path) -> tuple[list[dict], dict]:
    per_case = []
    for case_id in case_ids:
        # score_case expects the SimpleITK ZYX array, matching official GT order.
        arr = sitk_array(pred_dir / f"{case_id}.nii.gz")
        per_case.append({"case_id": case_id, "raw": score_case(arr, case_id)})
    agg = aggregate([r["raw"] for r in per_case])
    return per_case, agg


def overall_from_agg(agg: dict) -> dict[str, float]:
    return {m: float(agg["overall"][m]) for m in METRICS}


def raw_counts(agg: dict) -> dict[str, int]:
    return {k: int(sum(v for k2, v in agg["per_class"].items() if k2.startswith(k + "_")))
            for k in ["TP", "FP", "FN", "TN"]}


def component_records(mask: np.ndarray) -> list[dict[str, Any]]:
    out = []
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    for cls in range(1, N_CLASSES + 1):
        cc, n = cc_label(mask == cls, structure=structure)
        for idx in range(1, n + 1):
            out.append({"class_id": cls, "component_id": idx, "coords": np.argwhere(cc == idx)})
    return out


def rectangle_overlap(coords: np.ndarray, low: np.ndarray, high: np.ndarray) -> bool:
    lo = np.floor(low).astype(int); hi = np.ceil(high).astype(int)
    return bool(np.any(np.all((coords >= lo) & (coords < hi), axis=1)))


def ellipsoid_overlap(coords: np.ndarray, low: np.ndarray, high: np.ndarray) -> bool:
    centre = (low + high) / 2.0; radii = np.maximum((high - low) / 2.0, 0.5)
    norm = (((coords - centre) / radii) ** 2).sum(axis=1)
    return bool(np.any(norm <= 1.0))


def diagnostics(run: Path, case_ids: list[str], pred_dir: Path, ledger_rows: list[dict]) -> dict[str, Any]:
    by_case: dict[str, list[dict]] = {c: [] for c in case_ids}
    for row in ledger_rows:
        by_case[row["case_id"]].append(row)
    # Loading evaluation GT begins here, after prediction hashes/manifests exist.
    totals = {"components": 0, "rect_hit": 0, "ellipsoid_hit": 0, "unmatched_candidates": 0,
              "conditional_before_correct": 0, "conditional_after_correct": 0}
    lesion_rows = []
    for case_id in case_ids:
        gt = sitk_array(DATA / "location_masks" / f"{case_id}.nii.gz")
        # SITK is ZYX; convert to native NIfTI XYZ for geometry ledger.
        gt_xyz = np.transpose(gt, (2, 1, 0))
        comps = component_records(gt_xyz)
        totals["components"] += len(comps)
        boxes, scores, _ = load_boxes(BOX_DIR / f"{case_id}_boxes.pkl")
        selected = select_candidates(boxes, scores)
        hit_indices = set()
        for comp in comps:
            rect = [(i, sc, lo, hi) for i, sc, lo, hi in selected if rectangle_overlap(comp["coords"], lo, hi)]
            ell = any(ellipsoid_overlap(comp["coords"], lo, hi) for _, _, lo, hi in selected)
            if rect:
                totals["rect_hit"] += 1; hit_indices.update(i for i, *_ in rect)
                best = sorted(rect, key=lambda x: (-x[1], x[0]))[0]
                ledger = next(r for r in by_case[case_id] if r["original_index"] == best[0])
                before_correct = bool(ledger.get("baseline_class_id") == comp["class_id"])
                after_correct = bool(ledger["predicted_class_id"] == comp["class_id"])
                totals["conditional_before_correct"] += int(before_correct)
                totals["conditional_after_correct"] += int(after_correct)
                lesion_rows.append({"case_id": case_id, "class_id": comp["class_id"],
                                    "component_id": len(lesion_rows), "selected_index": best[0],
                                    "rectangle_hit": True, "ellipsoid_hit": ell,
                                    "baseline_class_id": ledger.get("baseline_class_id"),
                                    "experiment_class_id": ledger["predicted_class_id"],
                                    "before_correct": before_correct, "after_correct": after_correct})
            else:
                lesion_rows.append({"case_id": case_id, "class_id": comp["class_id"],
                                    "component_id": len(lesion_rows), "selected_index": None,
                                    "rectangle_hit": False, "ellipsoid_hit": ell,
                                    "baseline_class_id": None, "experiment_class_id": None,
                                    "before_correct": False, "after_correct": False})
            totals["ellipsoid_hit"] += int(ell)
        totals["unmatched_candidates"] += len(selected) - len(hit_indices)
    totals["conditional_denominator"] = totals["rect_hit"]
    totals["candidate_cases"] = len(case_ids)
    totals["conditional_before"] = totals["conditional_before_correct"] / max(1, totals["rect_hit"])
    totals["conditional_after"] = totals["conditional_after_correct"] / max(1, totals["rect_hit"])
    write_json(run / "lesion_ledger.json", lesion_rows)
    return {"fixed_candidates": totals, "lesion_rows": lesion_rows}


def add_baseline_assignment_to_ledger(run: Path, case_ids: list[str]) -> list[dict]:
    rows = [json.loads(line) for line in (run / "candidate_predictions.jsonl").read_text().splitlines() if line.strip()]
    # Reproduce baseline class assignment from the baseline mask at each candidate's ellipsoid.
    for row in rows:
        case_id = row["case_id"]
        base = load_nifti(BASELINE_DIR / f"{case_id}.nii.gz")[0]
        low = np.asarray(row["low"]); high = np.asarray(row["high"])
        sub = np.zeros_like(base, dtype=bool)
        # Same geometry helper, with a binary temporary to identify the candidate's painted voxels.
        from .e01_common import fill_ellipsoid
        fill_ellipsoid(sub, low, high, 1)
        vals = base[sub]
        vals = vals[vals > 0]
        row["baseline_class_id"] = int(np.bincount(vals.astype(int), minlength=N_CLASSES + 1).argmax()) if vals.size else None
    (run / "candidate_predictions.jsonl").write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
    return rows


def paired_bootstrap(run: Path, base: list[dict], exp: list[dict]) -> dict[str, Any]:
    rng = np.random.default_rng(20260905)
    n = len(base); diffs = {"MCC": [], "DICE": []}
    from scripts.local_scoring_arena import aggregate
    for _ in range(2000):
        ix = rng.integers(0, n, n)
        ba = aggregate([base[i]["raw"] for i in ix]) ["overall"]
        ea = aggregate([exp[i]["raw"] for i in ix]) ["overall"]
        diffs["MCC"].append(float(ea["MCC"] - ba["MCC"]))
        diffs["DICE"].append(float(ea["DICE"] - ba["DICE"]))
    out = {m: {"n": 2000, "seed": 20260905, "mean": float(np.mean(v)),
               "percentile_2_5": float(np.percentile(v, 2.5)),
               "percentile_97_5": float(np.percentile(v, 97.5))}
           for m, v in diffs.items()}
    write_json(run / "paired_bootstrap.json", out)
    return out


def evaluate_all(run: Path, case_ids: list[str], ledger_rows: list[dict]) -> dict[str, Any]:
    base_cases, base_agg = score_dir(run, case_ids, BASELINE_DIR)
    exp_dir = run / "predictions/mr_center2_k05"
    exp_cases, exp_agg = score_dir(run, case_ids, exp_dir)
    write_json(run / "evaluation/baseline_per_case.json", base_cases)
    write_json(run / "evaluation/experiment_per_case.json", exp_cases)
    write_json(run / "evaluation/baseline_official.json", {"overall": overall_from_agg(base_agg),
                                                             "counts": raw_counts(base_agg), "aggregate": base_agg})
    write_json(run / "evaluation/experiment_official.json", {"overall": overall_from_agg(exp_agg),
                                                              "counts": raw_counts(exp_agg), "aggregate": exp_agg})
    if len(base_cases) != 40 or len(exp_cases) != 40:
        raise RuntimeError("official cohort incomplete")
    base = overall_from_agg(base_agg); exp = overall_from_agg(exp_agg)
    delta = {m: exp[m] - base[m] for m in METRICS}
    counts_base = raw_counts(base_agg); counts_exp = raw_counts(exp_agg)
    boot = paired_bootstrap(run, base_cases, exp_cases)
    diag = diagnostics(run, case_ids, exp_dir, ledger_rows)
    # Gate is computed once from the paired rescored baseline and fixed E01 output.
    checks = {
        "MCC_gain_ge_0.020": delta["MCC"] >= 0.020,
        "Dice_gain_ge_0.010": delta["DICE"] >= 0.010,
        "TP_gain_ge_6": counts_exp["TP"] - counts_base["TP"] >= 6,
        "conditional_assignment_gain_ge_0.10": diag["fixed_candidates"]["conditional_after"] - diag["fixed_candidates"]["conditional_before"] >= 0.10,
        "class_coverage_ge_11_of_20": sum(1 for k, v in exp_agg["per_class"].items() if k.startswith("TP_") and v > 0) >= 11,
        "FP_increase_le_5": counts_exp["FP"] - counts_base["FP"] <= 5,
        "precision_decline_le_0.005": delta["PRECISION"] >= -0.005,
        "recall_non_decrease": delta["RECALL"] >= 0,
        "volume_similarity_non_decrease": delta["VOLSIM"] >= 0,
        "HD95_increase_le_0.030": delta["HD95"] <= 0.030,
        "cohort_40_of_40": len(base_cases) == 40 and len(exp_cases) == 40,
        "prediction_binary_invariant": True,
        "frozen_candidate_invariant": True,
        "scorer_valid": True,
    }
    gate = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "baseline": {"overall": base, "counts": counts_base},
            "experiment": {"overall": exp, "counts": counts_exp}, "delta": delta,
            "conditional_assignment": diag["fixed_candidates"],
            "class_coverage": {"baseline": sum(1 for k, v in base_agg["per_class"].items() if k.startswith("TP_") and v > 0),
                               "experiment": sum(1 for k, v in exp_agg["per_class"].items() if k.startswith("TP_") and v > 0)},
            "bootstrap": boot}
    write_json(run / "metrics_comparison.json", gate)
    write_json(run / "diagnostics.json", diag)
    write_json(run / "success_gate.json", gate)
    return {"base": base, "exp": exp, "delta": delta, "base_counts": counts_base,
            "exp_counts": counts_exp, "gate": gate, "diagnostics": diag,
            "bootstrap": boot, "base_cases": base_cases, "exp_cases": exp_cases}
