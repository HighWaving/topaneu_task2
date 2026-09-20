"""M1 official scoring (2026-08-15): fold1 checkpoint x center2 (40 MR cases,
honest 52-class assignment) x the OFFICIAL evaluate.py, wrapped via
local_scoring_arena.py (never reimplemented).

Coverage denominator is classes ACTUALLY PRESENT in this 40-case subset
(TP_i + FN_i > 0), not all 52 -- peer-directed correction, since 40 cases /
~58 lesions cannot possibly cover 52 classes and /52 would count "class
doesn't occur here" as "we missed it".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from local_scoring_arena import aggregate, score_case  # noqa: E402


def territory(name: str) -> int:
    tok = name.split(" ")[0].split("-")[-1]
    return int(tok.split(".")[0])


POSTERIOR = {1, 2}  # VB, PCA
ANTERIOR = {3, 4, 5}  # ICA, ACA, MCA


def load_prediction(path: Path) -> np.ndarray:
    import SimpleITK as sitk
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--case-ids-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    case_ids = json.loads(args.case_ids_json.read_text())
    location_names = {v: k for k, v in
                      json.loads(Path("/home/jovyan/rtx4claude-datavol-1/data_topaneu26/"
                                      "location_mapping.json").read_text())["labels"].items()}

    results, missing = [], []
    for case_id in case_ids:
        pred_path = args.predictions_dir / f"{case_id}.nii.gz"
        if not pred_path.is_file():
            missing.append(case_id)
            continue
        pred = load_prediction(pred_path)
        results.append(score_case(pred, case_id))

    print(f"scored {len(results)}/{len(case_ids)} cases ({len(missing)} missing predictions)", flush=True)
    if missing:
        print(f"missing: {missing}", flush=True)
    if not results:
        print("no predictions scored, aborting", flush=True)
        return 1

    agg = aggregate(results)
    per_class = agg["per_class"]

    present, hit = [], []
    territory_present = {"posterior": [], "anterior": [], "unmapped": []}
    territory_hit = {"posterior": [], "anterior": [], "unmapped": []}
    for i in range(1, 53):
        tp, fn = per_class.get(f"TP_{i}", 0), per_class.get(f"FN_{i}", 0)
        is_present = (tp + fn) > 0
        is_hit = tp > 0
        if not is_present:
            continue
        present.append(i)
        if is_hit:
            hit.append(i)
        name = location_names.get(i, "")
        terr = territory(name) if name else None
        bucket = "posterior" if terr in POSTERIOR else "anterior" if terr in ANTERIOR else "unmapped"
        territory_present[bucket].append(i)
        if is_hit:
            territory_hit[bucket].append(i)

    coverage = {
        "n_classes_present_in_center2_40cases": len(present),
        "n_classes_hit": len(hit),
        "coverage_fraction": len(hit) / len(present) if present else None,
        "present_class_ids": present, "hit_class_ids": hit,
        "by_territory": {
            b: {"n_present": len(territory_present[b]), "n_hit": len(territory_hit[b]),
                "coverage": len(territory_hit[b]) / len(territory_present[b])
                            if territory_present[b] else None}
            for b in ("posterior", "anterior", "unmapped")
        },
    }

    # Supplementary: same 6 metrics averaged only over classes actually
    # present in this subset -- diagnostic only, NOT the official number
    # (the official number always divides by all 52, mandated by evaluate.py
    # and unaffected by anything here), but the /52 official number is
    # heavily diluted when only ~10-15/52 classes occur in 40 cases, so this
    # supplementary view is what's actually informative about model quality.
    metric_names = ("DICE", "VOLSIM", "HD95", "PRECISION", "RECALL", "MCC")
    present_only_avg = {
        m: float(np.mean([per_class[f"{m}_{i}"] for i in present])) if present else None
        for m in metric_names
    }

    out = {
        "n_scored": len(results), "n_missing": len(missing),
        "official_overall_52class_average": agg["overall"],
        "present_classes_only_average_DIAGNOSTIC_NOT_OFFICIAL": present_only_avg,
        "coverage_center2_denominator": coverage,
        "per_class": per_class,
    }
    args.output.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print("official overall (52-class average, as GC will report it):")
    print(json.dumps(agg["overall"], indent=2, default=float))
    print("\npresent-classes-only average (diagnostic, NOT the official number):")
    print(json.dumps(present_only_avg, indent=2, default=float))
    print("\ncoverage (denominator = classes present in center2's 40 cases, NOT 52):")
    print(json.dumps(coverage, indent=2, default=float))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
