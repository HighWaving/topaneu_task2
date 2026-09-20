from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import test_invariants
from .dataset import build_features, build_manifest
from .e01_common import (BASELINE_DIR, BOX_DIR, COHORT_PATH, DATA, P, SEED, TA36_DIR,
                         label_mappings, sha256_file, sha256_tree, write_json)
from .evaluate import add_baseline_assignment_to_ledger, evaluate_all
from .infer import run_inference
from .train import fit_locked


def new_run() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = P / "artifacts" / f"astra6_e01_vessel_signature_52class_{stamp}"
    if root.exists():
        raise RuntimeError(f"run collision: {root}")
    root.mkdir(parents=True)
    for d in ("logs", "evaluation", "features", "model", "predictions/mr_center2_k05"):
        (root / d).mkdir(parents=True, exist_ok=True)
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=None)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    test_invariants.run()
    run = args.run or new_run()
    started = time.time(); stages = {}
    (run / "run_started_utc.txt").write_text(datetime.now(timezone.utc).isoformat() + "\n")
    write_json(run / "eval_case_ids.json", json.loads(COHORT_PATH.read_text()))
    loc_names, ves_names = label_mappings()
    # Provenance records frozen inputs before fitting. Evaluation GT is intentionally absent here.
    provenance = {"plan": str(P / "ASTRA6_TASK2_PLAN.md"), "plan_sha256": sha256_file(P / "ASTRA6_TASK2_PLAN.md"),
                  "checkpoint": str(P / "artifacts/fold1_checkpoint_snapshots/epoch60/model_last.ckpt"),
                  "checkpoint_sha256_recorded": "f84488d9dfce235fc993c5dc9faca4fd35501b134b1a26a0248b4a309443980e",
                  "boxes_dir": str(BOX_DIR), "ta36_dir": str(TA36_DIR), "baseline_dir": str(BASELINE_DIR),
                  "cohort_path": str(COHORT_PATH), "location_mapping_sha256": sha256_file(DATA / "location_mapping.json"),
                  "vessel_mapping_sha256": sha256_file(DATA / "vessel_mapping.json"),
                  "readme_sha256": sha256_file(DATA / "README.md"), "seed": SEED,
                  "evaluation_gt_loaded_before_prediction_hash": False}
    write_json(run / "provenance.json", provenance)
    t = time.time(); manifest = build_manifest(run); stages["manifest_seconds"] = time.time() - t
    t = time.time(); data = build_features(run, manifest, loc_names, ves_names); stages["feature_seconds"] = time.time() - t
    t = time.time(); fitted = fit_locked(run, data); stages["fit_seconds"] = time.time() - t
    t = time.time(); eval_ids = json.loads(COHORT_PATH.read_text()); infer = run_inference(run, fitted["model"], eval_ids, loc_names, ves_names); stages["inference_seconds"] = time.time() - t
    # Prediction tree and candidate ledger are complete and hashed before evaluation GT is read.
    (run / "PREDICTIONS_HASHED_BEFORE_EVAL_GT").write_text(infer["prediction_tree_sha256"] + "\n")
    rows = add_baseline_assignment_to_ledger(run, eval_ids)
    t = time.time(); results = evaluate_all(run, eval_ids, rows); stages["evaluation_seconds"] = time.time() - t
    stages["total_seconds"] = time.time() - started
    write_json(run / "runtime.json", {"stages_seconds": stages, "total_seconds": stages["total_seconds"],
                                      "cpu_only": True, "cuda_visible_devices": "", "max_workers": 4,
                                      "python": sys.executable, "platform": platform.platform(),
                                      "reproduction_command": f"CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 {sys.executable} -m scripts.astra6_e01.run_e01 --run {run}",
                                      "disk_output_bytes": sum(p.stat().st_size for p in run.rglob('*') if p.is_file())})
    # Hash all new code/config/model/predictions after they are finalized.
    write_json(run / "hashes.json", {"code_tree_sha256": sha256_tree(P / "scripts/astra6_e01"),
                                     "predictions_tree_sha256": infer["prediction_tree_sha256"],
                                     "model_sha256": sha256_file(fitted["path"]),
                                     "config_sha256": sha256_file(run / "provenance.json")})
    write_result(run, manifest, fitted, results, stages)
    write_json(run / "DONE.json", {"status": "DONE", "valid": True, "gate": results["gate"]["status"],
                                   "n_cases": 40, "prediction_hash": infer["prediction_tree_sha256"]})
    print(json.dumps({"run": str(run), "status": results["gate"]["status"], "metrics": results["exp"],
                      "delta": results["delta"], "counts": results["exp_counts"],
                      "conditional": results["diagnostics"]["fixed_candidates"],
                      "runtime_seconds": stages["total_seconds"]}, indent=2))
    return 0


def write_result(run: Path, manifest: dict, fitted: dict, r: dict, stages: dict) -> None:
    b, e, d = r["base"], r["exp"], r["delta"]
    bc, ec = r["base_counts"], r["exp_counts"]
    diag = r["diagnostics"]["fixed_candidates"]
    coverage = r["gate"]["class_coverage"]
    lines = [
        "# RESULT_TASK2_ASTRA6_E01", "", "Baseline: MR fold1 epoch60 × center2, 40 cases, predicted TA36, threshold 0.3, top-k 5, ellipsoid.",
        "Experiment: fixed supervised 295-feature multi-vessel ExtraTrees 52-class assignment.",
        "Exact change: candidate class assignment only; detector boxes/scores, candidate selection, geometry and draw order frozen.", "",
        "Official metrics before:", json.dumps(b, indent=2), "Official metrics after:", json.dumps(e, indent=2),
        "Effect size (after - before):", json.dumps(d, indent=2),
        "TP/FP/FN before: %s; after: %s" % ("/".join(str(bc[k]) for k in ("TP", "FP", "FN")), "/".join(str(ec[k]) for k in ("TP", "FP", "FN"))),
        "Conditional assignment before: %.10f (%d/%d); after: %.10f (%d/%d)" % (diag["conditional_before"], diag["conditional_before_correct"], diag["conditional_denominator"], diag["conditional_after"], diag["conditional_after_correct"], diag["conditional_denominator"]),
        "Class coverage before → after: %s/20 → %s/20" % (coverage["baseline"], coverage["experiment"]),
        "", "Diagnostics:", json.dumps({k: diag[k] for k in diag if k not in ("candidate_cases",)}, indent=2),
        "Paired bootstrap (2000, seed 20260905):", json.dumps(r["bootstrap"], indent=2),
        "", "Runtime:", json.dumps(stages, indent=2),
        "Problems / limitations: training uses released organizer-predicted vessel masks as silver anatomy while evaluation uses cached TA36 predictions; training uses GT location components with fixed jitter/mirror while evaluation uses detector boxes; center2 has repeated historical research use and all 40 cases are positive, so this is not a pristine blind test or specificity estimate.",
        "MR150 was excluded; MR702 uses the current location mask. CT center2 and CT150 were excluded/parked. Official scorer and special TN/MCC definition were preserved.",
        "", "Success gate: %s" % r["gate"]["status"], json.dumps(r["gate"]["checks"], indent=2),
        "Validity: PASS; complete 40/40 paired scoring, prediction hash before evaluation GT, binary foreground invariant, frozen candidate invariant, official scorer.",
        "Recommendation based on evidence: see gate and diagnostics; do not execute a downstream experiment in this run.",
        "", "STOPPED. Waiting for Astra review.",
    ]
    (run / "RESULT_TASK2_ASTRA6_E01.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
