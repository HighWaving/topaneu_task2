"""Prepare TopAneu-MR scans for zero-shot Task020FG inference, without touching Task020's own tree.

2026-08-14. First attempt at this mixed TopAneu-MR images directly into
`Task020FG_LocalAneurysm/raw_splitted/imagesTs/` alongside the existing 90
ADAM cases -- that turned out to violate this project's own established
convention (see `prepare_nndet_lausanne.py`'s `assert_not_task020` /
`PROTECTED_TASK020_ROOT`, written specifically to stop new cohorts from being
mixed into that shared tree). It was also silently wrong on its own terms:
`nndet_prep`'s CLI never preprocesses `imagesTs` at all (traced through
`external/nnDetection/scripts/preprocess.py`'s `run()` -- it only calls
`run_cropping_and_convert`/`run_planning_and_process` against `imagesTr`).
Test-set preprocessing under an EXISTING plan requires calling nnDetection's
planner directly via `run_preprocessing_test`, which is exactly what
`prepare_nndet_lausanne.py` already implements. The 308 TopAneu-MR files were
removed from Task020FG's directory before writing this.

This reuses that script's already-reviewed safety helpers directly (import,
not copy) -- `assert_not_task020`, `ensure_symlink`,
`assert_only_expected_raw_images`/`_preprocessed`, `link_manifest_images`,
`run_preprocessing` -- and only supplies TopAneu-MR-specific manifest
selection, since `prepare_nndet_lausanne.py` itself is hardcoded to the
Lausanne cohort filter and a YAML config indirection this project's TopAneu
manifest doesn't use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/jovyan/rtx4claude-datavol-1/new_aneurysms")
from src.scripts.prepare_nndet_lausanne import (  # noqa: E402
    assert_not_task020,
    assert_only_expected_preprocessed,
    link_manifest_images,
    run_preprocessing,
    sha256,
    _atomic_json,
    _load_plan,
)

MANIFEST = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task1/artifacts/phase0/case_manifest.json")
DEFAULT_ARTIFACT_ROOT = Path("/home/jovyan/rtx4claude-datavol-1/new_aneurysms/artifacts/nndet_topaneu_mr_zeroshot")


def select_topaneu_mr_records(manifest_path: Path, expected_scans: int) -> list[dict]:
    payload = json.loads(manifest_path.read_text())
    cases = payload["cases"]
    selected = [c for c in cases if c["modality"] == "mr"]
    if len(selected) != expected_scans:
        raise ValueError(f"manifest has {len(selected)} MR scans; expected {expected_scans}")
    records = [{"case_id": c["case_id"], "image": c["image"]} for c in selected]
    case_ids = [r["case_id"] for r in records]
    duplicates = sorted({c for c in case_ids if case_ids.count(c) > 1})
    if duplicates:
        raise ValueError(f"duplicate case_ids: {duplicates}")
    return sorted(records, key=lambda r: r["case_id"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True,
                        help="fold directory containing the authoritative plan.pkl")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--links-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assert_not_task020(args.artifact_root)

    records = select_topaneu_mr_records(MANIFEST, expected_scans=308)
    plan, plan_path = _load_plan(args.training_dir)
    if int(plan["num_modalities"]) != 1:
        raise ValueError(f"expected one-modality TOF plan, got {plan['num_modalities']}")

    raw_splitted_root = args.artifact_root / "raw_splitted"
    raw_images = raw_splitted_root / "imagesTs"
    preprocessed_root = args.artifact_root / "preprocessed"
    preprocessed_images = preprocessed_root / str(plan["data_identifier"]) / "imagesTs"
    case_ids = [r["case_id"] for r in records]
    assert_only_expected_preprocessed(preprocessed_images, case_ids)

    link_summary = link_manifest_images(records, raw_images)
    print(f"links: {link_summary}", flush=True)

    contract = {
        "dataset": "topaneu_mr", "expected_scans": 308, "case_ids": case_ids,
        "manifest": str(MANIFEST.resolve()), "manifest_sha256": sha256(MANIFEST),
        "training_plan": str(plan_path.resolve()), "training_plan_sha256": sha256(plan_path),
        "planner_id": str(plan["planner_id"]), "data_identifier": str(plan["data_identifier"]),
        "target_spacing": [float(v) for v in plan["target_spacing"]],
        "raw_splitted_root": str(raw_splitted_root.resolve()),
        "preprocessed_root": str(preprocessed_root.resolve()),
    }
    contract_path = args.artifact_root / "preparation_contract.json"
    if contract_path.is_file():
        previous = json.loads(contract_path.read_text())
        immutable = ("dataset", "expected_scans", "case_ids", "manifest_sha256",
                    "training_plan_sha256", "planner_id", "data_identifier",
                    "target_spacing", "raw_splitted_root", "preprocessed_root")
        changed = [k for k in immutable if previous.get(k) != contract.get(k)]
        if changed:
            raise RuntimeError(f"existing contract differs in {changed}; use a new artifact root")
    else:
        _atomic_json(contract, contract_path)

    if not args.links_only:
        run_preprocessing(training_dir=args.training_dir, preprocessed_root=preprocessed_root,
                          raw_splitted_root=raw_splitted_root, plan=plan,
                          num_processes=args.num_processes)

    completed = sum((preprocessed_images / f"{c}.npz").is_file() for c in case_ids)
    print(f"preprocessed: {completed}/{len(case_ids)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
