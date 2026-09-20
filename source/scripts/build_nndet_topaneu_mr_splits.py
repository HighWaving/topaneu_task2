"""Leave-one-center-out splits_final.pkl for the Task030FG_TopAneuMR nndet task.

2026-08-14, user-directed (relayed): after nndet_prep, train the detector and
measure the REAL candidate ceiling on a LOCO fold -- validating the ~0.99
estimate derived from the size-distribution weighting. nnDetection expects
`preprocessed/splits_final.pkl`: a list of dicts, each `{"train": [...],
"val": [...]}` of case-id strings (verified against Task020FG_LocalAneurysm's
existing file, not guessed).

Reuses topaneu2026_task2/configs/loco_splits.json's center-based fold
definitions, filtered to only the case_ids that actually exist in this
MR-only task. center4 is CTA-only, so it contributes zero MR cases -- the
resulting split has 3 folds (center1/center2/center5 held out in turn), not
4, purely because center4 has no MR data to hold out.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

LOCO_SPLITS = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/configs/loco_splits.json")
MANIFEST = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task1/artifacts/phase0/case_manifest.json")
TASK_DIR = Path("/home/jovyan/rtx4claude-datavol-1/nndet_data/Task030FG_TopAneuMR")


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    mr_case_ids = {c["case_id"] for c in manifest["cases"] if c["modality"] == "mr"}

    loco = json.loads(LOCO_SPLITS.read_text())
    folds = []
    fold_centers = []
    for center, fold in loco["folds"].items():
        val = sorted(set(fold["val"]) & mr_case_ids)
        train = sorted(set(fold["train"]) & mr_case_ids)
        if not val:
            print(f"skipping {center}: 0 MR cases held out (CTA-only center)")
            continue
        folds.append({"train": train, "val": val})
        fold_centers.append(center)
        print(f"fold (held-out {center}): train={len(train)}, val={len(val)}")

    assert sum(len(f["val"]) for f in folds) == len(mr_case_ids), \
        "every MR case should appear in exactly one fold's val set"

    out_dir = TASK_DIR / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "splits_final.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(folds, f)

    (out_dir / "splits_final_fold_centers.json").write_text(
        json.dumps({"fold_centers": fold_centers}, indent=2) + "\n")

    print(f"wrote {out_path} ({len(folds)} folds) and fold_centers mapping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
