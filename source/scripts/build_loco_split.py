"""Builds a leave-one-center-out (LOCO) split for TopAneu Task 2.

2026-08-14, final pivot priority #2 (after the local scoring arena). TopAneu's
417 cases come from 4 centers (center1 CHUV 200 MRA, center2 Geneva 47 CTA +
40 MRA, center4 Mie Chuo 62 CTA, center5 public 68 MRA). No patient group
spans multiple centers (checked directly against case_manifest.json --
longitudinal_patient_groups only lists repeat scans WITHIN center4), so a
center-based split is patient-clean by construction, no grouping logic
needed beyond "group by center".

One fold per center: that center held out as validation, the other three
pooled as train. This is deliberately NOT a single train/val split -- LOCO
is the whole point, since the eventual grand-challenge test set is scanners
this model has never seen, and a random in-distribution split would not
surface that gap the way holding out an entire center does.

center5 is itself the public Lausanne OpenNeuro cohort (id `4xx`, see
session plan) -- held out as its own fold like any other center, no special
casing.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

MANIFEST = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task1/artifacts/phase0/case_manifest.json")
OUT_DIR = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/configs")


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    cases = manifest["cases"]
    centers = sorted({c["center"] for c in cases})

    # Patient-group / center consistency check -- if this ever fails, a
    # naive per-center split would leak a patient's other scan into train.
    pg_centers: dict[str, set[str]] = {}
    for c in cases:
        pg_centers.setdefault(c["patient_group"], set()).add(c["center"])
    leaking = {pg: cs for pg, cs in pg_centers.items() if len(cs) > 1}
    assert not leaking, f"patient groups spanning multiple centers, LOCO would leak: {leaking}"

    folds = {}
    for held_out in centers:
        val_ids = sorted(c["case_id"] for c in cases if c["center"] == held_out)
        train_ids = sorted(c["case_id"] for c in cases if c["center"] != held_out)
        folds[held_out] = {"train": train_ids, "val": val_ids}

    # Sanity: every case appears in exactly one fold's val set, and in
    # train for every other fold.
    val_membership = Counter()
    for fold in folds.values():
        for case_id in fold["val"]:
            val_membership[case_id] += 1
    assert set(val_membership.keys()) == {c["case_id"] for c in cases}
    assert all(n == 1 for n in val_membership.values())

    out = {
        "schema_version": 1,
        "description": "Leave-one-center-out split. Each key is the held-out "
                       "center's fold; 'val' = that center only, 'train' = "
                       "the other three centers pooled.",
        "source_manifest": str(MANIFEST),
        "centers": centers,
        "folds": folds,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "loco_splits.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")

    print(f"wrote {out_path}")
    for held_out, fold in folds.items():
        print(f"  fold {held_out}: train={len(fold['train'])}, val={len(fold['val'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
