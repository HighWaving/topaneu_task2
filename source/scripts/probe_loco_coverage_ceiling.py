"""Per-LOCO-fold distinct-class coverage ceiling.

2026-08-14, peer-requested follow-up to the FP marginal cost probe.
evaluation_average() divides unweighted by N_CLASSES=52, and per-class
credit only requires any-overlap hit -- so a class with 1 true instance
and a class with 44 both contribute the same 1/52 if hit at all. This
means the real lever is HOW MANY DISTINCT CLASSES get hit at least once,
not per-instance segmentation quality. That makes "distinct classes
present in a held-out center's GT / 52" the theoretical Dice/Precision/
Recall/VS ceiling for that fold, regardless of how good the model is --
no team can beat it.

Also cross-checks whether the 9 classes with zero instances in the FULL
417-case training population (3,9,10,12,13,14,18,20,52 -- see
fp_marginal_cost_probe.json) are also zero within each individual
held-out center specifically. This matters because the placeholder-class
recommendation (use a globally-zero class for low-confidence guesses)
assumes the held-out center's distribution doesn't concentrate lesions
into one of those "globally rare" classes -- worth checking, not assuming.

Uses case_manifest.json for case->location_mask path lookup only; the class
list per case is recomputed LIVE from the location_mask volume on disk, not
from the manifest's cached "locations" field. 2026-08-15: the manifest
(shared with the Task1 line, not ours to edit) was built before TopAneu's
08-14 dataset update -- its cached "locations" field is stale for 8 cases
whose location_masks changed (mostly L/R or class-ID swaps, one real
geometry change on center1_mr_702) and it still lists center1_mr_150, which
the server has since removed entirely. Recomputing from the mask file
directly self-corrects both: a case whose mask file no longer exists on
disk is skipped automatically, and every remaining case reflects
data_topaneu26/'s current (peer-corrected, 2026-08-15) content.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

MANIFEST = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task1/artifacts/phase0/case_manifest.json")
LOCO_SPLITS = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/configs/loco_splits.json")
OUT_DIR = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/reports")

GLOBAL_ZERO_CLASSES = {3, 9, 10, 12, 13, 14, 18, 20, 52}
N_CLASSES = 52


def live_locations(location_mask_path: str) -> list[int]:
    p = Path(location_mask_path)
    if not p.is_file():
        return []  # e.g. center1_mr_150, removed server-side 2026-08-14
    arr = np.asarray(nib.load(str(p)).dataobj)
    return [int(v) for v in np.unique(arr) if v != 0]


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    cases_by_id = {c["case_id"]: c for c in manifest["cases"]}
    loco = json.loads(LOCO_SPLITS.read_text())

    report = {"n_classes": N_CLASSES, "global_zero_classes": sorted(GLOBAL_ZERO_CLASSES), "folds": {}}
    lines = ["# LOCO per-fold distinct-class coverage ceiling", "",
            "No submission can beat `n_distinct_classes_in_val_gt / 52` on Precision/Recall/"
            "Dice/VS for a given fold's held-out center, since evaluation_average() averages "
            "unweighted over all 52 classes and per-class credit only needs any-overlap.", "",
            "Recomputed 2026-08-15 against the peer-corrected `data_topaneu26/` (post 08-14 "
            "TopAneu dataset update) -- classes read live from each case's location_mask file, "
            "not the (now stale for 8 cases) manifest cache; `center1_mr_150` drops out "
            "automatically since its mask file no longer exists.", "",
            "| held-out center | n cases | n distinct classes in GT | ceiling (n/52) | "
            "global-zero classes also zero here | global-zero classes NONZERO here (placeholder assumption breaks) |",
            "|---|---:|---:|---:|---|---|"]

    for held_out, fold in loco["folds"].items():
        val_ids = [c for c in fold["val"]
                  if c in cases_by_id and Path(cases_by_id[c]["location_mask"]).is_file()]
        class_counts = Counter()
        for case_id in val_ids:
            for cls in live_locations(cases_by_id[case_id]["location_mask"]):
                class_counts[cls] += 1
        distinct = sorted(class_counts.keys())
        ceiling = len(distinct) / N_CLASSES

        zero_here = sorted(GLOBAL_ZERO_CLASSES - set(distinct))
        nonzero_here = sorted(GLOBAL_ZERO_CLASSES & set(distinct))

        report["folds"][held_out] = {
            "n_cases": len(val_ids),
            "n_distinct_classes": len(distinct),
            "ceiling": round(ceiling, 4),
            "class_counts": dict(sorted(class_counts.items())),
            "global_zero_classes_still_zero_here": zero_here,
            "global_zero_classes_nonzero_here": nonzero_here,
        }

        nz_str = ", ".join(str(c) for c in nonzero_here) if nonzero_here else "none -- assumption holds"
        lines.append(f"| {held_out} | {len(val_ids)} | {len(distinct)} | {ceiling:.3f} | "
                     f"{len(zero_here)}/9 | {nz_str} |")

    out_json = OUT_DIR / "loco_coverage_ceiling.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n")

    lines += ["", "## Per-fold class occurrence counts", ""]
    for held_out, fold in report["folds"].items():
        lines.append(f"**{held_out}** (n={fold['n_cases']}, {fold['n_distinct_classes']} distinct classes): "
                     f"{fold['class_counts']}")
        lines.append("")

    out_md = OUT_DIR / "loco_coverage_ceiling.md"
    out_md.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nwrote {out_json} and {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
