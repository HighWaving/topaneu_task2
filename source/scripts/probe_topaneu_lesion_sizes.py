"""TopAneu 417-case lesion size distribution, on the SAME convention as in-house/ADAM.

2026-08-14, user-directed (relayed). Scope corrected to a single project,
Task 2 only, three cohorts (TopAneu 417 + in-house + ADAM 90) pooled for the
detector/segmentation, but 52-class location assignment still TopAneu-only
(only TopAneu has location labels) -- so the long-tail coverage problem does
NOT get fixed by pooling; geometry is still the lever for it.

This checks whether TopAneu's lesions are the same SIZE population as the
in-house/ADAM cohorts already characterised, which determines whether the
small-lesion findings from those cohorts (candidate ceiling only 0.757 for
<2mm, the isotropic-spacing motivation for A3) transfer to TopAneu at all.

**Must use max_diameter (longest pairwise extent), not equivalent-sphere
diameter.** This project was burned by exactly this mix-up once already
(`adam90_baseline_findings.md`: in-house reported equivalent-sphere, published
cohorts report max diameter, median ratio ~1.309x -- conflating the two
inverted a real conclusion). `phase1b_mra_vs_cta_comparison.py` in this same
workspace computes `lesion_equivalent_diameter_mm` -- NOT reusable here, it's
the wrong convention for this comparison. This script instead imports
`max_diameter_mm` directly from `new_aneurysms/src/eval/matching.py` (the
exact function used to bin the in-house/ADAM cohorts already reported), so
bin edges and diameter definition are identical by construction, not by
manual replication.

Reuses `topaneu_location_class_counts.py`'s connected-component-per-class
approach (26-connectivity) but adds per-lesion diameter, and reads with
nibabel (matching `max_diameter_mm`'s own affine convention) rather than
SimpleITK.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, "/home/jovyan/rtx4claude-datavol-1/new_aneurysms")
from src.eval.matching import max_diameter_mm  # noqa: E402

DATA_ROOT = Path("/home/jovyan/rtx4claude-datavol-1/data_topaneu26")
MANIFEST = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task1/artifacts/phase0/case_manifest.json")
OUT_DIR = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/reports")

# Known reference numbers (already computed elsewhere in the project, not
# recomputed here -- both already use max_diameter_mm from the same source).
KNOWN_COHORTS = {
    "inhouse": {"lt2mm": 37, "2to4mm": 122, "ge4mm": 105, "n": 264, "median_max_diameter_mm": 3.73},
    "adam90": {"lt2mm": 18, "2to4mm": 40, "ge4mm": 75, "n": 133, "median_max_diameter_mm": 4.22},
}


def bin_name(d: float) -> str:
    return "lt2mm" if d < 2 else ("2to4mm" if d < 4 else "ge4mm")


def lesion_diameters(mask: np.ndarray, affine: np.ndarray) -> list[float]:
    labeled, n = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    diameters = []
    for component_id in range(1, n + 1):
        voxels = np.argwhere(labeled == component_id)
        diameters.append(max_diameter_mm(voxels, affine))
    return diameters


def summarise(diameters: list[float]) -> dict:
    if not diameters:
        return {"n": 0}
    arr = np.asarray(diameters)
    bins = {"lt2mm": 0, "2to4mm": 0, "ge4mm": 0}
    for d in diameters:
        bins[bin_name(d)] += 1
    return {"n": len(arr), "median_max_diameter_mm": round(float(np.median(arr)), 3),
           "iqr_max_diameter_mm": [round(float(np.percentile(arr, 25)), 3),
                                    round(float(np.percentile(arr, 75)), 3)],
           **bins}


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    cases_by_id = {c["case_id"]: c for c in manifest["cases"]}

    mask_files = sorted((DATA_ROOT / "location_masks").glob("*.nii.gz"))
    all_diameters = []
    by_modality = {"mr": [], "ct": []}
    by_center = {"center1": [], "center2": [], "center4": [], "center5": []}

    for i, path in enumerate(mask_files):
        if i % 50 == 0:
            print(f"progress: {i}/{len(mask_files)}", flush=True)
        case_id = path.name.removesuffix(".nii.gz")
        case = cases_by_id.get(case_id)
        if case is None:
            continue
        img = nib.load(path)
        data = np.asarray(img.dataobj)
        if not data.any():
            continue
        diameters = lesion_diameters(data > 0, img.affine)
        all_diameters.extend(diameters)
        by_modality[case["modality"]].extend(diameters)
        by_center[case["center"]].extend(diameters)

    report = {
        "topaneu_overall": summarise(all_diameters),
        "topaneu_by_modality": {m: summarise(d) for m, d in by_modality.items()},
        "topaneu_by_center": {c: summarise(d) for c, d in by_center.items()},
        "known_cohorts": KNOWN_COHORTS,
    }

    out_json = OUT_DIR / "topaneu_lesion_size_distribution.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n")

    def row(name: str, s: dict) -> str:
        if s.get("n", 0) == 0:
            return f"| {name} | 0 | - | - | - | - |"
        return (f"| {name} | {s['n']} | {s['lt2mm']} | {s['2to4mm']} | {s['ge4mm']} | "
               f"{s['median_max_diameter_mm']:.2f} |")

    lines = ["# TopAneu lesion size distribution vs in-house/ADAM-90 (max diameter, same convention)",
            "", "All diameters are **max_diameter_mm** (longest pairwise extent), matching "
            "`new_aneurysms/src/eval/matching.py`'s `max_diameter_mm` exactly -- same function, "
            "not a re-derivation. NOT equivalent-sphere diameter.", "",
            "## Three-cohort comparison", "",
            "| cohort | n | <2mm | 2-4mm | >=4mm | median max-diameter (mm) |",
            "|---|---:|---:|---:|---:|---:|",
            row("in-house (known)", KNOWN_COHORTS["inhouse"] | {"lt2mm": KNOWN_COHORTS["inhouse"]["lt2mm"],
                "2to4mm": KNOWN_COHORTS["inhouse"]["2to4mm"], "ge4mm": KNOWN_COHORTS["inhouse"]["ge4mm"]}),
            row("ADAM-90 (known)", KNOWN_COHORTS["adam90"] | {"lt2mm": KNOWN_COHORTS["adam90"]["lt2mm"],
                "2to4mm": KNOWN_COHORTS["adam90"]["2to4mm"], "ge4mm": KNOWN_COHORTS["adam90"]["ge4mm"]}),
            row("TopAneu (all 417)", report["topaneu_overall"]),
            "", "## TopAneu by modality", "",
            "| modality | n | <2mm | 2-4mm | >=4mm | median max-diameter (mm) |",
            "|---|---:|---:|---:|---:|---:|",
            row("MRA", report["topaneu_by_modality"]["mr"]),
            row("CTA", report["topaneu_by_modality"]["ct"]),
            "", "## TopAneu by center", "",
            "| center | n | <2mm | 2-4mm | >=4mm | median max-diameter (mm) |",
            "|---|---:|---:|---:|---:|---:|",
    ]
    for center in ("center1", "center2", "center4", "center5"):
        lines.append(row(center, report["topaneu_by_center"][center]))

    overall = report["topaneu_overall"]
    if overall.get("n", 0):
        topaneu_median = overall["median_max_diameter_mm"]
        inhouse_median = KNOWN_COHORTS["inhouse"]["median_max_diameter_mm"]
        adam_median = KNOWN_COHORTS["adam90"]["median_max_diameter_mm"]
        verdict = ("TopAneu's lesions are close in size to in-house/ADAM (same size population)"
                  if abs(topaneu_median - inhouse_median) < 0.5 and abs(topaneu_median - adam_median) < 0.5
                  else "TopAneu's lesion size distribution differs materially from in-house/ADAM")
        lines += ["", f"**Verdict: {verdict}.** TopAneu median max-diameter {topaneu_median:.2f}mm vs "
                 f"in-house {inhouse_median:.2f}mm, ADAM-90 {adam_median:.2f}mm."]

    out_md = OUT_DIR / "topaneu_lesion_size_distribution.md"
    out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out_json} and {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
