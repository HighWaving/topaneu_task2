"""Data-driven MRA vs CTA comparison: does TopAneu Task 2 need one model or two?

2026-08-14, user-directed. Compares intensity distribution, spacing
distribution, and lesion-size distribution between the 308 MRA and 109 CTA
cases. Zero GPU, deliberately lightweight (header-only spacing reads,
downsampled intensity sampling) to run alongside the higher-priority
diagnostic probes without competing heavily for I/O/CPU.

Verdict heuristic (not a hard rule, reported for a human/peer judgement
call): if intensity distributions barely overlap (as expected -- MRA and
CTA are fundamentally different physical quantities, Hounsfield units vs
arbitrary MR signal) that alone does not imply two models are needed if a
single network can learn a modality-conditioned normalisation. What matters
more for the one-vs-two-model decision is whether SPACING and LESION SIZE
distributions differ enough that a single spacing/patch-size choice would
be a bad compromise for one modality -- that's a genuine architecture-level
cost, not something normalisation fixes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

MANIFEST = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task1/artifacts/phase0/case_manifest.json")
OUT_DIR = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task1/artifacts/phase1b_mra_vs_cta")
INTENSITY_SAMPLE_VOXELS = 200_000  # random subsample per case, not full volume


def read_spacing(path: str) -> tuple[float, float, float]:
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.ReadImageInformation()
    return reader.GetSpacing()


def sample_intensities(path: str, rng: np.random.Generator) -> np.ndarray:
    arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
    flat = arr.ravel()
    if len(flat) > INTENSITY_SAMPLE_VOXELS:
        idx = rng.choice(len(flat), size=INTENSITY_SAMPLE_VOXELS, replace=False)
        flat = flat[idx]
    return flat.astype(np.float32)


def lesion_sizes_mm(location_mask_path: str, spacing: tuple[float, float, float]) -> list[float]:
    arr = sitk.GetArrayFromImage(sitk.ReadImage(location_mask_path))
    if not arr.any():
        return []
    labeled, n = ndimage.label(arr > 0, structure=np.ones((3, 3, 3), dtype=np.uint8))
    voxel_volume_mm3 = float(np.prod(spacing))
    sizes = []
    for component_id in range(1, n + 1):
        voxel_count = int((labeled == component_id).sum())
        volume_mm3 = voxel_count * voxel_volume_mm3
        equivalent_diameter = 2.0 * (3.0 * volume_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0)
        sizes.append(equivalent_diameter)
    return sizes


def summarise(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"n": 0}
    return {"n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
           "std": float(np.std(values)), "p10": float(np.percentile(values, 10)),
           "p90": float(np.percentile(values, 90))}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    rng = np.random.default_rng(20260814)

    by_modality: dict[str, dict[str, list]] = {
        "mr": {"spacing_inplane": [], "spacing_z": [], "intensities": [], "lesion_sizes_mm": []},
        "ct": {"spacing_inplane": [], "spacing_z": [], "intensities": [], "lesion_sizes_mm": []},
    }

    for i, case in enumerate(manifest["cases"]):
        if i % 50 == 0:
            print(f"progress: {i}/{len(manifest['cases'])}", flush=True)
        modality = case["modality"]
        bucket = by_modality[modality]

        spacing = read_spacing(case["image"])  # (x, y, z) SimpleITK convention
        bucket["spacing_inplane"].append(spacing[0])
        bucket["spacing_z"].append(spacing[2])

        sampled = sample_intensities(case["image"], rng)
        bucket["intensities"].append(sampled)

        sizes = lesion_sizes_mm(case["location_mask"], spacing)
        bucket["lesion_sizes_mm"].extend(sizes)

    results = {}
    for modality, bucket in by_modality.items():
        intensities = np.concatenate(bucket["intensities"]) if bucket["intensities"] else np.array([])
        results[modality] = {
            "n_cases": len(bucket["spacing_inplane"]),
            "spacing_inplane_mm": summarise(np.asarray(bucket["spacing_inplane"])),
            "spacing_z_mm": summarise(np.asarray(bucket["spacing_z"])),
            "intensity_sample": summarise(intensities),
            "lesion_equivalent_diameter_mm": summarise(np.asarray(bucket["lesion_sizes_mm"])),
        }

    out_json = OUT_DIR / "mra_vs_cta_comparison.json"
    out_json.write_text(json.dumps(results, indent=2, default=float))

    lines = ["# MRA vs CTA: distributional comparison for the one-vs-two-model decision", "",
            "Zero GPU. Intensity sampled (200k voxels/case, not full volume) since MRA/CTA "
            "intensities are fundamentally different physical quantities (arbitrary MR signal "
            "vs Hounsfield units) -- large intensity differences are EXPECTED and do not by "
            "themselves argue for two models (a modality-conditioned normalisation handles "
            "that). Spacing and lesion-size differences matter more for architecture choices "
            "(patch/spacing) that a single shared model would have to compromise on.", "",
            "| quantity | MRA (n=308) | CTA (n=109) |", "|---|---|---|"]
    for label, key in (("in-plane spacing (mm)", "spacing_inplane_mm"),
                       ("z spacing (mm)", "spacing_z_mm"),
                       ("intensity sample (raw units)", "intensity_sample"),
                       ("lesion equiv. diameter (mm)", "lesion_equivalent_diameter_mm")):
        mr = results["mr"][key]
        ct = results["ct"][key]
        if mr.get("n", 0) and ct.get("n", 0):
            lines.append(f"| {label} | mean {mr['mean']:.3g}, median {mr['median']:.3g}, "
                        f"std {mr['std']:.3g} (n={mr['n']}) | "
                        f"mean {ct['mean']:.3g}, median {ct['median']:.3g}, "
                        f"std {ct['std']:.3g} (n={ct['n']}) |")
        else:
            lines.append(f"| {label} | n={mr.get('n', 0)} | n={ct.get('n', 0)} |")

    args_json = out_json
    out_md = OUT_DIR / "mra_vs_cta_comparison.md"
    out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args_json} and {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
