"""Empirically derives the location(52) -> vessel-segment(36) hierarchy from real training data.

Motivation: vessel_mapping.json defines 36 named vessel segments (BA, R-M1,
L-ICA-C6-C7, ...), a coarser anatomical level than the 52 fine location
classes. If each fine class nests inside exactly one vessel segment, that
segment is a natural, already-available (every scan has a vessel mask,
regardless of aneurysm presence) coarse label for a hierarchical
classification head: predict the well-populated 36-way vessel segment first
(cheap, reliable, always-supervised), then the fine 52-way class only among
the small number of candidates nested in that segment.

This does NOT assume the nesting from class names -- it measures it
directly: for every positive voxel in every location_masks file, look up
the vessel_masks value at the same voxel (both are in the same image grid
per the release format) and accumulate a location-class x vessel-segment
contingency table.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk

DATA_ROOT = Path("/home/jovyan/rtx4claude-datavol-1/data_topaneu26")
LOCATION_MASKS = DATA_ROOT / "location_masks"
VESSEL_MASKS = DATA_ROOT / "vessel_masks"
OUT_DIR = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task1/artifacts/phase2_hierarchy")


def load_labels(mapping_path: Path) -> dict[int, str]:
    labels = json.loads(mapping_path.read_text())["labels"]
    return {v: k for k, v in labels.items()}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    location_names = load_labels(DATA_ROOT / "location_mapping.json")
    vessel_names = load_labels(DATA_ROOT / "vessel_mapping.json")

    location_files = sorted(f for f in LOCATION_MASKS.iterdir() if f.suffix == ".gz")
    # contingency[location_class][vessel_segment] = voxel count
    contingency: dict[int, Counter] = defaultdict(Counter)
    files_checked = 0
    files_missing_vessel = []

    for i, loc_file in enumerate(location_files):
        if i % 20 == 0:
            print(f"progress: {i}/{len(location_files)}", flush=True)
        vessel_file = VESSEL_MASKS / loc_file.name
        if not vessel_file.is_file():
            files_missing_vessel.append(loc_file.name)
            continue
        loc_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(loc_file)))
        if not loc_arr.any():
            continue  # negative case, nothing to accumulate
        vessel_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(vessel_file)))
        if loc_arr.shape != vessel_arr.shape:
            files_missing_vessel.append(f"{loc_file.name} (shape mismatch)")
            continue
        files_checked += 1
        present_classes = np.unique(loc_arr)
        for cls in present_classes:
            if cls == 0:
                continue
            mask = loc_arr == cls
            vessel_values, counts = np.unique(vessel_arr[mask], return_counts=True)
            for v, c in zip(vessel_values, counts):
                contingency[int(cls)][int(v)] += int(c)

    # Build the hierarchy: for each location class, its dominant vessel
    # segment(s) by voxel count, and whether the nesting is clean (one
    # dominant segment) or ambiguous (split across multiple).
    hierarchy = {}
    ambiguous = []
    for cls, counter in sorted(contingency.items()):
        total = sum(counter.values())
        ranked = counter.most_common()
        dominant_vessel, dominant_count = ranked[0]
        purity = dominant_count / total if total else 0.0
        hierarchy[location_names.get(cls, f"class_{cls}")] = {
            "location_class_id": cls,
            "dominant_vessel_segment": vessel_names.get(dominant_vessel, f"vessel_{dominant_vessel}"),
            "dominant_vessel_id": dominant_vessel,
            "purity": round(purity, 4),
            "total_voxels": total,
            "all_vessel_segments": {vessel_names.get(v, f"vessel_{v}"): c for v, c in ranked},
        }
        if purity < 0.9:
            ambiguous.append(location_names.get(cls, f"class_{cls}"))

    # Reverse: for each vessel segment, which location classes nest in it
    # (using each class's dominant assignment) -- this is the actual
    # decision-tree structure a hierarchical head would use.
    by_vessel = defaultdict(list)
    for loc_name, info in hierarchy.items():
        by_vessel[info["dominant_vessel_segment"]].append(loc_name)

    payload = {
        "files_checked": files_checked,
        "files_missing_vessel_or_mismatched": files_missing_vessel,
        "location_classes_with_any_data": len(hierarchy),
        "location_classes_total": 52,
        "ambiguous_classes_purity_below_0.9": ambiguous,
        "hierarchy_by_location_class": hierarchy,
        "hierarchy_by_vessel_segment": {k: sorted(v) for k, v in sorted(by_vessel.items())},
    }
    out_json = OUT_DIR / "location_vessel_hierarchy.json"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Location(52) -> vessel-segment(36) hierarchy, empirically derived",
        "",
        f"Checked {files_checked} positive cases with both masks present "
        f"({len(files_missing_vessel)} missing/mismatched, see JSON). "
        f"{len(hierarchy)}/52 location classes have at least one training voxel "
        f"(matches the {payload['location_classes_total'] - len(hierarchy)} zero-instance "
        "classes already known from the census).",
        "",
        f"**{len(ambiguous)} classes have purity < 0.9** (their voxels split across more "
        "than one vessel segment, not cleanly nested) -- listed below, not silently averaged over.",
        "",
        "| location class | dominant vessel segment | purity | total voxels |",
        "|---|---|---:|---:|",
    ]
    for loc_name, info in sorted(hierarchy.items(), key=lambda kv: kv[1]["location_class_id"]):
        flag = " ⚠" if info["purity"] < 0.9 else ""
        lines.append(f"| {loc_name} | {info['dominant_vessel_segment']}{flag} | "
                    f"{info['purity']:.3f} | {info['total_voxels']} |")

    lines += ["", "## By vessel segment (the actual coarse->fine decision tree)", "",
             "| vessel segment | nested location classes |", "|---|---|"]
    for vessel, locs in sorted(by_vessel.items()):
        lines.append(f"| {vessel} | {', '.join(locs)} |")

    out_md = OUT_DIR / "location_vessel_hierarchy.md"
    out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))
    print(f"\n... wrote {out_json} and {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
