"""Geometric vessel-attachment methods for the 52-class location task -- ORACLE upper bound.

**This uses the official ground-truth vessel_masks, not a predicted vessel
segmentation.** It measures whether attachment geometry CAN identify the
correct vessel segment(s) when the vessel labels are perfect -- an upper
bound on what any real deployed pipeline (which must predict its own vessel
segmentation first) could achieve. Do not read any number here as a
deployable accuracy.

Three methods, each turning a lesion's own voxels + the vessel_masks array
into a predicted vessel-segment set, compared against
``phase2b_location_class_to_vessel_map.py``'s curated ground truth:

* C: nearest non-background vessel voxel to the lesion's centroid.
* B: dilate the lesion by k voxels, take the shell (dilated minus original),
  histogram non-background vessel labels in the shell. Decision rule reads
  the top-2 bins' share of the histogram: if the top bin dominates (>=70% of
  shell voxels, a tunable threshold), predict "single" (that segment); if
  the top two are comparably sized, predict "junction" (both segments).
* D: for every lesion surface voxel, nearest vessel voxel (single global
  distance-transform-based nearest-label lookup per case, not per voxel --
  ``scipy.ndimage.distance_transform_edt`` with ``return_indices`` gives the
  nearest background-free voxel for the whole volume in one call), then mode
  over all surface voxels' nearest labels.

"position" class group is excluded from correctness scoring here by
design -- vessel segment ID alone cannot distinguish e.g. R-4.2 A1 from
R-4.3 A2 (same segment R-A1A2), so counting these as failures would
conflate two different questions ("did we find the right vessel" vs "did we
find the right position along it", the latter needs a feature this script
doesn't build). Reported separately as "segment-level ceiling", not folded
into the top-1 headline number.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from phase2b_location_class_to_vessel_map import LOCATION_TO_VESSEL  # noqa: E402

DATA_ROOT = Path("/home/jovyan/rtx4claude-datavol-1/data_topaneu26")
LOCATION_MASKS = DATA_ROOT / "location_masks"
VESSEL_MASKS = DATA_ROOT / "vessel_masks"
OUT_DIR = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/artifacts/phase2b_geometric_attachment")

SHELL_DILATION_VOXELS = (1, 2, 3, 4, 5)
SHELL_DOMINANT_FRACTION = 0.70


def load_labels(mapping_path: Path) -> dict[int, str]:
    labels = json.loads(mapping_path.read_text())["labels"]
    return {v: k for k, v in labels.items()}


def _bbox(mask: np.ndarray, margin: int) -> tuple[slice, slice, slice]:
    coords = np.argwhere(mask)
    lo = np.maximum(coords.min(axis=0) - margin, 0)
    hi = np.minimum(coords.max(axis=0) + margin + 1, np.asarray(mask.shape))
    return tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))


def method_c_nearest_centroid(lesion_mask: np.ndarray, vessel_names: dict[int, str],
                              nearest_label_volume: np.ndarray) -> set[str]:
    """O(1) via the precomputed per-case nearest-label volume -- no per-lesion
    brute-force search over every vessel voxel (that was the real cost here).
    """
    centroid = np.rint(ndimage.center_of_mass(lesion_mask)).astype(int)
    centroid = np.clip(centroid, 0, np.asarray(nearest_label_volume.shape) - 1)
    label = int(nearest_label_volume[tuple(centroid)])
    if label == 0:
        return set()
    return {vessel_names.get(label, f"vessel_{label}")}


def method_b_shell_histogram(lesion_mask: np.ndarray, vessel_arr: np.ndarray,
                             vessel_names: dict[int, str], k: int) -> set[str]:
    """Cropped to the lesion's own local bounding box + k+1 margin -- dilation
    cost then scales with the lesion's own (small) size, not the full head
    volume. This is the fix for the ~5h-projected full-volume version.
    """
    box = _bbox(lesion_mask, margin=k + 1)
    local_lesion = lesion_mask[box]
    local_vessel = vessel_arr[box]
    dilated = ndimage.binary_dilation(local_lesion, iterations=k)
    shell = dilated & ~local_lesion
    labels_in_shell = local_vessel[shell]
    labels_in_shell = labels_in_shell[labels_in_shell > 0]
    if len(labels_in_shell) == 0:
        return set()
    counts = Counter(labels_in_shell.tolist())
    ranked = counts.most_common()
    total = sum(counts.values())
    top_label, top_count = ranked[0]
    if top_count / total >= SHELL_DOMINANT_FRACTION or len(ranked) == 1:
        return {vessel_names.get(top_label, f"vessel_{top_label}")}
    second_label, _ = ranked[1]
    return {vessel_names.get(top_label, f"vessel_{top_label}"),
           vessel_names.get(second_label, f"vessel_{second_label}")}


def method_d_surface_nearest_mode(lesion_mask: np.ndarray, vessel_names: dict[int, str],
                                  nearest_label_volume: np.ndarray) -> set[str]:
    """Erosion cropped to the lesion's own local bounding box (small); the
    nearest-label lookup itself is O(1) per surface voxel either way since
    ``nearest_label_volume`` is precomputed once per case.
    """
    box = _bbox(lesion_mask, margin=2)
    local_lesion = lesion_mask[box]
    eroded = ndimage.binary_erosion(local_lesion)
    surface = local_lesion & ~eroded
    if not surface.any():
        surface = local_lesion  # tiny lesion, no interior to erode
    labels = nearest_label_volume[box][surface]
    labels = labels[labels > 0]
    if len(labels) == 0:
        return set()
    mode_label = Counter(labels.tolist()).most_common(1)[0][0]
    return {vessel_names.get(mode_label, f"vessel_{mode_label}")}


def nearest_label_lookup(vessel_arr: np.ndarray) -> np.ndarray:
    """Precompute, once per case, the nearest-nonzero-label at every voxel."""
    background = vessel_arr == 0
    _, indices = ndimage.distance_transform_edt(background, return_indices=True)
    return vessel_arr[tuple(indices)]


def correct(predicted: set[str], truth: frozenset[str]) -> bool:
    """Predicted segment set overlaps the true set -- exact match not required
    for junction classes (finding either true segment counts), but a
    single-segment true class must be matched by that exact segment (a
    prediction of {A, B} when truth is {A} is not a clean single-segment hit).
    """
    if not predicted:
        return False
    if len(truth) == 1:
        return predicted == truth
    return bool(predicted & truth)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vessel-mask-dir", type=Path, default=VESSEL_MASKS,
                        help="2026-08-17, peer-directed: source of vessel segment masks. "
                             "Defaults to the official GT (data_topaneu26/vessel_masks/, the "
                             "oracle upper bound this script was originally built for). Point "
                             "this at a PREDICTED vessel-mask directory (e.g. TA36 output) to "
                             "measure real-pipeline classifier accuracy instead -- algorithm is "
                             "unchanged either way, this only swaps the input source.")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--label", default="",
                        help="output filename suffix, so a predicted-vessel run doesn't "
                             "overwrite the oracle GT run's report. Empty (default) keeps the "
                             "original unsuffixed filenames -- probe_geometric_classifier_coverage.py "
                             "and the rest of the existing rerun chain depend on that exact name.")
    args = parser.parse_args()

    vessel_mask_dir = args.vessel_mask_dir
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    location_names = load_labels(DATA_ROOT / "location_mapping.json")
    vessel_names = load_labels(DATA_ROOT / "vessel_mapping.json")

    location_files = sorted(f for f in LOCATION_MASKS.iterdir() if f.suffix == ".gz")

    # per method: correct / total, split by group
    tally = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    per_class_tally = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    class_counts = Counter()
    skipped_no_map = Counter()
    files_checked = 0

    for i, loc_file in enumerate(location_files):
        if i % 20 == 0:
            print(f"progress: {i}/{len(location_files)}", flush=True)
        vessel_file = vessel_mask_dir / loc_file.name
        if not vessel_file.is_file():
            continue
        loc_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(loc_file)))
        if not loc_arr.any():
            continue
        vessel_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(vessel_file)))
        if loc_arr.shape != vessel_arr.shape:
            continue
        files_checked += 1

        nearest_label_volume = nearest_label_lookup(vessel_arr)

        for cls in np.unique(loc_arr):
            if cls == 0:
                continue
            cls_name = location_names.get(int(cls), f"class_{cls}")
            if cls_name not in LOCATION_TO_VESSEL:
                skipped_no_map[cls_name] += 1
                continue
            group, truth_segments = LOCATION_TO_VESSEL[cls_name]

            labeled, n_components = ndimage.label(
                loc_arr == cls, structure=np.ones((3, 3, 3), dtype=np.uint8))
            for component_id in range(1, n_components + 1):
                lesion_mask = labeled == component_id
                class_counts[cls_name] += 1

                pred_c = method_c_nearest_centroid(lesion_mask, vessel_names, nearest_label_volume)
                pred_d = method_d_surface_nearest_mode(lesion_mask, vessel_names, nearest_label_volume)

                for method_name, pred in (("C_nearest_centroid", pred_c),
                                          ("D_surface_nearest_mode", pred_d)):
                    tally[method_name][group]["total"] += 1
                    per_class_tally[method_name][cls_name]["total"] += 1
                    if correct(pred, truth_segments):
                        tally[method_name][group]["correct"] += 1
                        per_class_tally[method_name][cls_name]["correct"] += 1

                for k in SHELL_DILATION_VOXELS:
                    method_name = f"B_shell_k{k}"
                    pred_b = method_b_shell_histogram(lesion_mask, vessel_arr, vessel_names, k)
                    tally[method_name][group]["total"] += 1
                    per_class_tally[method_name][cls_name]["total"] += 1
                    if correct(pred_b, truth_segments):
                        tally[method_name][group]["correct"] += 1
                        per_class_tally[method_name][cls_name]["correct"] += 1

    baseline_class = class_counts.most_common(1)[0]
    total_lesions = sum(class_counts.values())
    baseline_accuracy = baseline_class[1] / total_lesions

    payload = {
        "files_checked": files_checked,
        "total_lesions": total_lesions,
        "classes_seen": len(class_counts),
        "classes_skipped_no_mapping": dict(skipped_no_map),
        "always_guess_most_common_baseline": {
            "class": baseline_class[0], "n": baseline_class[1],
            "accuracy": round(baseline_accuracy, 4)},
        "by_method_by_group": {
            method: {group: {**v, "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else None}
                    for group, v in groups.items()}
            for method, groups in tally.items()},
        "per_class": {
            method: {cls: {**v, "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else None}
                    for cls, v in classes.items()}
            for method, classes in per_class_tally.items()},
    }
    suffix = f"_{args.label}" if args.label else ""
    out_json = out_dir / f"geometric_attachment_results{suffix}.json"
    out_json.write_text(json.dumps(payload, indent=2))

    is_oracle = vessel_mask_dir == VESSEL_MASKS
    lines = [
        f"# Geometric vessel-attachment methods for 52-class location -- "
        f"{args.label or 'ORACLE upper bound'}",
        "",
        ("**Uses the official ground-truth vessel_masks, not a predicted vessel "
         "segmentation. This is an upper bound on what a deployed pipeline "
         "(which must predict its own vessel segmentation first) could achieve, "
         "not a deployable number.**" if is_oracle else
         f"**Uses PREDICTED vessel masks from `{vessel_mask_dir}`, not ground truth. "
         "This is a real-pipeline number (still against oracle GT lesion positions), "
         "not the oracle upper bound -- compare against the oracle_gt report to see "
         "the headroom attributable to vessel-mask quality.**"),
        "",
        f"{files_checked} positive cases, {total_lesions} lesion instances, "
        f"{len(class_counts)}/52 classes observed.",
        f"Baseline (always guess most common class, `{baseline_class[0]}`, "
        f"n={baseline_class[1]}): **{baseline_accuracy:.1%}**.",
        "",
        "Group sizes from this session's curated mapping (`phase2b_location_class_to_vessel_map.py`) "
        "-- documented as 26 single/18 junction/8 position, not necessarily identical to any other "
        "session's split of the same ambiguous class names; check the mapping file's inline reasoning.",
        "",
        "## Top-1 accuracy by method and group ('position' group is a segment-level "
        "ceiling, not a fair top-1 score -- see module docstring)",
        "",
        "| method | single (n) | junction (n) | position segment-ceiling (n) | "
        "overall (single+junction only) |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in sorted(tally):
        groups = tally[method]
        single = groups.get("single", {"correct": 0, "total": 0})
        junction = groups.get("junction", {"correct": 0, "total": 0})
        position = groups.get("position", {"correct": 0, "total": 0})
        overall_correct = single["correct"] + junction["correct"]
        overall_total = single["total"] + junction["total"]
        overall_acc = overall_correct / overall_total if overall_total else float("nan")
        single_acc = single["correct"] / single["total"] if single["total"] else float("nan")
        junction_acc = junction["correct"] / junction["total"] if junction["total"] else float("nan")
        position_acc = position["correct"] / position["total"] if position["total"] else float("nan")
        lines.append(f"| {method} | {single_acc:.1%} ({single['total']}) | "
                    f"{junction_acc:.1%} ({junction['total']}) | "
                    f"{position_acc:.1%} ({position['total']}) | "
                    f"**{overall_acc:.1%}** ({overall_total}) |")

    lines += ["", "## Per-class accuracy, best method "
             f"(picked per class from the table above's overall winner)", "",
             "| class | group | n | best accuracy |", "|---|---|---:|---:|"]
    best_method = max(
        (m for m in tally if m != "always_guess"),
        key=lambda m: (tally[m].get("single", {"correct": 0})["correct"] +
                       tally[m].get("junction", {"correct": 0})["correct"]))
    for cls_name, (group, _) in sorted(LOCATION_TO_VESSEL.items(),
                                       key=lambda kv: (kv[1][0], kv[0])):
        v = per_class_tally[best_method].get(cls_name, {"correct": 0, "total": 0})
        if v["total"] == 0:
            continue
        acc = v["correct"] / v["total"]
        lines.append(f"| {cls_name} | {group} | {v['total']} | {acc:.1%} |")

    lines += ["", f"Best method overall: **{best_method}**."]

    out_md = out_dir / f"geometric_attachment_results{suffix}.md"
    out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n... wrote {out_json} and {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
