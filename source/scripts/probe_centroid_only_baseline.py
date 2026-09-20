"""Experiment B (2026-08-15, peer-requested): position-only 52-class baseline,
NO vessel mask at all -- the one classification signal guaranteed available
at test time.

Motivation: every class-coverage number in this project so far
(`phase2b_geometric_attachment.py`, `geometric_classifier_coverage.md`, 82-94%)
uses the OFFICIAL GROUND-TRUTH vessel_masks to attach a lesion to a vessel
segment. The grand-challenge container gets a single angio volume at test
time -- no vessel mask, official or predicted. Any real submission has to
either (a) predict its own vessel segmentation first (a much harder,
never-measured problem: 36-way segment labeling at segment BOUNDARIES, not
binary vessel/background) or (b) skip vessel geometry entirely and classify
off something always available, like the lesion's raw spatial position.

This tests (b): does aneurysm location correlate with raw position strongly
enough on its own (Acom sits near the midline anteriorly, BA tip sits near
the midline posteriorly, etc.) to make a vessel model unnecessary? If this
baseline alone reaches 60-70% class coverage, the vessel-segmentation
dependency may not be worth building. If it's ~30%, the vessel model is
load-bearing and must be taken seriously.

Method: per-lesion centroid (voxel coords / that case's own array shape --
a coarse per-case FOV normalization, not a physical/atlas registration;
TopAneu's own format spec fixes LPS+ orientation across all 4 centers, so
axis meaning is at least consistent center-to-center, unlike the OLD
project's Lausanne-cohort RAS/LPS mismatch this session's plan flagged
elsewhere -- that finding does NOT apply to TopAneu's own release format).
k-NN in normalized-centroid space, LOCO split (train on 3 centers, test on
the 4th, matching every other fold convention in this project), k swept
over {1,3,5,7,9,15,21}, majority vote among the k nearest training lesions'
classes (ties broken toward the single nearest neighbor's class).

Unlike phase2b_geometric_attachment.py, "position"-group classes are NOT
excluded here -- position-along-a-segment (e.g. R-4.2 A1 vs R-4.3 A2) is
exactly the kind of distinction a raw-position method COULD resolve that a
vessel-SEGMENT method structurally cannot, so this is a fair, separate test
of that group, not an invalid one.
"""

from __future__ import annotations

import json
import re
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
OUT_DIR = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/reports")

CASE_RE = re.compile(r"^topaneu_(center\d)_(mr|ct)_")
K_VALUES = (1, 3, 5, 7, 9, 15, 21)
HIT_THRESHOLD = 0.5  # same convention as geometric_classifier_coverage.md


def load_labels(mapping_path: Path) -> dict[int, str]:
    labels = json.loads(mapping_path.read_text())["labels"]
    return {v: k for k, v in labels.items()}


def build_pool() -> list[dict]:
    """One record per lesion instance: center, class_id, class_name, normalized centroid."""
    location_names = load_labels(DATA_ROOT / "location_mapping.json")
    records = []
    files = sorted(f for f in LOCATION_MASKS.iterdir() if f.suffix == ".gz")
    for i, loc_file in enumerate(files):
        if i % 40 == 0:
            print(f"pool build progress: {i}/{len(files)}", flush=True)
        m = CASE_RE.match(loc_file.name)
        if not m:
            continue
        center, modality = m.group(1), m.group(2)
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(loc_file)))
        if not arr.any():
            continue
        shape = np.asarray(arr.shape, dtype=float)
        for cls in np.unique(arr):
            if cls == 0:
                continue
            cls_name = location_names.get(int(cls), f"class_{cls}")
            labeled, n_components = ndimage.label(
                arr == cls, structure=np.ones((3, 3, 3), dtype=np.uint8))
            for component_id in range(1, n_components + 1):
                centroid = np.asarray(ndimage.center_of_mass(labeled == component_id))
                records.append({
                    "case": loc_file.stem.replace(".nii", ""),
                    "center": center, "modality": modality,
                    "class_id": int(cls), "class_name": cls_name,
                    "centroid_norm": (centroid / shape).tolist(),
                })
    return records


def mirror_name(name: str) -> str | None:
    """L-/R- prefix flip, e.g. 'L-3.1 ICA infraclinoid C1-C5' -> 'R-3.1 ...'. None if unprefixed
    (the 4 unpaired midline classes -- 1.4 BA trunk, 1.5 VA-BA junction, 1.10 BA tip,
    4.1 Acom complex -- have no mirror)."""
    if name.startswith("L-"):
        return "R-" + name[2:]
    if name.startswith("R-"):
        return "L-" + name[2:]
    return None


TERRITORY_NAME = {1: "VB", 2: "PCA", 3: "ICA", 4: "ACA", 5: "MCA"}


def territory(name: str) -> int:
    """2026-08-15, peer-derived and independently verified against
    location_mapping.json (exact match, all 52 classes): the digit before the
    decimal point in the class's numeric prefix IS the vascular territory --
    1-17=VB, 18-21=PCA (posterior circulation, 21 classes), 22-35=ICA,
    36-44=ACA, 45-52=MCA (anterior circulation, 31 classes). Robust to the
    known 5.3-vs-5.4 labeling typo on classes 49-52 (all territory 5 either
    way) since only the pre-decimal digit is used, never the full token as a
    key."""
    tok = name.split(" ")[0].split("-")[-1]
    return int(tok.split(".")[0])


def classify_error(true: str, pred: str) -> str:
    """Three-tier error taxonomy (2026-08-15, peer-requested) -- each tier
    implies a different fix, so collapsing them into one "wrong" bucket
    would hide which one actually matters:
      territory  -- predicted vascular territory != true (MCA vs VB, say):
                    the method is fundamentally broken here, not just
                    imprecise; expected near 0, investigate if not.
      lr_mirror  -- same class, wrong side: the one error type affine
                    registration can concretely fix (also the exact error
                    type in TopAneu's own 08-14 relabeling of 6 cases --
                    even human annotators make this mistake).
      other_within_territory -- same territory, not a clean L/R flip
                    (typically same-side adjacent-segment confusion, e.g.
                    3.3->3.4): the expected-difficulty case, sets the
                    precision ceiling of a position-only method.
    """
    if territory(true) != territory(pred):
        return "territory"
    if pred == mirror_name(true):
        return "lr_mirror"
    return "other_within_territory"


def knn_predict(query: np.ndarray, train_pts: np.ndarray, train_labels: list[str], k: int) -> str:
    dists = np.linalg.norm(train_pts - query, axis=1)
    order = np.argsort(dists)[:k]
    votes = Counter(train_labels[i] for i in order)
    top_count = max(votes.values())
    tied = [lbl for lbl, c in votes.items() if c == top_count]
    if len(tied) == 1:
        return tied[0]
    # tie-break: nearest neighbor among the tied classes
    for i in order:
        if train_labels[i] in tied:
            return train_labels[i]
    return tied[0]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("building lesion-centroid pool from (corrected) location_masks...", flush=True)
    pool = build_pool()
    print(f"pool: {len(pool)} lesions across "
         f"{len({r['case'] for r in pool})} positive cases", flush=True)

    centers = sorted({r["center"] for r in pool})
    class_group = {name: grp for name, (grp, _) in LOCATION_TO_VESSEL.items()}

    report = {"k_values": list(K_VALUES), "hit_threshold": HIT_THRESHOLD,
             "n_lesions": len(pool), "centers": centers, "by_k": {}}

    # per (k, held_out_center) top-1 accuracy, and pooled per-class accuracy for coverage
    for k in K_VALUES:
        fold_results = {}
        # pooled across folds, keyed by class, for the coverage table
        per_class_tally = defaultdict(lambda: {"correct": 0, "total": 0})
        by_group_tally = defaultdict(lambda: {"correct": 0, "total": 0})
        error_tally = {"territory": 0, "lr_mirror": 0, "other_within_territory": 0}
        n_errors = 0

        for held_out in centers:
            train = [r for r in pool if r["center"] != held_out]
            test = [r for r in pool if r["center"] == held_out]
            if not train or not test:
                continue
            train_pts = np.asarray([r["centroid_norm"] for r in train])
            train_labels = [r["class_name"] for r in train]

            correct = 0
            for r in test:
                pred = knn_predict(np.asarray(r["centroid_norm"]), train_pts, train_labels, k)
                true = r["class_name"]
                hit = pred == true
                correct += hit
                per_class_tally[true]["total"] += 1
                per_class_tally[true]["correct"] += hit
                grp = class_group.get(true, "unmapped")
                by_group_tally[grp]["total"] += 1
                by_group_tally[grp]["correct"] += hit
                if not hit:
                    n_errors += 1
                    error_tally[classify_error(true, pred)] += 1

            fold_results[held_out] = {
                "n_test_lesions": len(test), "n_train_lesions": len(train),
                "top1_accuracy": round(correct / len(test), 4),
            }

        n_distinct = len(per_class_tally)
        n_identifiable = sum(1 for v in per_class_tally.values()
                             if v["total"] and v["correct"] / v["total"] >= HIT_THRESHOLD)
        coverage = n_identifiable / n_distinct if n_distinct else 0.0

        report["by_k"][k] = {
            "by_center": fold_results,
            "pooled_top1_accuracy": round(
                sum(v["correct"] for v in per_class_tally.values()) /
                sum(v["total"] for v in per_class_tally.values()), 4),
            "by_group_accuracy": {
                grp: {**v, "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else None}
                for grp, v in by_group_tally.items()},
            "n_distinct_classes_seen": n_distinct,
            "n_identifiable_classes": n_identifiable,
            "class_coverage": round(coverage, 4),
            "n_errors": n_errors,
            "error_taxonomy": {
                tier: {"n": n, "fraction": round(n / n_errors, 4) if n_errors else None}
                for tier, n in error_tally.items()},
            "per_class": {
                cls: {**v, "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else None}
                for cls, v in per_class_tally.items()},
        }

    out_json = OUT_DIR / "centroid_only_baseline.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n")

    lines = ["# Experiment B: position-only 52-class baseline (no vessel mask, k-NN on lesion centroid)",
            "", f"{len(pool)} lesion instances, LOCO split (train on 3 centers, test on the 4th), "
            "majority-vote k-NN on (voxel centroid / case's own array shape).", "",
            "**This is the one classification signal guaranteed available at test time** -- no "
            "ground-truth or predicted vessel mask used at all. Compare against "
            "`geometric_classifier_coverage.md`'s 82-94% (which uses the ORACLE ground-truth vessel "
            "mask, unavailable at test time) to gauge how much the vessel-segmentation dependency "
            "is actually worth.", "",
            "**These numbers are a FLOOR for position-only methods, not a ceiling.** Normalizing by "
            "each case's own array shape is the crudest possible spatial alignment -- it does not "
            "correct for head position/angle/scale differences within an image, only the image's "
            "own bounding box. A proper fix (affine-register every case to a common template, build "
            "the 52-class spatial prior in template space) would remove exactly the error this "
            "method can't -- see the 3-tier error breakdown below to gauge how much that's worth.", "",
            "## 3-tier error taxonomy (2026-08-15, peer-derived territory grouping, verified exact "
            "against `location_mapping.json`)", "",
            "Vascular territory = the digit before the decimal in a class's numeric prefix "
            "(1-17=VB, 18-21=PCA -> 21 posterior-circulation classes; 22-35=ICA, 36-44=ACA, "
            "45-52=MCA -> 31 anterior-circulation classes). Each tier implies a different fix, so "
            "they're kept separate rather than one 'wrong' bucket:", "",
            "- **territory**: predicted territory != true (e.g. MCA guessed as VB) -- the method is "
            "fundamentally broken here, not imprecise; expect near 0, investigate if not.",
            "- **lr_mirror**: same class, wrong side -- the one error type affine registration "
            "concretely fixes (also the exact error type in TopAneu's own 08-14 relabeling of 6 "
            "cases -- even human annotators make this mistake). Directly estimates what "
            "registration would recover.",
            "- **other_within_territory**: same territory, not a clean L/R flip (typically "
            "same-side adjacent-segment confusion, e.g. 3.3->3.4) -- the expected-difficulty case, "
            "sets this method's precision ceiling regardless of registration.", "",
            "| k | pooled top-1 accuracy | distinct classes identifiable (>=50% acc) | coverage | "
            "n errors | territory | lr_mirror | other_within_territory | "
            + " | ".join(f"{c} top-1" for c in centers) + " |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|" + "---:|" * len(centers)]
    for k in K_VALUES:
        r = report["by_k"][k]
        cols = " | ".join(f"{r['by_center'].get(c, {}).get('top1_accuracy', float('nan')):.1%}"
                          for c in centers)
        def fmt_tier(tier):
            t = r["error_taxonomy"][tier]
            return f"{t['n']} ({t['fraction']:.1%})" if t["fraction"] is not None else "n/a"
        lines.append(f"| {k} | {r['pooled_top1_accuracy']:.1%} | {r['n_identifiable_classes']}/"
                     f"{r['n_distinct_classes_seen']} | {r['class_coverage']:.1%} | {r['n_errors']} | "
                     f"{fmt_tier('territory')} | {fmt_tier('lr_mirror')} | "
                     f"{fmt_tier('other_within_territory')} | {cols} |")

    lines += ["", "## Accuracy by class group, best-coverage k", ""]
    best_k = max(K_VALUES, key=lambda k: report["by_k"][k]["class_coverage"])
    lines.append(f"Best k by coverage: **{best_k}**")
    lines.append("")
    lines.append("| group | n | accuracy |")
    lines.append("|---|---:|---:|")
    for grp, v in sorted(report["by_k"][best_k]["by_group_accuracy"].items()):
        acc = v["accuracy"]
        lines.append(f"| {grp} | {v['total']} | {acc:.1%} |" if acc is not None else f"| {grp} | {v['total']} | n/a |")

    out_md = OUT_DIR / "centroid_only_baseline.md"
    out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out_json} and {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
