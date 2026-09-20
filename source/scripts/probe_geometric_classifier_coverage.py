"""Geometric-rule classification head: per-fold DISTINCT-CLASS coverage, not top-1 accuracy.

2026-08-14, peer correction. My first plan was to report a lesion-weighted
top-1 accuracy for the geometric attachment head. Peer correctly flagged
that as the same mistake as the modal-class placeholder: TopAneu's
evaluation_average() is unweighted over 52 classes with any-overlap credit,
so what matters is HOW MANY DISTINCT CLASSES get identified at all, not how
many individual lesions get classified correctly. A classifier that is
right on 60% of lesions but only spans 5 common classes scores far worse
(~5/52) than one right on 45% of lesions spread across 20 classes (~20/52).

This reuses `phase2b_geometric_attachment.py`'s ALREADY-COMPUTED per-class
accuracy table (`geometric_attachment_results.json`, pooled across all 417
cases -- that script tests, for lesions of a KNOWN true class, whether
geometric vessel-attachment correctly identifies THAT class's designated
vessel segment; see its own docstring for why "position"-group classes are
reported separately as a segment-level ceiling, not folded into top-1).

For each LOCO fold, joins that per-class accuracy against the fold's own
held-out class list (from `loco_coverage_ceiling.json`) and reports:
  - class coverage = classes the geometric method gets right often enough
    to plausibly identify (accuracy >= HIT_THRESHOLD) / classes actually
    present in that fold's GT
  - the SAME coverage broken down by instance-count bucket (=1, =2, 3-9,
    >=10) -- this is the actual test the peer wants: if geometric hit
    rate in the "=1 instance" bucket is comparable to the ">=10" bucket,
    geometry genuinely does not need training signal and the case for
    this whole approach holds. If it collapses on the rare buckets too,
    the advantage is illusory and must be reported as such, not smoothed
    over.
  - the full per-class table (52 rows, not just a headline number)
"single"/"junction" groups only -- "position" is excluded per the upstream
script's own documented reasoning (vessel segment ID can't resolve it).
"""

from __future__ import annotations

import json
from pathlib import Path

LOCATION_MAPPING = Path("/home/jovyan/rtx4claude-datavol-1/data_topaneu26/location_mapping.json")
GEOMETRIC_RESULTS = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/artifacts/"
                         "phase2b_geometric_attachment/geometric_attachment_results.json")
LOCO_CEILING = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/reports/loco_coverage_ceiling.json")
OUT_DIR = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/reports")

HIT_THRESHOLD = 0.5  # geometric method's per-class accuracy needed to call that class "identifiable"


def instance_bucket(n: int) -> str:
    if n == 1:
        return "=1"
    if n == 2:
        return "=2"
    if n <= 9:
        return "3-9"
    return ">=10"


BUCKET_ORDER = ["=1", "=2", "3-9", ">=10"]


def main() -> int:
    id_to_name = {v: k for k, v in json.loads(LOCATION_MAPPING.read_text())["labels"].items() if k != "background"}
    geo = json.loads(GEOMETRIC_RESULTS.read_text())
    ceiling = json.loads(LOCO_CEILING.read_text())

    best_method = max(
        geo["by_method_by_group"],
        key=lambda m: (geo["by_method_by_group"][m].get("single", {}).get("correct", 0) +
                       geo["by_method_by_group"][m].get("junction", {}).get("correct", 0)))
    per_class = geo["per_class"][best_method]
    print(f"using best method: {best_method}", flush=True)

    report = {"best_method": best_method, "hit_threshold": HIT_THRESHOLD, "folds": {}}
    lines = ["# Geometric classification head: per-fold distinct-class coverage (not top-1 accuracy)", "",
            f"Best geometric method: **{best_method}** (from `phase2b_geometric_attachment.py`'s ORACLE "
            "run using ground-truth vessel masks -- upper bound, not deployable). Coverage below counts "
            f"a class as 'geometrically identifiable' if that method's per-class accuracy >= {HIT_THRESHOLD:.0%} "
            "on the FULL 417-case pool (not fold-specific accuracy -- too few instances per fold to "
            "estimate per-class accuracy reliably within a single fold, so this asks 'is this class "
            "identifiable at all, pooled over all data' and then checks how much of each fold's ceiling "
            "that reaches). 'position'-group classes excluded -- vessel segment alone can't resolve them "
            "(see phase2b_geometric_attachment.py docstring).", "",
            "| held-out center | distinct classes in GT | geometrically identifiable | coverage | "
            "=1-instance | =2-instance | 3-9-instance | >=10-instance |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]

    for center, fold in ceiling["folds"].items():
        class_counts = fold["class_counts"]  # {class_id_str: n_instances}
        identifiable = []
        not_identifiable = []
        excluded_position_or_unmapped = []
        for cls_id_str, n_instances in class_counts.items():
            cls_id = int(cls_id_str)
            cls_name = id_to_name.get(cls_id)
            row = per_class.get(cls_name) if cls_name else None
            if row is None or row.get("total", 0) == 0:
                excluded_position_or_unmapped.append({"class_id": cls_id, "name": cls_name, "n": n_instances})
                continue
            acc = row["accuracy"]
            entry = {"class_id": cls_id, "name": cls_name, "n": n_instances, "geo_accuracy": acc}
            (identifiable if acc >= HIT_THRESHOLD else not_identifiable).append(entry)

        n_distinct = len(class_counts)
        n_identifiable = len(identifiable)
        coverage = n_identifiable / n_distinct if n_distinct else 0.0

        scored = identifiable + not_identifiable
        bucket_stats = {}
        for bucket in BUCKET_ORDER:
            total_in_bucket = [e for e in scored if instance_bucket(e["n"]) == bucket]
            hit_in_bucket = [e for e in identifiable if instance_bucket(e["n"]) == bucket]
            bucket_stats[bucket] = {
                "identifiable": len(hit_in_bucket), "total": len(total_in_bucket),
                "rate": round(len(hit_in_bucket) / len(total_in_bucket), 4) if total_in_bucket else None,
            }

        report["folds"][center] = {
            "n_distinct_classes": n_distinct,
            "n_identifiable": n_identifiable,
            "coverage": round(coverage, 4),
            "by_instance_bucket": bucket_stats,
            "identifiable_classes": identifiable,
            "not_identifiable_classes": not_identifiable,
            "excluded_position_or_unmapped": excluded_position_or_unmapped,
        }

        def fmt_bucket(b):
            s = bucket_stats[b]
            return f"{s['identifiable']}/{s['total']}" if s["total"] else "n/a"

        lines.append(f"| {center} | {n_distinct} | {n_identifiable} | {coverage:.1%} | "
                     f"{fmt_bucket('=1')} | {fmt_bucket('=2')} | {fmt_bucket('3-9')} | {fmt_bucket('>=10')} |")

    # Pooled across all 4 folds' (class, instance-count) pairs -- the decisive
    # test: does the geometric hit rate hold up in the rare buckets, or does
    # it collapse there just like a learned classifier would?
    pooled_bucket = {b: {"identifiable": 0, "total": 0} for b in BUCKET_ORDER}
    for fold in report["folds"].values():
        for b, stats in fold["by_instance_bucket"].items():
            pooled_bucket[b]["identifiable"] += stats["identifiable"]
            pooled_bucket[b]["total"] += stats["total"]
    for b in pooled_bucket:
        t = pooled_bucket[b]["total"]
        pooled_bucket[b]["rate"] = round(pooled_bucket[b]["identifiable"] / t, 4) if t else None
    report["pooled_by_instance_bucket"] = pooled_bucket

    out_json = OUT_DIR / "geometric_classifier_coverage.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n")

    lines += ["", "## Decisive test: does geometric hit rate hold up on rare classes, pooled across all 4 folds",
             "", "If the `=1` bucket's rate is comparable to the `>=10` bucket's rate, geometry genuinely "
             "does not need training signal and the case for this approach holds. If it collapses on the "
             "rare buckets too, the advantage is illusory.", "",
             "| instance bucket | identifiable | total | rate |", "|---|---:|---:|---:|"]
    for b in BUCKET_ORDER:
        s = pooled_bucket[b]
        rate_str = f"{s['rate']:.1%}" if s["rate"] is not None else "n/a"
        lines.append(f"| {b} | {s['identifiable']} | {s['total']} | {rate_str} |")

    lines += ["", "## Full per-class table, all 4 folds (52 rows, not a headline number)", "",
             "| held-out center | class | n instances | bucket | geo accuracy | identifiable (>=50%)? |",
             "|---|---|---:|---|---:|---|"]
    for center, fold in report["folds"].items():
        all_entries = sorted(fold["identifiable_classes"] + fold["not_identifiable_classes"],
                             key=lambda e: -e["n"])
        for e in all_entries:
            hit = "yes" if e in fold["identifiable_classes"] else "no"
            lines.append(f"| {center} | {e['name']} | {e['n']} | {instance_bucket(e['n'])} | "
                         f"{e['geo_accuracy']:.1%} | {hit} |")

    lines += ["", "## Excluded (position-group or no vessel mapping) per fold -- not counted either way", ""]
    for center, fold in report["folds"].items():
        exc = fold["excluded_position_or_unmapped"]
        if exc:
            lines.append(f"**{center}**: " + ", ".join(f"{e['name']}(n={e['n']})" for e in exc))
    lines.append("")

    out_md = OUT_DIR / "geometric_classifier_coverage.md"
    out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out_json} and {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
