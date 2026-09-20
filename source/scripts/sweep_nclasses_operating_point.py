"""Sweeps N-classes-per-scan (configuration B) against the real 6-metric score.

2026-08-14, peer-requested. Configuration A (top-K candidates) is dominated
by same-class redundancy: at K=50, many candidates map to the same common
class (ACom, M1-M2 junction, ...), wasting mask capacity and diluting
Dice/VS without helping Precision/Recall/MCC (that metric only needs ONE
hit per class). Configuration B instead keeps the single best-scoring
candidate PER PREDICTED CLASS, then keeps only the N highest-scoring
classes per scan -- N is directly interpretable ("draw this many classes"),
unlike a score threshold whose meaning is data-dependent.

N=all (no cap) is the recall/coverage CEILING -- already built separately
(build_e2e_geometric_pipeline.py --mode one-per-class --class-score-threshold 0,
no --max-classes-per-scan) since TopAneu averages only ~1.2 true classes per
scan, so drawing every geometrically-assignable class per scan is expected
to be a heavy net negative on MCC/Precision despite maximizing coverage --
this sweep is what finds where the real optimum sits between coverage and
precision.

Reports TWO curves per N, as requested: distinct-class coverage (fraction
of the held-out center's own distinct classes hit at least once) and the
mean-of-6-metrics summary score. Their peaks are expected to sit at
different N -- coverage favors large N, the summary score favors small N,
and the gap between them is the precision/MCC penalty for over-drawing.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

BOXES_DIR = "/home/jovyan/rtx4claude-datavol-1/new_aneurysms/artifacts/nndet_task020_zeroshot_topaneu_mr"
HELD_OUT_CENTER = "center5"
N_VALUES = (1, 2, 3, 5, 8, 12, 20)  # "all" is the separately-built threshold=0 reference point
PY = "/home/jovyan/rtx4claude-datavol-1/conda_envs/topcow_claim_official/bin/python"
SCRIPTS = "/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/scripts"
ARTIFACTS = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/artifacts")
REPORTS = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/reports")
LOCO_CEILING = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/reports/loco_coverage_ceiling.json")


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-3000:], flush=True)
        print(result.stderr[-3000:], flush=True)
        raise RuntimeError(f"command failed: {' '.join(cmd)}")


def distinct_class_coverage(predictions_dir: Path, held_out_center: str) -> float:
    """Fraction of the center's own distinct GT classes hit at least once by any prediction mask."""
    import nibabel as nib
    import numpy as np

    ceiling = json.loads(LOCO_CEILING.read_text())
    gt_classes = set(int(k) for k in ceiling["folds"][held_out_center]["class_counts"].keys())
    predicted_classes = set()
    for path in predictions_dir.glob("*.nii.gz"):
        arr = np.asarray(nib.load(path).dataobj)
        predicted_classes.update(int(c) for c in np.unique(arr) if c != 0)
    hit = gt_classes & predicted_classes
    return len(hit) / len(gt_classes) if gt_classes else 0.0


def main() -> int:
    results = {}
    for n in N_VALUES:
        print(f"\n=== N={n} ===", flush=True)
        masks_dir = ARTIFACTS / f"e2e_geometric_masks_n{n}"
        run([PY, f"{SCRIPTS}/build_e2e_geometric_pipeline.py",
            "--boxes-dir", BOXES_DIR, "--output-dir", str(masks_dir),
            "--held-out-center", HELD_OUT_CENTER, "--mode", "one-per-class",
            "--class-score-threshold", "0.0", "--max-classes-per-scan", str(n)])

        label = f"e2e_n{n}"
        run([PY, f"{SCRIPTS}/score_e2e_placeholder_on_loco.py",
            "--predictions-dir", str(masks_dir), "--held-out-center", HELD_OUT_CENTER,
            "--label", label])

        score_path = REPORTS / f"{label}_{HELD_OUT_CENTER}_score.json"
        overall = json.loads(score_path.read_text())["overall"]
        summary_score = sum(v for k2, v in overall.items() if k2 != "HD95") / 5 - overall["HD95"] / 5
        coverage = distinct_class_coverage(masks_dir, HELD_OUT_CENTER)
        results[n] = {**overall, "summary_score": summary_score, "distinct_class_coverage": coverage}
        print(f"N={n}: {overall}, summary={summary_score:.4f}, coverage={coverage:.4f}", flush=True)

    out_json = REPORTS / "nclasses_operating_point_sweep.json"
    out_json.write_text(json.dumps({"held_out_center": HELD_OUT_CENTER, "results_by_n": results},
                                   indent=2) + "\n")

    lines = ["# N-classes-per-scan operating point sweep (configuration B) -- center5", "",
            "| N | PRECISION | RECALL | MCC | DICE | VOLSIM | HD95 | summary | distinct-class coverage |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in N_VALUES:
        r = results[n]
        lines.append(f"| {n} | {r['PRECISION']:.4f} | {r['RECALL']:.4f} | {r['MCC']:.4f} | "
                     f"{r['DICE']:.4f} | {r['VOLSIM']:.4f} | {r['HD95']:.4f} | "
                     f"{r['summary_score']:.4f} | {r['distinct_class_coverage']:.1%} |")
    best_n_summary = max(results, key=lambda n: results[n]["summary_score"])
    best_n_coverage = max(results, key=lambda n: results[n]["distinct_class_coverage"])
    lines += ["", f"Best N by summary score: **{best_n_summary}**.",
             f"Best N by distinct-class coverage: **{best_n_coverage}**.",
             "The gap between these two N's is the precision/MCC penalty for over-drawing classes."]

    out_md = REPORTS / "nclasses_operating_point_sweep.md"
    out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out_json} and {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
