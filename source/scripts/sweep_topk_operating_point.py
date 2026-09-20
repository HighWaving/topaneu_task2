"""Sweeps top-K candidates/scan against the REAL 6-metric TopAneu score, using existing boxes.

2026-08-14, peer-requested. Answers three questions from one sweep, using
already-computed zero-shot boxes.pkl (no new detector inference):
  1. Candidate ceiling: recall at the largest K (~200, "keep everything") is
     the detector's actual achievable ceiling on this held-out center.
  2. Best operating point: the K with the highest rank-averaged score across
     the 6 official metrics.
  3. Detection vs classification as the real bottleneck: if ceiling-K's
     recall approaches the oracle geometric-attachment number (21/24=87.5%
     at GT lesion locations), the gap is a threshold problem (free to fix by
     loosening K); if it stays far below even at max K, detection itself is
     short and needs a better-trained model, not a looser threshold.

No score floor -- pure per-scan rank cutoff (build_e2e_geometric_pipeline.py's
SCORE_THRESHOLD=0.0), so each K tests exactly "keep this many candidates per
scan," not a confounded score-cutoff question.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BOXES_DIR = "/home/jovyan/rtx4claude-datavol-1/new_aneurysms/artifacts/nndet_task020_zeroshot_topaneu_mr"
HELD_OUT_CENTER = "center5"
K_VALUES = (1, 3, 5, 10, 20, 50, 200)
PY = "/home/jovyan/rtx4claude-datavol-1/conda_envs/topcow_claim_official/bin/python"
SCRIPTS = "/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/scripts"
ARTIFACTS = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/artifacts")
REPORTS = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/reports")

# Reference ceiling from oracle geometric attachment at TRUE lesion locations
# (loco_coverage_ceiling.py / geometric_classifier_coverage.json): 21/24 = 87.5%
# distinct-class coverage for center5. Not directly the same statistic as
# per-class official RECALL, but the right ballpark reference for "does the
# gap close at high K."
ORACLE_LOCATION_COVERAGE_CENTER5 = 21 / 24


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-3000:], flush=True)
        print(result.stderr[-3000:], flush=True)
        raise RuntimeError(f"command failed: {' '.join(cmd)}")


def main() -> int:
    results = {}
    for k in K_VALUES:
        print(f"\n=== K={k} ===", flush=True)
        masks_dir = ARTIFACTS / f"e2e_geometric_masks_topk{k}"
        run([PY, f"{SCRIPTS}/build_e2e_geometric_pipeline.py",
            "--boxes-dir", BOXES_DIR, "--output-dir", str(masks_dir),
            "--held-out-center", HELD_OUT_CENTER, "--top-k", str(k)])

        label = f"e2e_topk{k}"
        run([PY, f"{SCRIPTS}/score_e2e_placeholder_on_loco.py",
            "--predictions-dir", str(masks_dir), "--held-out-center", HELD_OUT_CENTER,
            "--label", label])

        score_path = REPORTS / f"{label}_{HELD_OUT_CENTER}_score.json"
        overall = json.loads(score_path.read_text())["overall"]
        # rank-average as the official ranking does: average of the 6 metrics
        # (HD95 is already oriented so lower is better in evaluate.py's own
        # convention -- but for a SINGLE-arm summary here we just report raw
        # values; a proper rank-average needs multiple arms to rank against.
        # This mean is a simple summary proxy, not the competition's own
        # cross-submission rank-average.)
        metric_mean = sum(v for k2, v in overall.items() if k2 != "HD95") / 5 - overall["HD95"] / 5
        results[k] = {**overall, "summary_score": metric_mean}
        print(f"K={k}: {overall}, summary_score={metric_mean:.4f}", flush=True)

    out_json = REPORTS / "topk_operating_point_sweep.json"
    out_json.write_text(json.dumps({
        "held_out_center": HELD_OUT_CENTER,
        "oracle_location_coverage_reference": ORACLE_LOCATION_COVERAGE_CENTER5,
        "results_by_k": results,
    }, indent=2) + "\n")

    lines = ["# Top-K operating point sweep -- center5, zero-shot Task020FG boxes", "",
            f"Oracle geometric-attachment coverage at TRUE lesion locations (reference "
            f"ceiling, not the same statistic): {ORACLE_LOCATION_COVERAGE_CENTER5:.1%}", "",
            "| K | PRECISION | RECALL | MCC | DICE | VOLSIM | HD95 | summary (5 metrics - HD95)/5 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in K_VALUES:
        r = results[k]
        lines.append(f"| {k} | {r['PRECISION']:.4f} | {r['RECALL']:.4f} | {r['MCC']:.4f} | "
                     f"{r['DICE']:.4f} | {r['VOLSIM']:.4f} | {r['HD95']:.4f} | "
                     f"{r['summary_score']:.4f} |")
    best_k = max(results, key=lambda k: results[k]["summary_score"])
    ceiling_recall = results[K_VALUES[-1]]["RECALL"]
    lines += ["", f"Best K by summary score: **{best_k}**.",
             f"Recall at K={K_VALUES[-1]} (candidate ceiling): **{ceiling_recall:.1%}** "
             f"vs oracle location-based reference {ORACLE_LOCATION_COVERAGE_CENTER5:.1%}."]

    out_md = REPORTS / "topk_operating_point_sweep.md"
    out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out_json} and {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
