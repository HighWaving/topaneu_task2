"""Validates local_scoring_arena.py against the official reference outputs.

Reconstructs the "all_zero" and "all_correct" scenarios using REAL TopAneu
location_masks (not synthetic spheres like the official test.py -- that
needs data this session doesn't have reference GT filenames for), and
checks our wrapper's aggregate metrics land in the same place the official
README documents. This is checking the WRAPPER'S plumbing (load_gt
patching, function call correctness), not re-deriving the metrics math
itself -- that math is the official evaluate.py, untouched.

2026-08-14 correction: the overall (52-class-averaged) HD95 for all_zero is
NOT expected to be ~1.0 on a small real sample, even though the official
test_evaluations/outputs-all_zero.json fixture shows ~0.9999997 there.
Root cause (confirmed by reading evaluate.py directly, not guessed):
`hd_fn` (evaluate.py:108) correctly returns exactly 1.0 (normalized) for
any class where one of {gt, pred} is empty and the other isn't -- that
part of the wrapper is correct. But `evaluation_average` (evaluate.py:338-
346) divides the summed per-class HD95 unconditionally by N_CLASSES=52,
with NO weighting by how many cases/classes actually had GT presence.
Classes with zero GT presence anywhere in the sample get the explicit
true-negative HD95=0 default (evaluate.py's own "avoid biasing toward
TN classes" convention). The official reference fixture was generated
from a broad synthetic test set touching most/all 52 classes, where this
dilution is negligible. A 5-case real sample only touches a handful of
classes, so the 52-class average is dominated by classes that never
appear at all -- diluting ~1.0 down toward (n_classes_present / 52). This
is a sampling-size artifact of the *test*, not a defect in the wrapper or
in evaluate.py. Confirmed: 5-case run gave overall HD95=0.11538, and the
sample touched exactly 6 distinct classes across those 5 cases --
6/52=0.11538, matching to 5 decimal places.
The fix here: check per-class HD95 for classes that actually appear in
the sample's GT (the invariant that holds regardless of sample size),
instead of the confounded overall 52-class average.
For the all_zero scenario specifically, PRECISION/RECALL/MCC/DICE/VOLSIM
are NOT affected: a "present-in-GT-but-missed" class and a true-negative
class both naturally evaluate to 0 there (TP=0 either way), so no dilution
asymmetry exists for all_zero -- only HD95 special-cases an empty-vs-
nonempty pair to 1.0 instead of 0.0.

But the SAME dilution mechanism DOES hit the all_correct scenario, for
every metric, not just HD95: a perfectly-matched present class gives
PRECISION=RECALL=DICE=VOLSIM=1, while a true-negative class (absent
everywhere in the sample) gives TP=FP=FN=0 -> precision=recall~=0/eps~=0,
dice=volsim=0 explicitly. So on a sparse sample, overall PRECISION/RECALL/
DICE/VOLSIM for all_correct is ALSO diluted well below 1.0, not just HD95.
All scenario-2 checks below are therefore also done per-class, on classes
actually present in the sample's GT, not on the confounded overall average.
"""

import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent))
from local_scoring_arena import aggregate, score_case  # noqa: E402

MANIFEST = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task1/artifacts/phase0/case_manifest.json")


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    # restrict to a handful of positive cases (nonzero GT) for a fast check
    sample = []
    for case in manifest["cases"]:
        gt = sitk.GetArrayFromImage(sitk.ReadImage(case["location_mask"]))
        if gt.any():
            sample.append((case["case_id"], gt))
        if len(sample) >= 5:
            break

    print(f"validating against {len(sample)} real positive cases", flush=True)

    import time
    # Scenario 1: all-zero predictions (should match README: all metrics 0 except HD95~1)
    results_zero = []
    for case_id, gt in sample:
        t0 = time.time()
        results_zero.append(score_case(np.zeros_like(gt), case_id))
        print(f"  scored {case_id} (zero) in {time.time()-t0:.1f}s", flush=True)
    agg_zero = aggregate(results_zero)
    print("all_zero overall:", agg_zero["overall"])
    assert agg_zero["overall"]["PRECISION"] == 0.0
    assert agg_zero["overall"]["RECALL"] == 0.0
    assert agg_zero["overall"]["DICE"] == 0.0
    assert agg_zero["overall"]["VOLSIM"] == 0.0
    print("PASS: all_zero precision/recall/dice/volsim=0 (matches official README)")

    # Overall HD95 is NOT checked here -- it is confounded by how many of the
    # 52 classes the sample happens to touch (see module docstring). Instead
    # check the invariant that actually holds regardless of sample size: any
    # class present in this sample's GT must score exactly HD95=1.0 (worst
    # case, empty prediction vs nonempty GT).
    classes_present = sorted({
        cls for case_id, gt in sample for cls in np.unique(gt) if cls != 0
    })
    print(f"classes present across sample: {classes_present}")
    per_class = agg_zero["per_class"]
    for cls in classes_present:
        hd95_cls = per_class[f"HD95_{int(cls)}"]
        assert hd95_cls > 0.99, f"class {cls}: expected HD95~1.0 (empty pred vs present GT), got {hd95_cls}"
    print(f"PASS: all {len(classes_present)} GT-present classes score HD95~1.0 for all-zero predictions")

    # Scenario 2: perfect predictions (predictions == gt exactly)
    results_perfect = [score_case(gt.copy(), case_id) for case_id, gt in sample]
    agg_perfect = aggregate(results_perfect)
    print("all_correct overall (informational only, confounded by class coverage -- see docstring):",
          agg_perfect["overall"])

    per_class_perfect = agg_perfect["per_class"]
    for cls in classes_present:
        c = int(cls)
        precision, recall = per_class_perfect[f"PRECISION_{c}"], per_class_perfect[f"RECALL_{c}"]
        dice, volsim, hd95 = (per_class_perfect[f"DICE_{c}"], per_class_perfect[f"VOLSIM_{c}"],
                              per_class_perfect[f"HD95_{c}"])
        assert precision > 0.99, f"class {c}: expected PRECISION~1.0, got {precision}"
        assert recall > 0.99, f"class {c}: expected RECALL~1.0, got {recall}"
        assert dice > 0.99, f"class {c}: expected DICE~1.0, got {dice}"
        assert volsim > 0.99, f"class {c}: expected VOLSIM~1.0, got {volsim}"
        assert hd95 < 0.01, f"class {c}: expected HD95~0.0, got {hd95}"
    print(f"PASS: all {len(classes_present)} GT-present classes score "
         "precision/recall/dice/volsim~1.0, hd95~0.0 for perfect predictions")

    print("\nWrapper validated: local_scoring_arena.py reproduces the official evaluate.py's "
         "documented per-class behaviour on both extremes (all-zero and perfect predictions).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
