"""Vessel-based 52-class location classifier for the real (non-oracle-lesion)
end-to-end pipeline. 2026-08-17, peer-directed CT/MR honest-baseline work.

`phase2b_geometric_attachment.py` only measures whether attachment geometry
CAN recover a lesion's KNOWN true vessel segment(s) -- it's a correctness
checker, not a predictor (it never turns a vessel-segment observation into a
novel class GUESS). This module builds the missing forward direction: given
a detected lesion's position and a vessel-segment mask (oracle GT or TA36-
predicted), pick one of the 52 location classes.

Method: reuses `method_c_nearest_centroid` from `phase2b_geometric_attachment.py`
UNCHANGED (the winning method in that script's own oracle comparison,
`geometric_attachment_results.md` -- "Best method overall: C_nearest_centroid").
Turns its single predicted vessel segment into a class via an INVERSE of
`LOCATION_TO_VESSEL` (segment -> classes that include it). All 52 classes
have a vessel-segment entry (26 single + 18 junction + 8 position), so the
inverse map has full coverage; ambiguity is real, not a bug -- e.g. segment
"R-A1A2" is shared by both "R-4.2 A1" and "R-4.3 A2" (documented in
`phase2b_geometric_attachment.py`'s own docstring as structurally
indistinguishable by vessel geometry alone). Ties broken toward the smallest
segment-set class (prefer "single" over "junction" identity for a
single-segment observation) then toward global training-population
frequency (same "most common" convention used elsewhere in this codebase).
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from phase2b_location_class_to_vessel_map import LOCATION_TO_VESSEL  # noqa: E402
from phase2b_geometric_attachment import (  # noqa: E402
    method_c_nearest_centroid, nearest_label_lookup,
)

DATA_ROOT = Path("/home/jovyan/rtx4claude-datavol-1/data_topaneu26")


def load_labels(mapping_path: Path) -> dict[int, str]:
    labels = json.loads(mapping_path.read_text())["labels"]
    return {v: k for k, v in labels.items()}


def build_inverse_segment_map() -> dict[str, list[tuple[str, int]]]:
    """segment -> [(class_name, len(truth_segments)), ...]"""
    inverse = defaultdict(list)
    for name, (_, segs) in LOCATION_TO_VESSEL.items():
        for seg in segs:
            inverse[seg].append((name, len(segs)))
    return dict(inverse)


class VesselGeometricClassifier:
    def __init__(self, exclude_centers: set[str] = frozenset()):
        self.location_names = load_labels(DATA_ROOT / "location_mapping.json")
        self.vessel_names = load_labels(DATA_ROOT / "vessel_mapping.json")
        self.class_id_by_name = {v: k for k, v in self.location_names.items()}
        self.inverse_segment_map = build_inverse_segment_map()

        # Global class frequency for tie-breaking, from location_masks
        # excluding the held-out center(s) -- same LOCO discipline as
        # location_classifier_m1.py, not fitted on the eval set.
        import re
        case_re = __import__("re").compile(r"^topaneu_(center\d)_(mr|ct)_")
        freq = Counter()
        for f in sorted((DATA_ROOT / "location_masks").iterdir()):
            if f.suffix != ".gz":
                continue
            m = case_re.match(f.name)
            if not m or m.group(1) in exclude_centers:
                continue
            import nibabel as nib
            arr = nib.load(str(f)).get_fdata()
            for cls in np.unique(arr):
                if cls == 0:
                    continue
                freq[self.location_names.get(int(cls), f"class_{int(cls)}")] += 1
        self.class_frequency = freq
        self.most_common_class = freq.most_common(1)[0][0] if freq else None

    def _resolve_segment_to_class(self, segment: str) -> str | None:
        candidates = self.inverse_segment_map.get(segment)
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0][0]
        min_len = min(n for _, n in candidates)
        tied = [name for name, n in candidates if n == min_len]
        if len(tied) == 1:
            return tied[0]
        return max(tied, key=lambda name: self.class_frequency.get(name, 0))

    def classify_from_nearest_label_volume(self, lesion_mask: np.ndarray,
                                           nearest_label_volume: np.ndarray) -> int | None:
        """Cheap per-lesion classification given an ALREADY-COMPUTED
        nearest_label_volume (see `nearest_label_lookup`, re-exported below --
        an O(whole-volume) distance transform, must be computed ONCE per case
        and reused across every lesion/box in that case, never per-lesion;
        2026-08-17, peer-caught: the original `classify_lesion_mask` recomputed
        it on every call, ~20x redundant per case)."""
        segments = method_c_nearest_centroid(lesion_mask, self.vessel_names, nearest_label_volume)
        if not segments:
            return self.class_id_by_name.get(self.most_common_class) if self.most_common_class else None
        segment = next(iter(segments))
        cls_name = self._resolve_segment_to_class(segment)
        if cls_name is None:
            return self.class_id_by_name.get(self.most_common_class) if self.most_common_class else None
        return self.class_id_by_name[cls_name]

    def classify_lesion_mask(self, lesion_mask: np.ndarray, vessel_arr: np.ndarray) -> int | None:
        """Convenience wrapper for single-lesion/one-off use -- computes
        nearest_label_lookup internally. For multiple lesions in the same
        case, compute `nearest_label_lookup(vessel_arr)` once and call
        `classify_from_nearest_label_volume` instead (see above)."""
        nearest_label_volume = nearest_label_lookup(vessel_arr)
        return self.classify_from_nearest_label_volume(lesion_mask, nearest_label_volume)
