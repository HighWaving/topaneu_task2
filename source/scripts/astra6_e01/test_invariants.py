from __future__ import annotations

import numpy as np

from .e01_common import N_CLASSES, feature_schema, label_mappings, lr_pair_map


def run() -> None:
    loc, ves = label_mappings()
    assert len(feature_schema(loc, ves)) == 295
    assert sorted(loc) == list(range(1, 53))
    assert sorted(ves) == list(range(1, 37))
    for mapping in (loc, ves):
        pair = lr_pair_map(mapping)
        assert all(pair[pair[k]] == k for k in pair)
    print("E01 static invariants passed")


if __name__ == "__main__":
    run()
