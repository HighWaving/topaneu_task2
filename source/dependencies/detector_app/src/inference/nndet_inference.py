"""nnDetection inference-time overrides, kept free of torch and nndet imports.

The candidate-ceiling experiment needs to change one postprocessing parameter
and to be able to test that change in the environment the test suite runs in,
which has neither torch nor nndet installed.  Nothing here may import them.
"""

from __future__ import annotations

#: nnDetection's own default, ``BoxEnsemblerSelective.get_default_parameters``.
DEFAULT_DETECTIONS_PER_IMAGE = 100
#: Defaults of the two pools that feed the cap, from the same place.
DEFAULT_POOL = 1000
#: How much room the pools get above the cap so NMS, not the pool, decides.
POOL_HEADROOM = 10


def detection_cap_override(max_detections: int) -> dict:
    """Ensembler parameters that lift the per-case box cap to ``max_detections``.

    A case comes back with at most ``model_detections_per_image`` boxes, applied
    after NMS, and that is the limit that truncates a case -- not the model's own
    ``detections_per_img``, which is per patch.  The two pre-NMS pools are lifted
    alongside it, otherwise ``model_topk``/``ensemble_topk`` would silently
    become the new ceiling and the measurement would describe them instead.

    The pools get an order of magnitude of headroom rather than matching the cap
    exactly: NMS runs between them and the cap, so a pool merely equal to the cap
    would still be the effective limit whenever NMS discards anything.

    ``BoxEnsembler.from_case`` merges this into the defaults, so keys left out
    here keep the values inference has always used.
    """
    if max_detections < 1:
        raise ValueError(f"max_detections must be positive, got {max_detections}")
    pool = max(POOL_HEADROOM * max_detections, DEFAULT_POOL)
    return {"model_detections_per_image": max_detections,
            "model_topk": pool,
            "ensemble_topk": pool}
