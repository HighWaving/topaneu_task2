"""Route nnDetection's 3-D NMS through its CPU implementation.

nnDetection ships one CUDA extension, ``nndet._C``, and it exports exactly one
symbol: ``nms``.  Our copy of that extension is a prebuilt ``.so`` reused from a
build made on an earlier GPU generation, so on hardware whose compute capability
was not in that build it raises

    RuntimeError: CUDA error: no kernel image is available for execution on the device

Rebuilding the extension is the clean fix but it is a long, fragile CUDA-toolchain
job.  Since the extension has a single entry point and nnDetection already ships a
pure-torch CPU equivalent (``nms_cpu``, greedy IoU suppression -- the same
algorithm), routing that one call through the CPU costs a device transfer of a few
thousand boxes and leaves every convolution on the GPU.

``nms_cpu`` materialises an N x N IoU matrix, so cost grows quadratically with the
pre-NMS pool.  At the default pools (1000) that is trivial; at a lifted candidate
cap (pools 10000) it is roughly 400 MB and noticeably slower, which is a reason to
keep the cap modest rather than a reason not to do this.
"""

from __future__ import annotations


def cuda_nms_is_usable() -> bool:
    """Probe the CUDA NMS kernel on the current device.

    Returns False both when the extension is missing and when it is present but
    was built without this device's architecture -- the caller wants the same
    fallback either way.
    """
    import torch

    if not torch.cuda.is_available():
        return False
    try:
        from nndet._C import nms as nms_gpu
    except ImportError:
        return False

    boxes = torch.tensor([[0.0, 0.0, 0.0, 4.0, 4.0, 4.0],
                          [1.0, 1.0, 1.0, 5.0, 5.0, 5.0]], device="cuda")
    scores = torch.tensor([0.9, 0.1], device="cuda")
    try:
        nms_gpu(boxes.float(), scores.float(), 0.5)
        torch.cuda.synchronize()
    except RuntimeError:
        return False
    return True


def install_cpu_nms_fallback() -> None:
    """Replace the CUDA NMS symbol with one that computes on the CPU.

    ``nndet.core.boxes.nms.nms`` dispatches on ``boxes.is_cuda`` and reads
    ``nms_gpu`` as a module global, so rebinding that global is enough; every
    caller goes through it.  Indices come back on the caller's device so nothing
    downstream can tell the difference.
    """
    import importlib

    import torch

    # importlib, not ``from nndet.core.boxes import nms``: the package re-exports the
    # nms *function* under that name, so the plain import binds a function object and
    # rebinding an attribute on it is a silent no-op.
    nms_module = importlib.import_module("nndet.core.boxes.nms")

    def nms_gpu_via_cpu(boxes: "torch.Tensor", scores: "torch.Tensor",
                        thresh: float) -> "torch.Tensor":
        device = boxes.device
        keep = nms_module.nms_cpu(boxes.detach().float().cpu(),
                                  scores.detach().float().cpu(),
                                  thresh)
        return keep.to(device)

    nms_module.nms_gpu = nms_gpu_via_cpu
