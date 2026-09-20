"""Run a frozen nnDetection checkpoint, or a fold ensemble, on preprocessed ADAM scans.

This intentionally bypasses ``nndet_predict`` because that entry point loads
all checkpoints in a fold directory.  A live training directory contains both
``model_best.ckpt`` and ``model_last.ckpt``; ensembling those would not answer
the intended question about the fold-1 best checkpoint.

``--ensemble-from`` is the deliberate exception: it takes one training directory
per fold, checks that they were planned identically, and stages exactly one
``model_best.ckpt`` from each into a private snapshot directory before handing
that to nnDetection's own multi-model predictor.  Ensembling happens inside the
predictor, before NMS, which is what actually suppresses false positives -- not
a post-hoc union of five box lists.  Staging is what keeps the guarantee above:
the snapshot holds five checkpoints because five were asked for, and nothing
else can leak in from a live training directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
from functools import partial
from pathlib import Path

import torch
from omegaconf import OmegaConf

from nndet.inference.helper import predict_dir
from nndet.inference.loading import load_all_models, load_final_model
from nndet.io.load import load_pickle

from src.inference.cpu_nms_fallback import cuda_nms_is_usable, install_cpu_nms_fallback
from src.inference.nndet_inference import detection_cap_override


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, required=True,
                        help="single-model source, and the config/plan source when "
                             "--ensemble-from is given")
    parser.add_argument("--ensemble-from", type=Path, action="append", default=None,
                        help="repeat once per fold training directory to ensemble their "
                             "model_best checkpoints inside the predictor")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best", choices=("best", "last"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-tta", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--max-detections", type=int, default=None,
                        help="lift the ensembler's per-case box cap; the default of 100 "
                             "is what truncates a case, not the per-patch head limit")
    parser.add_argument("--inference-plan", type=Path, default=None,
                        help="JSON holding a swept inference plan, whose numeric entries are "
                             "merged into the ensembler parameters. --max-detections still "
                             "wins for the keys it sets")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_ensemble_snapshot(fold_dirs: list[Path], checkpoint: str, plan_source: Path,
                            snapshot: Path) -> list[dict]:
    """Copy one checkpoint per fold into ``snapshot`` and describe what was staged.

    nnDetection's ``load_all_models`` globs ``*.ckpt``, so the snapshot must
    contain the ensemble and nothing else. Folds that were resumed carry extra
    ``model_best-v1.ckpt`` files next to ``model_best.ckpt``; naming the copies
    per fold is what keeps those out and makes the staged set auditable.
    """
    reference_plan = hashlib.md5((plan_source / "plan.pkl").read_bytes()).hexdigest()
    snapshot.mkdir(parents=True, exist_ok=True)
    staged = []
    for fold_dir in fold_dirs:
        source = fold_dir / f"model_{checkpoint}.ckpt"
        if not source.is_file():
            raise FileNotFoundError(source)
        plan_digest = hashlib.md5((fold_dir / "plan.pkl").read_bytes()).hexdigest()
        if plan_digest != reference_plan:
            raise ValueError(f"{fold_dir} was planned differently ({plan_digest} != "
                             f"{reference_plan}); its predictions are not ensemble-compatible")
        target = snapshot / f"model_{checkpoint}_{fold_dir.name}.ckpt"
        if not target.is_file():
            shutil.copy2(source, target)
        staged.append({"fold_dir": str(fold_dir.resolve()), "staged_as": target.name,
                       "checkpoint_sha256": sha256(target),
                       "checkpoint_epoch": int(torch.load(target, map_location="cpu")["epoch"])})
    return staged


def main() -> None:
    args = parse_args()
    checkpoint_path = args.training_dir / f"model_{args.checkpoint}.ckpt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but a new CUDA context is not available")

    # The prebuilt nndet._C was compiled for an earlier GPU generation; on hardware
    # it does not cover, its one kernel raises instead of running.  Probe rather
    # than assume, and record the answer in the run metadata.
    cpu_nms = args.device.startswith("cuda") and not cuda_nms_is_usable()
    if cpu_nms:
        install_cpu_nms_fallback()
        print("nndet._C NMS kernel unusable on this device; routing NMS through the "
              "CPU implementation. Convolutions still run on the GPU.", flush=True)

    cfg = OmegaConf.load(args.training_dir / "config.yaml")
    for module_name in cfg.get("additional_imports", []):
        importlib.import_module(module_name)
    plan = load_pickle(args.training_dir / "plan.pkl")
    if args.batch_size is not None:
        plan["batch_size"] = args.batch_size
    if args.inference_plan is not None:
        payload = json.loads(args.inference_plan.read_text())
        swept = payload.get("plan", payload)
        # only numeric entries: the NMS callables are named as strings in the JSON and
        # the sweep left them at their defaults, so there is nothing to resolve
        numeric = {key: value for key, value in swept.items()
                   if isinstance(value, (int, float)) and not isinstance(value, bool)}
        plan["inference_plan"] = {**plan.get("inference_plan", {}), **numeric}
    if args.max_detections is not None:
        plan["inference_plan"] = {**plan.get("inference_plan", {}),
                                  **detection_cap_override(args.max_detections)}
        plan["architecture"]["detections_per_img"] = args.max_detections

    available = sorted(path.stem for path in args.source_dir.glob("*.npz"))
    requested = available if args.case_id is None else args.case_id
    missing = sorted(set(requested) - set(available))
    if missing:
        raise FileNotFoundError(f"preprocessed cases not found: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending = [case_id for case_id in requested
               if args.overwrite or not (args.output_dir / f"{case_id}_boxes.pkl").is_file()]

    if args.ensemble_from:
        snapshot = args.output_dir / "model_snapshot"
        staged = stage_ensemble_snapshot(args.ensemble_from, args.checkpoint,
                                         args.training_dir, snapshot)
        shutil.copy2(args.training_dir / "config.yaml", snapshot / "config.yaml")
        shutil.copy2(args.training_dir / "plan.pkl", snapshot / "plan.pkl")
        source_models, model_fn, num_models = snapshot, load_all_models, len(staged)
    else:
        staged = None
        source_models = args.training_dir
        model_fn = partial(load_final_model, identifier=args.checkpoint)
        num_models = 1

    metadata = {
        "training_dir": str(args.training_dir.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": int(torch.load(checkpoint_path, map_location="cpu")["epoch"]),
        "ensemble_members": staged,
        "num_models": num_models,
        "source_dir": str(args.source_dir.resolve()),
        "device": args.device,
        "num_tta": args.num_tta,
        "batch_size": int(plan["batch_size"]),
        "requested_cases": len(requested),
        "pending_cases_at_start": len(pending),
        "nms_backend": "cpu_fallback" if cpu_nms else "nndet._C cuda kernel",
        "max_detections": args.max_detections,
        "inference_plan_source": (None if args.inference_plan is None
                                  else str(args.inference_plan)),
        "inference_plan": {key: value for key, value in plan.get("inference_plan", {}).items()
                           if isinstance(value, (int, float, str, bool, type(None)))},
        "postprocessing": (
            "nnDetection BoxEnsemblerSelective defaults; no ADAM tuning"
            if args.inference_plan is None and args.max_detections is None else
            "BoxEnsemblerSelective with an inference plan fitted on in-house folds 0/2; "
            "nothing here was selected on ADAM90"),
    }
    metadata_name = f"run_metadata_{requested[0]}_to_{requested[-1]}.json"
    (args.output_dir / metadata_name).write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    if not pending:
        print("All requested cases already have predictions.", flush=True)
        return

    predict_dir(
        source_dir=args.source_dir,
        target_dir=args.output_dir,
        cfg=cfg,
        plan=plan,
        source_models=source_models,
        model_fn=model_fn,
        num_models=num_models,
        num_tta_transforms=args.num_tta,
        restore=True,
        case_ids=pending,
        save_state=False,
        device=args.device,
        ensemble_on_device=not args.device.startswith("cpu"),
    )


if __name__ == "__main__":
    main()
