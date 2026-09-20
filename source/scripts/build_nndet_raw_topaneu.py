"""Builds nnDetection raw dataset layout for TopAneu, split by modality.

2026-08-14, peer priority: TopAneu detector preprocessing (task #26), CPU-only,
doesn't compete with A3's GPU training. Checked nnDetection source
(`nndet/planning/experiment/base.py`, `nndet/preprocessing/preprocessor.py`)
before writing this: `dataset.json`'s "modalities" field is keyed by CHANNEL
INDEX and applied UNIFORMLY across every case in the dataset -- CT cases get
`normalize_ct` (global training-set intensity stats: mean/std/percentile
bounds fit once across the whole dataset), non-CT cases get `normalize_other`
(per-case z-score). A single task mixing TopAneu's 308 MR + 109 CT cases
would either corrupt the CT intensity-property fit with MR's completely
different physical units, or apply MR-style per-case z-score normalization to
Hounsfield-unit CT data -- neither is correct. So this builds TWO separate
nnDetection raw datasets:
  - Task030FG_TopAneuMR (308 cases, modality "TOF" -- same string
    Task020FG_LocalAneurysm already uses for the in-house MRA cohort, so
    normalization scheme matches exactly)
  - Task031FG_TopAneuCT (109 cases, modality "CT")

Detector task is single-class ("Aneurysm", like Task020FG) -- the 52-way
location classification happens downstream (geometric attachment / a
separate classification head), not in the detector. `location_masks/` (0-52
semantic classes) are binarized (>0) then 26-connected-component labeled to
produce nnDetection's expected per-case instance mask + json
{"instances": {"<instance_id>": 0}} mapping (all instances -> class 0), the
same format Task020FG_LocalAneurysm/raw_splitted/labelsTr/*.json already
uses (verified by reading an existing file directly, not assumed).

Images are copied (not symlinked) into imagesTr with the SAME filename
TopAneu already uses (`{case_id}_0000.nii.gz`) -- data_topaneu26/images/
already follows nnDetection's `_0000` channel-suffix convention exactly, so
no renaming needed, just routing into the right modality-specific task dir.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

MANIFEST = Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task1/artifacts/phase0/case_manifest.json")
DATA_ROOT = Path("/home/jovyan/rtx4claude-datavol-1/data_topaneu26")
NNDET_DATA_ROOT = Path("/home/jovyan/rtx4claude-datavol-1/nndet_data")

TASKS = {
    "mr": {"task_dir": NNDET_DATA_ROOT / "Task030FG_TopAneuMR",
          "task_name": "Task030FG_TopAneuMR", "modality": "TOF"},
    "ct": {"task_dir": NNDET_DATA_ROOT / "Task031FG_TopAneuCT",
          "task_name": "Task031FG_TopAneuCT", "modality": "CT"},
}


def build_instance_label(location_mask_path: Path) -> tuple[np.ndarray, dict, nib.Nifti1Image]:
    img = nib.load(location_mask_path)
    data = np.asarray(img.dataobj)
    binary = data > 0
    labeled, n_instances = ndimage.label(binary, structure=np.ones((3, 3, 3), dtype=np.uint8))
    instances = {str(i): 0 for i in range(1, n_instances + 1)}  # single class "Aneurysm" = 0
    label_img = nib.Nifti1Image(labeled.astype(np.uint8), img.affine, img.header)
    return labeled, instances, label_img


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())

    for modality, cfg in TASKS.items():
        task_dir = cfg["task_dir"]
        images_out = task_dir / "raw_splitted" / "imagesTr"
        labels_out = task_dir / "raw_splitted" / "labelsTr"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        cases = [c for c in manifest["cases"] if c["modality"] == modality]
        print(f"{cfg['task_name']}: {len(cases)} cases", flush=True)

        total_instances = 0
        empty_cases = 0
        for i, case in enumerate(cases):
            if i % 50 == 0:
                print(f"  {modality} progress: {i}/{len(cases)}", flush=True)
            case_id = case["case_id"]

            src_image = Path(case["image"])
            dst_image = images_out / f"{case_id}_0000.nii.gz"
            if not dst_image.exists():
                shutil.copy2(src_image, dst_image)

            dst_label = labels_out / f"{case_id}.nii.gz"
            dst_json = labels_out / f"{case_id}.json"
            if not dst_label.exists() or not dst_json.exists():
                _, instances, label_img = build_instance_label(Path(case["location_mask"]))
                nib.save(label_img, dst_label)
                dst_json.write_text(json.dumps({"instances": instances}))
                total_instances += len(instances)
                if not instances:
                    empty_cases += 1

        dataset_json = {
            "name": cfg["task_name"].split("_", 1)[1],
            "task": cfg["task_name"],
            "target_class": None,
            "test_labels": False,
            "labels": {"0": "Aneurysm"},
            "modalities": {"0": cfg["modality"]},
            "dim": 3,
        }
        (task_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))

        print(f"{cfg['task_name']}: wrote {len(cases)} images, ~{total_instances} lesion instances "
             f"(recomputed only for missing files), {empty_cases} empty-label cases (negative controls)",
             flush=True)

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
