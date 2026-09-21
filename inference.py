"""Grand Challenge Task2 adapter around frozen r5. No model-policy changes."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import SimpleITK as sitk

from postprocessing import aligned_output

ROOT = Path(__file__).resolve().parent
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

IMAGE_EXTENSIONS = (
    ".nii.gz",
    ".nii",
    ".mha",
    ".mhd",
    ".tif",
    ".tiff",
)


def find_image_files(location: Path) -> list[Path]:
    if not location.exists():
        return []
    if location.is_file():
        name_lower = location.name.lower()
        if any(name_lower.endswith(ext) for ext in IMAGE_EXTENSIONS):
            return [location]
        return []

    found = []
    # 1. Direct children
    for item in location.iterdir():
        if item.is_file():
            name_lower = item.name.lower()
            if any(name_lower.endswith(ext) for ext in IMAGE_EXTENSIONS):
                found.append(item)

    # 2. Recursive fallback
    if not found:
        for item in location.rglob("*"):
            if item.is_file():
                name_lower = item.name.lower()
                if any(name_lower.endswith(ext) for ext in IMAGE_EXTENSIONS):
                    found.append(item)

    return sorted(found)


def make_empty_mask(reference: sitk.Image) -> sitk.Image:
    """Create a valid empty (all-zeros, uint8) segmentation mask matching reference geometry."""
    empty = sitk.Image(reference.GetSize(), sitk.sitkUInt8)
    empty.CopyInformation(reference)
    return empty


def predict_image(image: sitk.Image, modality: str) -> sitk.Image:
    detector = os.environ.get("TASK2_DETECTOR_PYTHON", "/opt/envs/detector/bin/python")
    refinement = os.environ.get("TASK2_REFINEMENT_PYTHON", sys.executable)
    env = os.environ.copy()
    env["GIT_PYTHON_REFRESH"] = "quiet"
    with tempfile.TemporaryDirectory(prefix="task2-case-") as td:
        td = Path(td)
        raw, pred = td / "input.nii.gz", td / "prediction.nii.gz"
        sitk.WriteImage(image, str(raw), True)
        proc = subprocess.run(
            [
                refinement,
                str(ROOT / "run_inference.py"),
                "--image",
                str(raw),
                "--modality",
                modality,
                "--output",
                str(pred),
                "--work",
                str(td / "work"),
                "--gpu",
                os.environ.get("TASK2_GPU", "0"),
                "--detector-python",
                detector,
                "--refinement-python",
                refinement,
                "--mr-policy",
                "detector_control",
            ],
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"run_inference.py failed with exit code {proc.returncode}")
        if not pred.is_file():
            raise RuntimeError(f"Prediction file {pred} was not generated")
        return aligned_output(sitk.ReadImage(str(pred)), image)


def infer_ct(img):
    return predict_image(img, "CT")


def infer_mr(img):
    return predict_image(img, "MR")


def resolve_input(input_dir: Path) -> tuple[sitk.Image, str]:
    interfaces = {
        "head-ct-angiography": ("head-ct-angio", "CT"),
        "head-mr-angiography": ("head-mr-angio", "MR"),
    }
    inputs_json_path = input_dir / "inputs.json"
    folder = None
    modality = None

    if inputs_json_path.is_file():
        try:
            inputs = json.loads(inputs_json_path.read_text())
            slugs = tuple(sorted(v["socket"]["slug"] for v in inputs))
            for slug in slugs:
                if slug in interfaces:
                    folder, modality = interfaces[slug]
                    break
        except Exception as e:
            print(f"[*] Warning reading inputs.json: {e}", file=sys.stderr)

    if folder is None:
        ct_dir = input_dir / "images" / "head-ct-angio"
        mr_dir = input_dir / "images" / "head-mr-angio"
        ct_files = find_image_files(ct_dir) if ct_dir.is_dir() else []
        mr_files = find_image_files(mr_dir) if mr_dir.is_dir() else []
        if ct_files and not mr_files:
            folder, modality = "head-ct-angio", "CT"
        elif mr_files and not ct_files:
            folder, modality = "head-mr-angio", "MR"
        else:
            all_imgs = find_image_files(input_dir / "images") + find_image_files(input_dir)
            if all_imgs:
                if "mr" in str(all_imgs[0]).lower():
                    folder, modality = "head-mr-angio", "MR"
                else:
                    folder, modality = "head-ct-angio", "CT"
            else:
                raise ValueError("Could not find input images or resolve modality")

    image_folder = input_dir / "images" / folder
    files = find_image_files(image_folder)
    if not files and image_folder.parent.exists():
        files = find_image_files(image_folder.parent)
    if not files:
        raise ValueError(f"Expected at least one input volume in {image_folder}")

    print(f"[*] Loading input volume from: {files[0]} (Modality: {modality})", flush=True)
    reference = sitk.ReadImage(str(files[0]))
    return reference, modality


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("/input"))
    p.add_argument("--output-dir", type=Path, default=Path("/output"))
    a = p.parse_args()

    target = a.output_dir / "images/aneurysm-segmentation/output.mha"
    target.parent.mkdir(parents=True, exist_ok=True)

    reference = None
    try:
        reference, modality = resolve_input(a.input_dir)
        try:
            result = predict_image(reference, modality)
        except Exception as pred_err:
            print(f"[ERROR] Task 2 inference failed: {pred_err}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            print("[*] Falling back to safe empty mask (all zeros) matching input geometry...", file=sys.stderr)
            result = make_empty_mask(reference)

        validated = aligned_output(result, reference)
        sitk.WriteImage(validated, str(target), True)

        written = sitk.ReadImage(str(target))
        aligned_output(written, reference)
        for attr in ("GetSize", "GetSpacing", "GetOrigin", "GetDirection"):
            if getattr(written, attr)() != getattr(reference, attr)():
                raise ValueError(f"MHA output did not preserve exact geometry: {attr}")

        print(f"[*] Task 2 output successfully written and verified: {target}", flush=True)
        return 0

    except Exception as e:
        print(f"[FATAL] Task 2 top-level failure: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        if reference is not None:
            try:
                fallback = make_empty_mask(reference)
                sitk.WriteImage(fallback, str(target), True)
                print("[*] Emergency fallback mask written.", flush=True)
                return 0
            except Exception as fe:
                print(f"[FATAL] Failed to write emergency fallback: {fe}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
