"""Grand Challenge Task2 adapter around frozen r5. No model-policy changes."""
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path
import SimpleITK as sitk
from postprocessing import aligned_output
ROOT = Path(__file__).resolve().parent

def predict_image(image, modality):
    detector = os.environ.get("TASK2_DETECTOR_PYTHON", "/opt/envs/detector/bin/python")
    refinement = os.environ.get("TASK2_REFINEMENT_PYTHON", sys.executable)
    with tempfile.TemporaryDirectory(prefix="task2-case-") as td:
        td = Path(td)
        raw, pred = td/"input.nii.gz", td/"prediction.nii.gz"
        sitk.WriteImage(image, str(raw), True)
        subprocess.run([refinement, str(ROOT/"run_inference.py"), "--image", str(raw),
                        "--modality", modality, "--output", str(pred), "--work", str(td/"work"),
                        "--gpu", os.environ.get("TASK2_GPU", "0"), "--detector-python", detector,
                        "--refinement-python", refinement, "--mr-policy", "detector_control"], check=True)
        return aligned_output(sitk.ReadImage(str(pred)), image)

def infer_ct(img): return predict_image(img, "CT")
def infer_mr(img): return predict_image(img, "MR")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("/input"))
    p.add_argument("--output-dir", type=Path, default=Path("/output"))
    a=p.parse_args()
    interfaces={"head-ct-angiography":("head-ct-angio","CT"), "head-mr-angiography":("head-mr-angio","MR")}
    inputs=json.loads((a.input_dir/"inputs.json").read_text())
    slugs=tuple(sorted(v["socket"]["slug"] for v in inputs))
    if len(slugs)!=1 or slugs[0] not in interfaces:
        raise ValueError("Unsupported GC input interface: " + str(slugs))
    folder, modality=interfaces[slugs[0]]
    files=[f for f in (a.input_dir/"images"/folder).iterdir() if f.is_file() and f.name.lower().endswith((".mha",".tif",".tiff",".nii.gz",".nii"))]
    if len(files)!=1: raise ValueError("Expected exactly one input volume")
    reference=sitk.ReadImage(str(files[0]))
    result=predict_image(reference, modality)
    target=a.output_dir/"images/aneurysm-segmentation/output.mha"
    target.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(result,str(target),True)
    written=sitk.ReadImage(str(target))
    aligned_output(written,reference)
    for attr in ("GetSize","GetSpacing","GetOrigin","GetDirection"):
        if getattr(written,attr)()!=getattr(reference,attr)():
            raise ValueError("MHA output did not preserve exact geometry")
if __name__=="__main__": main()
