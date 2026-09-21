#!/usr/bin/env python3
"""Portable fixed MR D/S32 control and retained CT E16 image-only entry."""
import argparse,json,os,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def resolve_model_dir(root: Path) -> Path:
    candidates = [
        Path(os.environ.get("GRAND_CHALLENGE_MODEL_ROOT", "/opt/ml/model")),
        Path(os.environ.get("GRAND_CHALLENGE_MODEL_ROOT", "/opt/ml/model")) / "models",
        root / "models",
    ]
    for c in candidates:
        plan_ct = c / "detector_CT" / "plan.pkl"
        if plan_ct.is_file():
            try:
                head = plan_ct.open("rb").read(30)
                if not head.startswith(b"version https://git-lfs"):
                    return c
            except Exception:
                pass
    return candidates[0]

def main():
 p=argparse.ArgumentParser()
 for k in ['image','output','work']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--mr-policy',choices=['detector_control','dense_fusion'],default='detector_control');p.add_argument('--modality',choices=['MR','CT'],required=True);p.add_argument('--gpu',default='0');p.add_argument('--detector-python',default=os.environ.get('TASK2_DETECTOR_PYTHON'));p.add_argument('--refinement-python',default=os.environ.get('TASK2_REFINEMENT_PYTHON',sys.executable));a=p.parse_args()
 if not a.detector_python:p.error('Provide --detector-python for the installed nnDetection environment')
 model_dir = resolve_model_dir(ROOT)
 plan_test = model_dir / ("detector_CT" if a.modality == "CT" else "detector_MR") / "plan.pkl"
 if not plan_test.is_file() or plan_test.open("rb").read(30).startswith(b"version https://git-lfs"):
     sys.stderr.write(
         f"\n[ERROR] Task 2 model weights not found or are unpopulated Git LFS pointers in {model_dir}.\n"
         "Please upload the model tarball (topaneu-task2-models.tar.gz) in the Grand Challenge 'Models' tab.\n"
         "Grand Challenge automatically extracts the tarball to /opt/ml/model/ at runtime.\n\n"
     )
     sys.stderr.flush()
     raise RuntimeError(f"Missing Task 2 model weights in {model_dir}")
 common=['--image',str(a.image.resolve()),'--output',str(a.output.resolve()),'--work',str(a.work.resolve()),'--gpu',a.gpu]
 env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
 if a.modality=='MR':
  cmd=[a.refinement_python,'-m','scripts.delivery.raw_dense_fusion',*common,'--policy',a.mr_policy,'--bundle',str(ROOT),'--model-dir',str(model_dir),'--calibration',str(ROOT/'configs/MR_dense_fusion_policy.json'),'--detector-python',a.detector_python,'--refinement-python',a.refinement_python];raise SystemExit(subprocess.run(cmd,cwd=ROOT/'source',env=env).returncode)
 modelcfg=json.loads((ROOT/'configs/models.json').read_text());cfg={k:v for k,v in modelcfg.items() if k.endswith('_threshold') or k.endswith('_architecture')};cfg.update(detector_python=str(Path(a.detector_python).resolve()),refinement_python=str(Path(a.refinement_python).resolve()),CT_detector=str(model_dir/'detector_CT'),detector_code=str(ROOT/'source/dependencies/detector_app'),ta36_code=str(ROOT/'source/dependencies/ta36_app'),ta36_models=str(model_dir/'ta36'),CT_location_classifier=str(model_dir/'location_CT.joblib'),CT_fp_filter=str(model_dir/'filter_CT.pt'),CT_anatomical_filter=str(model_dir/'filter_CT_anatomical.joblib'))
 with tempfile.TemporaryDirectory(prefix='task2-runtime-') as tmp:
  path=Path(tmp)/'runtime.json';path.write_text(json.dumps(cfg));cmd=[a.refinement_python,'-m','scripts.delivery.raw_pipeline','--runtime-config',str(path),*common,'--modality','CT','--segmentation',str(model_dir/'segmentation_CT.pt')];raise SystemExit(subprocess.run(cmd,cwd=ROOT/'source',env=env).returncode)
if __name__=='__main__':main()
