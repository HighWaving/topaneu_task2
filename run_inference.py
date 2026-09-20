#!/usr/bin/env python3
"""Portable fixed MR D/S32 control and retained CT E16 image-only entry."""
import argparse,json,os,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser()
 for k in ['image','output','work']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--mr-policy',choices=['detector_control','dense_fusion'],default='detector_control');p.add_argument('--modality',choices=['MR','CT'],required=True);p.add_argument('--gpu',default='0');p.add_argument('--detector-python',default=os.environ.get('TASK2_DETECTOR_PYTHON'));p.add_argument('--refinement-python',default=os.environ.get('TASK2_REFINEMENT_PYTHON',sys.executable));a=p.parse_args()
 if not a.detector_python:p.error('Provide --detector-python for the installed nnDetection environment')
 common=['--image',str(a.image.resolve()),'--output',str(a.output.resolve()),'--work',str(a.work.resolve()),'--gpu',a.gpu]
 env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
 if a.modality=='MR':
  cmd=[a.refinement_python,'-m','scripts.delivery.raw_dense_fusion',*common,'--policy',a.mr_policy,'--bundle',str(ROOT),'--calibration',str(ROOT/'configs/MR_dense_fusion_policy.json'),'--detector-python',a.detector_python,'--refinement-python',a.refinement_python];raise SystemExit(subprocess.run(cmd,cwd=ROOT/'source',env=env).returncode)
 modelcfg=json.loads((ROOT/'configs/models.json').read_text());cfg={k:v for k,v in modelcfg.items() if k.endswith('_threshold') or k.endswith('_architecture')};cfg.update(detector_python=str(Path(a.detector_python).resolve()),refinement_python=str(Path(a.refinement_python).resolve()),CT_detector=str(ROOT/'models/detector_CT'),detector_code=str(ROOT/'source/dependencies/detector_app'),ta36_code=str(ROOT/'source/dependencies/ta36_app'),ta36_models=str(ROOT/'models/ta36'),CT_location_classifier=str(ROOT/'models/location_CT.joblib'),CT_fp_filter=str(ROOT/'models/filter_CT.pt'),CT_anatomical_filter=str(ROOT/'models/filter_CT_anatomical.joblib'))
 with tempfile.TemporaryDirectory(prefix='task2-runtime-') as tmp:
  path=Path(tmp)/'runtime.json';path.write_text(json.dumps(cfg));cmd=[a.refinement_python,'-m','scripts.delivery.raw_pipeline','--runtime-config',str(path),*common,'--modality','CT','--segmentation',str(ROOT/'models/segmentation_CT.pt')];raise SystemExit(subprocess.run(cmd,cwd=ROOT/'source',env=env).returncode)
if __name__=='__main__':main()
