"""Assemble weights and inference source only; no container construction or environments."""
from pathlib import Path
import json,shutil,hashlib
P=Path(__file__).resolve().parents[2];DEST=P/'submission_models_20260909'
def cp(src,dst):
 src=Path(src);dst=Path(dst)
 if src.is_dir():
  dst.mkdir(parents=True,exist_ok=True)
  for f in src.iterdir():
   if f.name in ['__pycache__','.git'] or f.suffix=='.pyc':continue
   cp(f,dst/f.name)
 else:
  dst.parent.mkdir(parents=True,exist_ok=True)
  if dst.exists() and (src.stat().st_size,src.stat().st_mtime_ns)==(dst.stat().st_size,dst.stat().st_mtime_ns):return
  shutil.copy2(src,dst)
def main():
 best=json.loads((P/'BEST_VALIDATED_20260909.json').read_text());base=P/'delivery_20260909/models';mapping={base/'detector_MR':'detector_MR',base/'detector_CT':'detector_CT',base/'ta36':'ta36',P/best['segmenter']:'segmentation_MR.pt',P/best.get('CT_segmenter',best['segmenter']):'segmentation_CT.pt'}
 # Segmentation paths can coincide, so preserve both destination names explicitly.
 for name in ['detector_MR','detector_CT','ta36']:cp(base/name,DEST/'models'/name)
 weights={'segmentation_MR.pt':best['segmenter'],'segmentation_CT.pt':best.get('CT_segmenter',best['segmenter']),'location_MR.joblib':best['classifier'],'location_CT.joblib':best['CT_classifier'],'filter_MR.pt':best['MR_filter'],'filter_CT.pt':best['CT_filter'],'filter_CT_anatomical.joblib':best['CT_anatomical_filter']}
 for name,src in weights.items():cp(P/src,DEST/'models'/name)
 cp(P/'delivery_20260909/source/dependencies',DEST/'source/dependencies');cp(P/'scripts',DEST/'source/scripts');cp(P/'vendor/delivery_runtime_deps',DEST/'source/vendor/delivery_runtime_deps')
 cp(P/'delivery_20260909/configs/environment_nndet.json',DEST/'configs/environment_nndet.json');cp(P/'delivery_20260909/configs/environment_nnunet_v100.json',DEST/'configs/environment_nnunet_v100.json');cp(P/'BEST_VALIDATED_20260909.json',DEST/'reports/BEST_VALIDATED.json');cp(P/'reports/MODEL_COMPARISON_INTEGRITY_20260909.json',DEST/'reports/COMPARISON_INTEGRITY.json')
 cp(P/'model_update_20260909/reports',DEST/'reports/history')
 config={'MR_fp_threshold':best['MR_filter_threshold'],'CT_fp_threshold':best['CT_filter_threshold'],'CT_anatomical_threshold':best['CT_anatomical_threshold'],'system_version':best['system_version']};(DEST/'configs/models.json').write_text(json.dumps(config,indent=2)+'\n')
 manifest={str(f.relative_to(DEST)):{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted((DEST/'models').rglob('*')) if f.is_file()};(DEST/'MODEL_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');print('MODEL_RELEASE_ASSEMBLED',DEST,len(manifest),flush=True)
if __name__=='__main__':main()
