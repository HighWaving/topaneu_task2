"""Sequential next-capability research, invoked only after E25 full evaluation."""
from pathlib import Path
import json,os,subprocess,time
P=Path(__file__).resolve().parents[2];RUN=P/'artifacts/astra6_e28_joint_anatomy_location_20260909';F=P/'artifacts/astra6_e25_oof_shape_fp_filter_20260909';S=P/'artifacts/astra6_e17_MR_detector_crop_segmentation_20260909';PY=P.parent/'conda_envs/nnunet_v100/bin/python';EVAL=P/'.venv_official_eval_20260909/bin/python'
def main():
 assert (F/'evaluation/PAIRED_ERROR_ATTRIBUTION.json').exists() and (F/'evaluation/DECISION.json').exists()
 decision=json.loads((F/'evaluation/DECISION.json').read_text());baseline=decision['valid_MR_baseline_for_next_research'];parent=F if baseline=='E25' else S;diagnosis=json.loads((parent/'evaluation/lesion_diagnostics.json').read_text());lesions=[l for c in diagnosis['cases'].values() for l in c['lesions']];errors=[l for l in lesions if l['matched'] and not l['class_correct'] and l['class']%2==l['predicted_class']%2 and ((22<=l['class']<=35 and 22<=l['predicted_class']<=35) or (45<=l['class']<=52 and 45<=l['predicted_class']<=52))]
 from scripts.astra6_e01.e01_common import write_json,sha256_file
 if len(errors)<8:
  write_json(RUN/'NEXT_BOTTLENECK_REVIEW_REQUIRED.json',{'baseline':baseline,'remaining_same_side_internal_ICA_MCA_errors':len(errors),'reason':'Prespecified representation hypothesis no longer dominant enough for automatic fit; inspect E25 outcome before choosing next training.'});return
 marker=RUN/'RESEARCH_START.json';start={'baseline':baseline,'remaining_same_side_internal_ICA_MCA_errors':len(errors),'baseline_candidates_sha256':sha256_file(parent/'candidate_predictions.jsonl'),'E25_decision_sha256':sha256_file(F/'evaluation/DECISION.json'),'reason':'E25 full diagnostics complete; at least8 same-side internal ICA/MCA errors remain in the valid baseline. Address image/lesion-parent representation, retaining effective family/laterality model.','no_E23_restart':True,'start_timestamp':time.time()}
 if marker.exists():
  previous=json.loads(marker.read_text());assert previous['baseline']==baseline and previous['baseline_candidates_sha256']==start['baseline_candidates_sha256']
 else:write_json(marker,start)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='GPU-a643ded3-193b-58e8-b362-be93dc8eac14',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
 for module in ['prepare','train','infer']:
  subprocess.run([str(PY),'-m',f'scripts.astra6_e28.{module}'],cwd=P,env=env,check=True)
 subprocess.run([str(EVAL),'-m','scripts.analysis.current_official_evaluation','--version','E28'],cwd=P,env=env,check=True)
 subprocess.run([str(PY),'-m','scripts.astra6_e28.evaluate'],cwd=P,env=env,check=True)
 subprocess.run([str(EVAL),'-m','scripts.astra6_e28.compare'],cwd=P,env=env,check=True)
if __name__=='__main__':main()
