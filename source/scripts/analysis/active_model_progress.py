"""Persist a concise, read-only model-workflow snapshot for monitoring."""
import json
from pathlib import Path
from datetime import datetime,timezone
P=Path(__file__).resolve().parents[2]
def read(path,default=None):
 try:return json.loads(path.read_text())
 except (OSError,json.JSONDecodeError):return default

def main():
 r19=P/'artifacts/astra6_e19_MR_anatomical_fp_filter_20260909';r20=P/'artifacts/astra6_e20_CT_detector_crop_segmentation_20260909';cache=P/'artifacts/source_MR_TA36_distribution_audit_20260909';ids=read(r19/'TA36_REQUIRED_CASES.json',{'cases':[]})['cases']
 state={'time_UTC':datetime.now(timezone.utc).isoformat(),'frozen_best':read(P/'LATEST_MODEL_RELEASE.json',{}).get('archive'),'E19':{'source_TA36_completed':sum((cache/c/'PROVENANCE.json').exists() for c in ids),'source_TA36_required':len(ids),'source_TA36_ready':(r19/'TA36_READY.json').exists(),'full_training_complete':(r19/'model/TRAINING_COMPLETE.json').exists(),'source_validation':read(r19/'evaluation/source_validation.json'),'adoption':read(r19/'ADOPTION.json')},'E20':{'source_ready':(r20/'features/SOURCE_READY.json').exists(),'waiting_for_GPU_dependency':not (r19/'TA36_READY.json').exists(),'full_training_complete':(r20/'model/LOCKED.json').exists(),'source_validation':read(r20/'evaluation/source_validation.json'),'adoption':read(r20/'ADOPTION.json')}}
 for stage in ['development','final']:
  h=read(r20/f'model/{stage}_history.json',[]);state['E20'][stage]={'completed_epochs':len(h),'planned_epochs':13,'last':h[-1] if h else None}
 for key,run,version in [('E19',r19,'E19'),('E20',r20,'CT_E20')]:
  pred=run/('predictions_ct' if key=='E20' else 'predictions/mr_center2_k05');state[key]['prediction_cases']=len(list(pred.glob('*.nii.gz')));state[key]['official_complete']=(P/f'artifacts/current_official_20260909/{version}/DONE.json').exists()
 for name,folder in [('E21','astra6_e21_CT_native_checkpoint_selection_20260909'),('E22','astra6_e22_CT_only_segmentation_20260909')]:
  run=P/'artifacts'/folder;history=read(run/'model/native_history.json',[]);final=read(run/'model/final_history.json',[]);state[name]={'development_completed_epochs':len(history),'source_native_best':max([a['native_Dice'] for a in history],default=None),'final_completed_epochs':len(final),'full_training_complete':(run/'model/LOCKED.json').exists(),'adoption':read(run/'ADOPTION.json'),'source_validation':read(run/'evaluation/source_validation.json')}
 dest=P/'LIVE_MODEL_PROGRESS_20260909.json';temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(dest)
 print(json.dumps(state,ensure_ascii=False))
if __name__=='__main__':main()
