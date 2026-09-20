from pathlib import Path
import sys,runpy,json
P=Path('/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2');B=P/'submission_models_20260909';DATA=P.parent/'data_topaneu26';cid='topaneu_center2_mr_025';image=DATA/f'images/{cid}_0000.nii.gz';vessel=P/f'artifacts/ta36_mr_center2_output/{cid}.nii.gz';boxes=P/f'artifacts/fold1_eval_center2_epoch60/{cid}_boxes.pkl';out=P/'artifacts/bundle_native_readscope_20260909/MR025.mha';out.parent.mkdir(parents=True,exist_ok=True);ledger=out.parent/'READ_SCOPE.json';allowed={image.resolve(),vessel.resolve(),boxes.resolve(),out.resolve(),out.with_suffix('.json').resolve(),ledger.resolve()};reads=set()
def audit(event,args):
 if event!='open' or not isinstance(args[0],(str,bytes)):return
 try:path=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).resolve()
 except (OSError,ValueError):return
 if (path.is_relative_to(P) or path.is_relative_to(DATA)) and not path.is_relative_to(B):
  if path not in allowed:raise RuntimeError(f'Inference attempted non-input project/data access: {path}')
  reads.add(str(path))
sys.addaudithook(audit);sys.path.insert(0,str(B/'source'));cfg=json.loads((B/'configs/models.json').read_text());sys.argv=['refine_with_filter','--image',str(image),'--predicted-vessel',str(vessel),'--boxes',str(boxes),'--classifier',str(B/'models/location_MR.joblib'),'--segmentation',str(B/'models/segmentation_MR.pt'),'--fp-filter',str(B/'models/filter_MR.pt'),'--fp-threshold',str(cfg['MR_fp_threshold']),'--modality','MR','--device','cpu','--output',str(out)];runpy.run_module('scripts.delivery.refine_with_filter',run_name='__main__');ledger.write_text(json.dumps({'passed':True,'original_project_and_GT_data_access_blocked_except_explicit_inputs':True,'explicit_inputs':[str(image),str(vessel),str(boxes)],'observed_permitted_accesses':sorted(reads),'bundle':str(B)},indent=2)+'\n')
