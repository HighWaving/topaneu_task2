from pathlib import Path
import json,shutil
import numpy as np,torch
from scripts.astra6_e12 import train as training
from scripts.astra6_e14.common import RUN,Filter,P
from scripts.astra6_e01.e01_common import load_boxes,select_candidates,write_json,sha256_file
PARENT=P/'artifacts/astra6_e12_MR_expanded_fp_filter_20260909'
def combine(paths):
 model=Filter()
 for net,path in zip(model.models,paths):net.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state_dict'])
 return model

def main():
 for part in ['model','evaluation','features']:(RUN/part).mkdir(parents=True,exist_ok=True)
 rec=[json.loads(s) for s in (PARENT/'features/records.jsonl').read_text().splitlines()];tr=[i for i,r in enumerate(rec) if not r['development']];dv=[i for i,r in enumerate(rec) if r['development']];all_ix=list(range(len(rec)));assert not {rec[i]['group'] for i in tr}&{rec[i]['group'] for i in dv};source_split=json.loads((PARENT/'source_split.json').read_text());source_cases=set(source_split['train_cases']+source_split['development_cases']);assert len(source_cases)==267 and {r['case_id'] for r in rec}<=source_cases and not any('center2' in r['case_id'] or '_ct_' in r['case_id'] for r in rec)
 write_json(RUN/'config.json',{'experiment':'E14','hypothesis':'Three fixed-seed MR image filters reduce single-run variance observed in E12, without changing data, architecture, epoch budget or proposal selection','seeds':[20260909,20260910,20260911],'member0':'reuse fully trained E12 selected25epochs','new_members':'two seeds, each source-development25epochs and full267case25epochs','aggregation':'arithmetic mean of three probabilities; one predeclared ensemble, no subset or seed sweep','threshold':'minimum mean probability among source-dev original score.3/top5 positive candidates','source':'same267MR source cases and case-group split as E12; MR40/CT5 excluded','adoption':'official all-six Pareto gain versus E06 and at least53/58 matched lesions; otherwise preserve E06','budget_hours':3,'checkpoint':'each epoch optimizer and all RNG state'})
 shutil.copy2(PARENT/'source_split.json',RUN/'source_split.json');shutil.copy2(PARENT/'features/records.jsonl',RUN/'features/records.jsonl');development=[PARENT/'model/development_best.pt'];final=[PARENT/'model/final_last.pt'];assert torch.load(development[0],map_location='cpu',weights_only=False)['epoch']==25
 for seed in [20260910,20260911]:
  work=RUN/f'seed{seed}';(work/'model').mkdir(parents=True,exist_ok=True)
  if not (work/'features').exists():(work/'features').symlink_to(PARENT/'features',target_is_directory=True)
  training.RUN=work;training.SEED=seed
  if not (work/'model/COMPLETE.json').exists():
   training.fit('development',tr,dv,epochs=25);training.fit('final',all_ix,epochs=25);write_json(work/'model/COMPLETE.json',{'seed':seed,'development_epochs':25,'final_epochs':25,'development_last_sha256':sha256_file(work/'model/development_last.pt'),'final_sha256':sha256_file(work/'model/final_last.pt')})
  development.append(work/'model/development_last.pt');final.append(work/'model/final_last.pt')
 model=combine(development).cuda().eval();images=np.load(PARENT/'features/images.npy',mmap_mode='r');ps=[]
 with torch.inference_mode():
  for start in range(0,len(dv),32):ps.extend(model(torch.from_numpy(np.asarray(images[dv[start:start+32]],np.float32)).cuda()).sigmoid().cpu().tolist())
 yy=np.array([rec[i]['y'] for i in dv]);pool={}
 for cid in {rec[i]['case_id'] for i in dv}:
  folder=P/('artifacts/fold1_source_center5_epoch60_20260909' if 'center5' in cid else 'artifacts/fold1_eval_center1_epoch60');bx,sc,_=load_boxes(folder/f'{cid}_boxes.pkl');pool[cid]={r[0] for r in select_candidates(bx,sc)}
 deployed=np.array([rec[i]['augmentation']==0 and rec[i]['original_index'] in pool[rec[i]['case_id']] for i in dv]);pp=np.array(ps);positive=deployed&(yy==1);threshold=float(np.nextafter(pp[positive].min(),0.));write_json(RUN/'model/THRESHOLD.json',{'threshold':threshold,'source_only':True,'MR40_used':False,'selection':'minimum mean of3 source-dev positive probabilities in original deployment pool','source_positive':int(positive.sum()),'source_negative':int((deployed&(yy==0)).sum()),'source_negative_rejection':float(np.mean(pp[deployed&(yy==0)]<threshold))});write_json(RUN/'model/development_predictions.json',{'rows':dv,'p':ps,'y':yy.tolist(),'deployed':deployed.tolist()});del model;torch.cuda.empty_cache();model=combine(final);torch.save({'state_dict':model.state_dict(),'architecture':'E14_mean3','members':3,'epochs_per_member':25},RUN/'model/final_last.pt');write_json(RUN/'model/LOCKED.json',{'model_sha256':sha256_file(RUN/'model/final_last.pt'),'threshold_sha256':sha256_file(RUN/'model/THRESHOLD.json'),'member_final_sha256':[sha256_file(path) for path in final],'member_development_sha256':[sha256_file(path) for path in development],'config_sha256':sha256_file(RUN/'config.json'),'code_sha256':sha256_file(Path(__file__)),'architecture_sha256':sha256_file(Path(__file__).with_name('common.py')),'source':json.loads((PARENT/'features/READY.json').read_text()),'all_new_members_formal_training_complete':True});print('FULL_FORMAL_TRAINING_COMPLETE',flush=True)
if __name__=='__main__':main()
