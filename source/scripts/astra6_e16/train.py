from pathlib import Path
import json
import numpy as np,torch
from torch.utils.data import DataLoader
from scripts.astra6_e16.prepare import RUN,P
from scripts.astra6_e04.run_e04 import fit,Segmenter,SegData,SUPPORT,PRIOR,largest
from scripts.astra6_e01.e01_common import write_json,sha256_file
HELPER=P/'artifacts/astra6_e10_mask_contact_location_20260909/source_segmenter/model/final_last.pt'
def evaluate(weights,dv,rec):
 model=Segmenter().cuda();model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=False)['state_dict']);model.eval();scores=[]
 with torch.inference_mode():
  for x,y in DataLoader(SegData(RUN,dv),batch_size=32,num_workers=0):
   pp=model(x.cuda()).sigmoid().cpu().numpy()[:,0]*SUPPORT
   for prob,truth in zip(pp,y.numpy()[:,0]>0):
    mask=largest(prob>=.5)
    if not mask.any():mask=PRIOR>0
    scores.append(float(2*(mask&truth).sum()/max(1,mask.sum()+truth.sum())))
 del model;torch.cuda.empty_cache();ct=[score for score,i in zip(scores,dv) if '_ct_' in rec[i]['case_id']];return {'all_components':len(scores),'all_crop_Dice':float(np.mean(scores)),'CT_components':len(ct),'CT_crop_Dice':float(np.mean(ct)),'component_scores':scores}
def main():
 assert (RUN/'features/SOURCE_READY.json').exists()
 if (RUN/'model/LOCKED.json').exists():print('Full training already locked',flush=True);return
 torch.set_num_threads(4);split=json.loads((RUN/'source_split.json').read_text());tr,dv=split['train_rows'],split['development_rows'];rec=[json.loads(s) for s in (RUN/'features/train_records.jsonl').read_text().splitlines()];helper_lock=json.loads(HELPER.with_name('LOCKED.json').read_text());assert not {rec[i]['case_id'] for i in helper_lock['train_rows']}&set(split['development_cases'])
 net=Segmenter().cuda();x,y=next(iter(DataLoader(SegData(RUN,tr),batch_size=32)));z=net(x.cuda());loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y.cuda());assert torch.isfinite(loss) and x.shape[1:]==(2,32,32,32);loss.backward();write_json(RUN/'model/PREFLIGHT.json',{'finite_forward_backward':True,'batch_shape':list(x.shape),'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated()});del net,x,y,z,loss;torch.cuda.empty_cache()
 fit(RUN,'development',tr,dv,epochs=13);result={'E04_recipe_old_source_helper':evaluate(HELPER,dv,rec),'E16_new_source_helper':evaluate(RUN/'model/development_last.pt',dv,rec),'fixed_epochs':13,'source_only':True};write_json(RUN/'evaluation/source_validation.json',result);print('source_validation',json.dumps(result),flush=True);fit(RUN,'final',list(range(len(rec))),epochs=13);write_json(RUN/'model/LOCKED.json',{'model_sha256':sha256_file(RUN/'model/final_last.pt'),'config_sha256':sha256_file(RUN/'config.json'),'code_sha256':sha256_file(Path(__file__)),'shared_training_code_sha256':sha256_file(Path('scripts/astra6_e04/run_e04.py')),'source':json.loads((RUN/'features/SOURCE_READY.json').read_text()),'epochs':13,'steps_per_epoch':188,'total_updates':2444,'CT5_or_MR40_used_for_fit':False,'MR_deployment_unchanged':True});print('FULL_FORMAL_TRAINING_COMPLETE',flush=True)
if __name__=='__main__':main()
