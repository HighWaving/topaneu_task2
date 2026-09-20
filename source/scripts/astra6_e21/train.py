"""Fixed13epoch development, native Dice selection, full selected-duration fit."""
import json,math,os
from pathlib import Path
import torch,numpy as np
from scipy.ndimage import map_coordinates
from scripts.astra6_e21.prepare import RUN,P
from scripts.astra6_e04.run_e04 import fit,SegData,largest,Segmenter
from scripts.astra6_e01.e01_common import sha256_file,write_json

def evaluate(model,targets,x):
 model.eval()
 with torch.inference_mode():probs=model(x.cuda()).sigmoid().cpu().numpy()[:,0]
 scores=[]
 for prob,t in zip(probs,targets):
  fg=largest(map_coordinates(prob,t['coords'].reshape(3,-1),order=1,mode='constant',cval=0,prefilter=False).reshape(t['truth'].shape)>=.5)
  if not fg.any():fg=t['ellipse']
  scores.append(float(2*(fg&t['truth']).sum()/max(1,fg.sum()+int(t['GT_voxels']))))
 return scores

def main():
 torch.set_num_threads(4);assert (RUN/'native_validation/READY.json').exists()
 if (RUN/'model/LOCKED.json').exists():return
 split=json.loads((RUN/'source_split.json').read_text());records=[json.loads(s) for s in (RUN/'features/train_records.jsonl').read_text().splitlines()];targets=[dict(np.load(RUN/f'native_validation/{j:02d}.npz')) for j in range(28)];dataset=SegData(RUN,split['development_detector_rows']);x=torch.stack([dataset[i][0] for i in range(28)]);path=RUN/'model/native_history.json';history=json.loads(path.read_text()) if path.exists() else []
 reference=Segmenter().cuda();reference.load_state_dict(torch.load(P/'artifacts/astra6_e20_CT_detector_crop_segmentation_20260909/model/development_last.pt',map_location='cpu',weights_only=False)['state_dict']);reference_score=float(np.mean(evaluate(reference,targets,x)));assert abs(reference_score-.8114654824539228)<1e-7;write_json(RUN/'native_validation/DECODER_EQUIVALENCE.json',{'E20_native_Dice':reference_score,'matches_independent_source_audit':True});del reference;torch.cuda.empty_cache()
 for epoch in range(len(history)+1,14):
  model=fit(RUN,'development',split['train_rows'],epochs=epoch);scores=evaluate(model,targets,x);value=float(np.mean(scores));best_before=max([r['native_Dice'] for r in history],default=-1)
  if value>best_before:
   temp=RUN/'model/development_native_best.tmp';torch.save({'state_dict':model.state_dict(),'epoch':epoch,'source_native_Dice':value},temp);os.replace(temp,RUN/'model/development_native_best.pt')
  history.append({'epoch':epoch,'native_Dice':value,'scores':scores});write_json(path,history);del model;torch.cuda.empty_cache();print('E21_NATIVE_VALIDATION',epoch,value,flush=True)
 selected=max(history,key=lambda r:r['native_Dice']);passed=selected['native_Dice']>=.8208194051141334+.005;write_json(RUN/'evaluation/source_validation.json',{'baseline_E16_native_Dice':.8208194051141334,'selected_epoch':selected['epoch'],'selected_native_Dice':selected['native_Dice'],'source_gate_passed':passed,'full_development_epochs':13,'selection_source_only':True});write_json(RUN/'model/SELECTED_DURATION.json',selected)
 fit(RUN,'final',list(range(len(records))),epochs=selected['epoch'])
 write_json(RUN/'model/LOCKED.json',{'model_sha256':sha256_file(RUN/'model/final_last.pt'),'epochs':selected['epoch'],'development_epochs_completed':13,'total_updates':selected['epoch']*math.ceil(len(records)/32),'source_gate_passed':passed,'config_sha256':sha256_file(RUN/'config.json'),'source_split_sha256':sha256_file(RUN/'source_split.json'),'source_data':json.loads((RUN/'features/SOURCE_READY.json').read_text()),'code_sha256':sha256_file(Path(__file__)),'no_CT5_or_MR40_fit':True});print('E21_FULL_FORMAL_TRAINING_COMPLETE',flush=True)
if __name__=='__main__':main()
