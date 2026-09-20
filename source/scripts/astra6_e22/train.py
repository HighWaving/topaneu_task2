"""CT-only formal training with fixed compute budget and source-native selection."""
import json,math,os
from pathlib import Path
import numpy as np,torch
from scripts.astra6_e22.prepare import RUN,P,DVALID
from scripts.astra6_e21.train import evaluate
from scripts.astra6_e04.run_e04 import fit,SegData,Segmenter
from scripts.astra6_e01.e01_common import write_json,sha256_file

def main():
 torch.set_num_threads(4)
 if (RUN/'model/LOCKED.json').exists():return
 split=json.loads((RUN/'source_split.json').read_text());config=json.loads((RUN/'config.json').read_text());targets=[dict(np.load(RUN/f'native_validation/{j:02d}.npz')) for j in range(28)];ds=SegData(DVALID,split['source_native_validation_rows_in_E20']);x=torch.stack([ds[i][0] for i in range(28)]);path=RUN/'model/native_history.json';history=json.loads(path.read_text()) if path.exists() else []
 for epoch in range(len(history)+1,config['development_epochs']+1):
  model=fit(RUN,'development',split['train_rows'],epochs=epoch);scores=evaluate(model,targets,x);value=float(np.mean(scores));previous=max([r['native_Dice'] for r in history],default=-1)
  if value>previous:
   temp=RUN/'model/development_native_best.tmp';torch.save({'state_dict':model.state_dict(),'epoch':epoch,'source_native_Dice':value},temp);os.replace(temp,RUN/'model/development_native_best.pt')
  history.append({'epoch':epoch,'native_Dice':value,'scores':scores});write_json(path,history);del model;torch.cuda.empty_cache();print('E22_NATIVE_VALIDATION',epoch,value,flush=True)
 selected=max(history,key=lambda r:r['native_Dice']);passed=selected['native_Dice']>=.8258194051141334;write_json(RUN/'evaluation/source_validation.json',{'baseline_E16_native_Dice':.8208194051141334,'selected_epoch':selected['epoch'],'selected_native_Dice':selected['native_Dice'],'source_gate_passed':passed,'full_development_epochs':config['development_epochs'],'selection_source_only':True});write_json(RUN/'model/SELECTED_DURATION.json',selected)
 fit(RUN,'final',split['final_rows'],epochs=selected['epoch'])
 write_json(RUN/'model/LOCKED.json',{'model_sha256':sha256_file(RUN/'model/final_last.pt'),'epochs':selected['epoch'],'development_epochs_completed':len(history),'total_updates':selected['epoch']*math.ceil(len(split['final_rows'])/32),'source_gate_passed':passed,'config_sha256':sha256_file(RUN/'config.json'),'source_split_sha256':sha256_file(RUN/'source_split.json'),'final_training_rows':len(split['final_rows']),'final_training_cases':split['final_cases'],'training_modality':'CT_only','code_sha256':sha256_file(Path(__file__)),'no_CT5_or_MR40_fit':True});print('E22_FULL_FORMAL_TRAINING_COMPLETE',flush=True)
if __name__=='__main__':main()
