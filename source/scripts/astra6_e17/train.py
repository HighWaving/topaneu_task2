from pathlib import Path
import json,math
import torch
from scripts.astra6_e17.prepare import RUN,P,evaluate
from scripts.astra6_e04.run_e04 import fit
from scripts.astra6_e01.e01_common import write_json,sha256_file
FROZEN=P/'artifacts/astra6_e14_MR_filter_seed_stability_20260909'
def main():
 assert (RUN/'features/SOURCE_READY.json').exists();pre=json.loads((RUN/'evaluation/SOURCE_PRECONDITION.json').read_text());assert pre['paired_same_source_lesions'] and pre['proceed_to_training'];torch.set_num_threads(4)
 if (RUN/'model/LOCKED.json').exists():print('Training already complete',flush=True);return
 split=json.loads((RUN/'source_split.json').read_text());rec=[json.loads(s) for s in (RUN/'features/train_records.jsonl').read_text().splitlines()];tr,dv=split['train_rows'],split['development_detector_rows'];assert not {rec[i]['case_id'] for i in tr}&set(split['development_cases']);fit(RUN,'development',tr,dv,epochs=13);new={'GT_boxes':evaluate(RUN/'model/development_last.pt',split['development_GT_rows']),'detector_boxes':evaluate(RUN/'model/development_last.pt',dv)};write_json(RUN/'evaluation/source_validation.json',{'baseline':pre['baseline'],'new':new,'paired_same_source_lesions':True,'fixed_epochs':13});print('source_validation',json.dumps({k:v['crop_Dice_mean'] for k,v in new.items()}),flush=True);fit(RUN,'final',list(range(len(rec))),epochs=13);write_json(RUN/'model/LOCKED.json',{'model_sha256':sha256_file(RUN/'model/final_last.pt'),'source':json.loads((RUN/'features/SOURCE_READY.json').read_text()),'config_sha256':sha256_file(RUN/'config.json'),'code_sha256':sha256_file(Path(__file__)),'shared_training_sha256':sha256_file(Path('scripts/astra6_e04/run_e04.py')),'epochs':13,'total_updates':13*math.ceil(len(rec)/32),'frozen_filter_assignments_sha256':sha256_file(FROZEN/'candidate_predictions.jsonl'),'no_MR40_or_CT5_fit':True});print('FULL_FORMAL_TRAINING_COMPLETE',flush=True)
if __name__=='__main__':main()
