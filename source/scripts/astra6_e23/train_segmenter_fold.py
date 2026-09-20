"""Train source-only shape helper for OOF MR candidates; no warm start leakage."""
import argparse,json,math
import torch
from scripts.astra6_e23.prepare import RUN,P,group
from scripts.astra6_e04.run_e04 import fit
from scripts.astra6_e01.e01_common import sha256_file,write_json
BASE=P/'artifacts/astra6_e16_CT_expanded_segmentation_20260909'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,choices=[0,1],required=True);a=ap.parse_args();torch.set_num_threads(4);out=RUN/f'segmenters/fold{a.fold}';(out/'model').mkdir(parents=True,exist_ok=True)
 if (out/'model/LOCKED.json').exists():return
 if not (out/'features').exists():(out/'features').symlink_to(BASE/'features',target_is_directory=True)
 detector_complete=json.loads((RUN/f'checkpoints/fold{a.fold}_TRAINED.json').read_text());assert detector_complete['global_step']>=45000, 'Do not consume a detector before its full45000-update budget'
 split=json.loads((RUN/'source_split.json').read_text())['folds'][a.fold];allowed=set(split['train']);excluded={group(c) for c in split['val']};records=[json.loads(s) for s in (BASE/'features/train_records.jsonl').read_text().splitlines()];indices=[i for i,r in enumerate(records) if '_ct_' in r['case_id'] or r['case_id'] in allowed];cases=sorted({records[i]['case_id'] for i in indices});assert not {group(c) for c in cases}&excluded;assert all('center2_mr' not in c for c in cases);ct5=set(json.loads((BASE/'source_split.json').read_text())['fixed_CT5']);assert not set(cases)&ct5
 write_json(out/'source_split.json',{'training_rows':indices,'training_cases':cases,'MR_heldout_cases':split['val'],'epochs':13,'initialization':'from scratch; never E16 weights','source_arrays':json.loads((BASE/'features/SOURCE_READY.json').read_text())});fit(out,'final',indices,epochs=13);write_json(out/'model/LOCKED.json',{'sha256':sha256_file(out/'model/final_last.pt'),'epochs':13,'updates':13*math.ceil(len(indices)/32),'source_split_sha256':sha256_file(out/'source_split.json'),'MR_validation_cases_never_fit':True});print('E23_OOF_SHAPE_HELPER_COMPLETE',a.fold,flush=True)
if __name__=='__main__':main()
