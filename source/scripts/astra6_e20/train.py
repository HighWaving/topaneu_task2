"""Fixed-budget, resumable source-only CT detector crop training."""
import json,math,time
from pathlib import Path
import torch
from scripts.astra6_e20.prepare import RUN,P,AUDIT
from scripts.astra6_e04.run_e04 import fit
from scripts.astra6_e05.run_e05 import group
from scripts.astra6_e01.e01_common import write_json,sha256_file
import scripts.astra6_e17.prepare as scoring

def main():
    assert (RUN/'features/SOURCE_READY.json').exists()
    start=time.monotonic();ready=P/'artifacts/astra6_e19_MR_anatomical_fp_filter_20260909/TA36_READY.json'
    while not ready.exists():
        assert time.monotonic()-start<21600,'GPU dependency exceeded six hours'
        time.sleep(10)
    torch.set_num_threads(4)
    if (RUN/'model/LOCKED.json').exists():return
    split=json.loads((RUN/'source_split.json').read_text());records=[json.loads(s) for s in (RUN/'features/train_records.jsonl').read_text().splitlines()];tr=split['train_rows'];dv=split['development_detector_rows'];assert len(dv)==28
    assert not {group(records[i]['case_id']) for i in tr}&{group(c) for c in split['development_cases']}
    fit(RUN,'development',tr,dv,epochs=13)
    scoring.RUN=RUN
    result={'GT_boxes':scoring.evaluate(RUN/'model/development_last.pt',split['development_GT_paired_rows']),'detector_boxes':scoring.evaluate(RUN/'model/development_last.pt',dv)}
    pre=json.loads((AUDIT/'RESULT.json').read_text());passed=result['detector_boxes']['crop_Dice_mean']>=pre['scores']['detector']['mean_Dice']
    write_json(RUN/'evaluation/source_validation.json',{'baseline':pre['scores'],'new':result,'source_gate_passed':passed,'paired_same_source_lesions':True,'fixed_epochs':13})
    print('SOURCE_VALIDATION',json.dumps(result),'PASSED',passed,flush=True)
    fit(RUN,'final',list(range(len(records))),epochs=13)
    write_json(RUN/'model/LOCKED.json',{'model_sha256':sha256_file(RUN/'model/final_last.pt'),'epochs':13,'total_updates':13*math.ceil(len(records)/32),'source_gate_passed':passed,'source':json.loads((RUN/'features/SOURCE_READY.json').read_text()),'config_sha256':sha256_file(RUN/'config.json'),'code_sha256':sha256_file(Path(__file__)),'shared_training_sha256':sha256_file(P/'scripts/astra6_e04/run_e04.py'),'no_CT5_or_MR40_fit':True})
    print('E20_FULL_FORMAL_TRAINING_COMPLETE',flush=True)
if __name__=='__main__':main()
