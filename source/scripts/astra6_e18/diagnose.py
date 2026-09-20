"""Recompute changed-case lesions; reuse only byte-identical E17 predictions."""
import copy,json
import numpy as np
from scripts.astra6_e18.infer import RUN,BASE,DATA,sha256_file,write_json
from scripts.astra6_e01.e01_common import load_nifti
from scripts.analysis.paired_segmentation_diagnostics import diagnose

def main():
    lock=json.loads((RUN/'PREDICTIONS_LOCKED.json').read_text());changed=set(lock['changed_cases']);old=json.loads((BASE/'evaluation/lesion_diagnostics.json').read_text());cases=copy.deepcopy(old['cases']);candidates=[json.loads(s) for s in (RUN/'candidate_predictions.jsonl').read_text().splitlines()]
    for cid in lock['cases']:
        new=RUN/f'predictions/mr_center2_k05/{cid}.nii.gz';ref=BASE/f'predictions/mr_center2_k05/{cid}.nii.gz'
        if cid not in changed:assert sha256_file(new)==sha256_file(ref);continue
        ledger=json.loads(new.with_suffix('.json').read_text());expected={r['original_index']:r for r in candidates if r['case_id']==cid}
        assert all(r['keep']==expected[r['index']]['filter_keep'] for r in ledger['filter_decisions'])
        assert all(r['class']==expected[r['index']]['predicted_class_id'] for r in ledger['candidates'])
        gt,aff,_=load_nifti(DATA/f'location_masks/{cid}.nii.gz');pred,pa,_=load_nifti(new);assert np.allclose(aff,pa,atol=1e-4);cases[cid]=diagnose(gt,pred,aff);print('E18_DIAGNOSE',cid,flush=True)
    lesions=[r for c in cases.values() for r in c['lesions']];matched=[r for r in lesions if r['matched']];summary={'GT_components':len(lesions),'matched':len(matched),'recall':len(matched)/len(lesions),'location_correct':sum(r['class_correct'] for r in matched),'location_denominator':len(matched),'FP_per_case':sum(c['FP'] for c in cases.values())/len(cases),'matched_binary_Dice_mean':float(np.mean([r['binary_dice'] for r in matched])),'size':{}}
    for k in ['<=3','(3,5]','(5,7]','>7']:
        ls=[r for r in lesions if r['size_bin']==k];ms=[r for r in ls if r['matched']];summary['size'][k]={'total':len(ls),'matched':len(ms),'location_correct':sum(r['class_correct'] for r in ms)}
    write_json(RUN/'evaluation/lesion_diagnostics.json',{'E18':summary,'E17':old['E17'],'cases':cases,'reuse_rule':'unchanged prediction bytes and same GT/evaluator; three changed cases independently recomputed'})
    comparison=json.loads((RUN/'evaluation/paired_official_comparison.json').read_text());delta=comparison['delta'];pareto=all(delta[k]>=-1e-10 for k in ['PRECISION','RECALL','MCC','DICE','VOLSIM']) and delta['HD95']<=1e-10 and any(abs(v)>1e-8 for v in delta.values());adopt=pareto and summary['matched']>=53 and summary['location_correct']>=36
    write_json(RUN/'ADOPTION.json',{'adopt':adopt,'official_pareto_gate':pareto,'official_delta':delta,'lesion_gate':summary,'training_already_complete':'E07 C/E17 S/E14 F','no_threshold_changes':True});print('E18_RESULT',summary,'ADOPT',adopt,flush=True)

if __name__=='__main__':main()
