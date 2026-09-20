"""Full-cohort lesion accounting, reusing only identical prediction/GT versions."""
import copy,json
import numpy as np
from scripts.astra6_e19.infer import RUN,BASE,P,DATA,sha256_file,write_json
from scripts.astra6_e01.e01_common import load_nifti
from scripts.analysis.paired_segmentation_diagnostics import diagnose

def main():
    old=json.loads((BASE/'evaluation/lesion_diagnostics.json').read_text());cases=copy.deepcopy(old['cases']);root=P/'artifacts/current_official_20260909';before={r['case_id']:r for r in json.loads((root/'E17/per_case.json').read_text())};after={r['case_id']:r for r in json.loads((root/'E19/per_case.json').read_text())};assert set(cases)==set(before)==set(after);changed=[]
    for cid in sorted(cases):
        for k in ['gt','evaluator']:assert before[cid]['hashes'][k]==after[cid]['hashes'][k]
        new=RUN/f'predictions/mr_center2_k05/{cid}.nii.gz';ref=BASE/f'predictions/mr_center2_k05/{cid}.nii.gz'
        if sha256_file(new)==sha256_file(ref):continue
        gt,aff,_=load_nifti(DATA/f'location_masks/{cid}.nii.gz');pred,pa,_=load_nifti(new);assert np.allclose(aff,pa,atol=1e-4);cases[cid]=diagnose(gt,pred,aff);changed.append(cid);print('E19_DIAGNOSE',cid,flush=True)
    lesions=[r for c in cases.values() for r in c['lesions']];matched=[r for r in lesions if r['matched']];summary={'GT_components':len(lesions),'matched':len(matched),'recall':len(matched)/len(lesions),'location_correct':sum(r['class_correct'] for r in matched),'location_denominator':len(matched),'FP_per_case':sum(c['FP'] for c in cases.values())/len(cases),'matched_binary_Dice_mean':float(np.mean([r['binary_dice'] for r in matched])) if matched else 0.,'size':{}}
    for k in ['<=3','(3,5]','(5,7]','>7']:
        ls=[r for r in lesions if r['size_bin']==k];ms=[r for r in ls if r['matched']];summary['size'][k]={'total':len(ls),'matched':len(ms),'location_correct':sum(r['class_correct'] for r in ms)}
    write_json(RUN/'evaluation/lesion_diagnostics.json',{'E19':summary,'E17':old['E17'],'cases':cases,'changed_cases':changed,'unchanged_reuse':'prediction bytes, GT hash and evaluator hash all identical'})
    comparison=json.loads((RUN/'evaluation/paired_official_comparison.json').read_text());delta=comparison['delta'];pareto=all(delta[k]>=-1e-10 for k in ['PRECISION','RECALL','MCC','DICE','VOLSIM']) and delta['HD95']<=1e-10 and any(abs(v)>1e-8 for v in delta.values());adopt=pareto and summary['matched']>=53 and summary['location_correct']>=36
    write_json(RUN/'ADOPTION.json',{'adopt':adopt,'official_pareto_gate':pareto,'official_delta':delta,'lesion_gate':summary,'source_threshold_unchanged':True});print('E19_RESULT',summary,'ADOPT',adopt,flush=True)

if __name__=='__main__':main()
