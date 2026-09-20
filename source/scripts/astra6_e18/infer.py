"""One controlled C-only integration of completed E07 C with current E17 S/E14 F."""
import json,shutil,subprocess,sys,time
import numpy as np,SimpleITK as sitk
from scripts.astra6_e01.e01_common import P,DATA,sha256_file,sha256_tree,write_json

RUN=P/'artifacts/astra6_e18_MR_hierarchical_C_with_current_F_20260909'
BASE=P/'artifacts/astra6_e17_MR_detector_crop_segmentation_20260909'
C=P/'artifacts/astra6_e07_hierarchical_location_20260909'
F=P/'artifacts/astra6_e14_MR_filter_seed_stability_20260909'

def main():
    for name in ['evaluation','model','logs','predictions/mr_center2_k05']:(RUN/name).mkdir(parents=True,exist_ok=True)
    config={'experiment':'E18','hypothesis':'The prior hierarchical C gained location recall but lost precision before stronger F; removing other false positives can alter the class-wise precision tradeoff of the same C predictions','change':'Only replace E02 C with completed E07 C; retain exact E17 S/E14 F/D, all thresholds, candidate ordering and no-refill policy','single_fixed_combination':True,'no_new_training_required':'E07 parent and child classifiers already completed fixed512trees each; E17 S and E14 F full formal training complete','comparison':'E17 MR40, repeatedly observed development cohort, not a blind test','adoption':'All six official metrics no worse and at least one better, retain>=53matched lesions and>=36correct locations','budget_hours':2,'no_GT_read_during_inference':True,'no_threshold_or_seed_search':True}
    write_json(RUN/'config.json',config)
    lock=json.loads((C/'model/CLASSIFIER_LOCKED.json').read_text());assert sha256_file(C/'model/classifier.joblib')==lock['sha256']
    original=[json.loads(s) for s in (BASE/'candidate_predictions.jsonl').read_text().splitlines()];hier={(r['case_id'],r['original_index']):r for r in map(json.loads,(C/'candidate_predictions.jsonl').read_text().splitlines())};rows=[];changed=set()
    for r in original:
        rr=hier[r['case_id'],r['original_index']];assert r['score']==rr['score'] and r['low']==rr['low'] and r['high']==rr['high'];new={**r,'E17_class':r['predicted_class_id'],'predicted_class_id':rr['predicted_class_id']};rows.append(new)
        if r['filter_keep'] and new['predicted_class_id']!=r['predicted_class_id']:changed.add(r['case_id'])
    write_json(RUN/'INPUTS_LOCKED.json',{'classifier_sha256':lock['sha256'],'segmentation_sha256':sha256_file(BASE/'model/final_last.pt'),'filter_sha256':sha256_file(F/'model/final_last.pt'),'base_assignments_sha256':sha256_file(BASE/'candidate_predictions.jsonl'),'changed_cases':sorted(changed)})
    ids=json.loads((P/'artifacts/astra6_e02_multiscale_vessel_signature_20260908T173916Z/eval_case_ids.json').read_text());start=time.monotonic();threshold=json.loads((F/'model/THRESHOLD.json').read_text())['threshold']
    for cid in ids:
        dest=RUN/f'predictions/mr_center2_k05/{cid}.nii.gz';ref=BASE/f'predictions/mr_center2_k05/{cid}.nii.gz'
        if cid in changed:
            cmd=[sys.executable,'-m','scripts.delivery.refine_with_filter','--image',str(DATA/f'images/{cid}_0000.nii.gz'),'--predicted-vessel',str(P/f'artifacts/ta36_mr_center2_output/{cid}.nii.gz'),'--boxes',str(P/f'artifacts/fold1_eval_center2_epoch60/{cid}_boxes.pkl'),'--classifier',str(C/'model/classifier.joblib'),'--segmentation',str(BASE/'model/final_last.pt'),'--fp-filter',str(F/'model/final_last.pt'),'--fp-threshold',str(threshold),'--modality','MR','--device','cpu','--output',str(dest)]
            with (RUN/f'logs/{cid}.log').open('w') as log:subprocess.run(cmd,cwd=P,check=True,stdout=log,stderr=subprocess.STDOUT)
            a=sitk.ReadImage(str(dest));b=sitk.ReadImage(str(ref));assert np.array_equal(sitk.GetArrayViewFromImage(a)>0,sitk.GetArrayViewFromImage(b)>0),cid
        else:shutil.copy2(ref,dest)
        print('E18_INFERENCE',cid,'C_changed',cid in changed,flush=True)
    (RUN/'candidate_predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n');write_json(RUN/'PREDICTIONS_LOCKED.json',{'prediction_tree_sha256':sha256_tree(RUN/'predictions'),'cases':ids,'changed_cases':sorted(changed),'GT_not_read':True,'foreground_identical_to_E17':True,'seconds':time.monotonic()-start})

if __name__=='__main__':main()
