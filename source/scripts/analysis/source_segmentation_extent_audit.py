"""Quantify source-only GT truncation imposed by writing S only inside D boxes."""
import json
import numpy as np
from scripts.astra6_e01.e01_common import P,DATA,load_nifti,write_json
from scripts.astra6_e04.run_e04 import component_records_fast

def main():
    src=P/'artifacts/astra6_e17_MR_detector_crop_segmentation_20260909';out=P/'artifacts/source_segmentation_extent_audit_20260909';out.mkdir(exist_ok=True)
    config={'hypothesis':'Current S targets and native output are clipped to the detector box even though the input contains twice its width; this may make full-lesion recovery impossible','precondition':'On31 same source lesions, at least6 lose>=10percent GT voxels outside native detector bounds and extending to the existing input context raises the mean oracle Dice bound by>=.02','next_if_pass':'Train the same32cube S on full-component targets across existing2x context; change native output extent consistently; fixed13development+13full epochs, no new network architecture or F/C/D threshold changes','no_MR40_or_CT5_access':True,'source_detector_training_exposure_disclosed':True}
    write_json(out/'config.json',config)
    rec=[json.loads(s) for s in (src/'features/train_records.jsonl').read_text().splitlines()];pair=json.loads((src/'evaluation/SOURCE_PRECONDITION.json').read_text());groups={}
    for gi,di in zip(pair['paired_GT_rows'],pair['paired_detector_rows']):groups.setdefault(rec[di]['case_id'],[]).append((gi,di))
    rows=[]
    for cid,items in sorted(groups.items()):
        gt,aff,shape=load_nifti(DATA/f'location_masks/{cid}.nii.gz');comps=component_records_fast(gt)
        for gi,di in items:
            r,g=rec[di],rec[gi];lo,hi=np.array(r['low']),np.array(r['high']);matches=[c for c in comps if c['class_id']==g['class_id'] and np.array_equal(c['coords'].min(0)-.5,g['low']) and np.array_equal(c['coords'].max(0)+.5,g['high'])];assert len(matches)==1,(cid,gi);coords=matches[0]['coords'];center=(lo+hi)/2;width=np.maximum(hi-lo,1)
            fractions={}
            for name,low,high in [('detector',lo,hi),('context',center-width,center+width)]:fractions[name]=float(np.mean(np.all((coords>=np.floor(low))&(coords<np.ceil(high)),axis=1)))
            rows.append({'case_id':cid,'GT_row':gi,'detector_row':di,'GT_voxels':len(coords),'detector_fraction':fractions['detector'],'context_fraction':fractions['context'],'detector_oracle_Dice_bound':2*fractions['detector']/(1+fractions['detector']),'context_oracle_Dice_bound':2*fractions['context']/(1+fractions['context'])})
        print('SOURCE_EXTENT_AUDIT',cid,flush=True)
    old=float(np.mean([r['detector_oracle_Dice_bound'] for r in rows]));new=float(np.mean([r['context_oracle_Dice_bound'] for r in rows]));limited=sum(r['detector_fraction']<=.9 for r in rows)
    report={'n':len(rows),'at_least10percent_GT_truncated':limited,'mean_detector_oracle_Dice_bound':old,'mean_context_oracle_Dice_bound':new,'oracle_Dice_bound_gain':new-old,'precondition_passed':limited>=6 and new-old>=.02,'rows':rows,'interpretation':'Oracle upper bounds expose geometric restrictions; they are not measured model quality and do not guarantee gains from a larger output extent.'};write_json(out/'RESULT.json',report);print('SOURCE_EXTENT_RESULT',{k:v for k,v in report.items() if k!='rows'},flush=True)

if __name__=='__main__':main()
