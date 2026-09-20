"""Check CT source detector-box clipping separately from the completed MR audit."""
import json
import numpy as np
from scripts.astra6_e01.e01_common import P,DATA,load_nifti,load_nifti_geometry,load_boxes,select_candidates,write_json
from scripts.astra6_e04.run_e04 import component_records_fast

def main():
    src=P/'artifacts/astra6_e16_CT_expanded_segmentation_20260909';out=P/'artifacts/source_CT_segmentation_extent_audit_20260909';out.mkdir(exist_ok=True)
    config={'hypothesis':'CT aneurysm size and detector localization may impose a different S output-box ceiling from MR','source':'fixed E16 source-development CT components only; CT5 excluded','precondition':'at least20 paired source CT lesions, at least6 lose>=10percent GT outside native D bounds, and2x context raises mean oracleDice bound by>=.02','next_if_pass':'Same E16 image/case bank and network, retarget full existing2x input context and extend output consistently; fixed13development+13finalepochs; CT-only comparison to CT_E16','no_CT5_or_MR40_access':True,'limitation':'Source detector training exposure persists; oracle bounds are not measured S performance'};write_json(out/'config.json',config)
    split=json.loads((src/'source_split.json').read_text());rec=[json.loads(s) for s in (src/'features/train_records.jsonl').read_text().splitlines()];groups={}
    for i in split['development_rows']:
        if '_ct_' in rec[i]['case_id']:groups.setdefault(rec[i]['case_id'],[]).append(i)
    assert sum(map(len,groups.values()))==30;rows=[];unmatched=[]
    for cid,indices in sorted(groups.items()):
        assert cid not in split['fixed_CT5'];gt,aff,shape=load_nifti(DATA/f'location_masks/{cid}.nii.gz');ia,ish=load_nifti_geometry(DATA/f'images/{cid}_0000.nii.gz');assert shape==ish and np.allclose(aff,ia,atol=1e-4);comps=component_records_fast(gt);bx,sc,_=load_boxes(P/f'artifacts/ct_fold2_all109_boxes/{cid}_boxes.pkl');primary={}
        for idx,score,low,high in select_candidates(bx,sc):
            choices=[]
            for comp in comps:
                coords=comp['coords'];fraction=float(np.mean(np.all((coords>=low)&(coords<high),axis=1)))
                if fraction<.1:continue
                lo,hi=coords.min(0)-.5,coords.max(0)+.5;inter=np.maximum(0,np.minimum(high,hi)-np.maximum(low,lo)).prod();iou=inter/max(1e-12,(high-low).prod()+(hi-lo).prod()-inter);choices.append((iou,comp))
            if not choices:continue
            comp=max(choices,key=lambda r:r[0])[1];key=(comp['class_id'],comp['component_id'])
            if key not in primary or score>primary[key][1]:primary[key]=(idx,score,low,high)
        for i in indices:
            r=rec[i];matches=[c for c in comps if c['class_id']==r['class_id'] and np.array_equal(c['coords'].min(0)-.5,r['low']) and np.array_equal(c['coords'].max(0)+.5,r['high'])];assert len(matches)==1,(cid,i);comp=matches[0];key=(comp['class_id'],comp['component_id'])
            if key not in primary:unmatched.append({'case_id':cid,'GT_row':i});continue
            idx,score,low,high=primary[key];coords=comp['coords'];center=(low+high)/2;width=np.maximum(high-low,1);fractions={name:float(np.mean(np.all((coords>=np.floor(lo))&(coords<np.ceil(hi)),axis=1))) for name,lo,hi in [('detector',low,high),('context',center-width,center+width)]}
            rows.append({'case_id':cid,'GT_row':i,'detector_index':idx,'score':score,'low':low.tolist(),'high':high.tolist(),'detector_fraction':fractions['detector'],'context_fraction':fractions['context'],'detector_oracle_Dice_bound':2*fractions['detector']/(1+fractions['detector']),'context_oracle_Dice_bound':2*fractions['context']/(1+fractions['context'])})
        print('SOURCE_CT_EXTENT',cid,flush=True)
    old=float(np.mean([r['detector_oracle_Dice_bound'] for r in rows]));new=float(np.mean([r['context_oracle_Dice_bound'] for r in rows]));limited=sum(r['detector_fraction']<=.9 for r in rows);result={'paired_lesions':len(rows),'source_GT_components':30,'unmatched_source_lesions':unmatched,'at_least10percent_GT_truncated':limited,'mean_detector_oracle_Dice_bound':old,'mean_context_oracle_Dice_bound':new,'oracle_gain':new-old,'precondition_passed':len(rows)>=20 and limited>=6 and new-old>=.02,'rows':rows};write_json(out/'RESULT.json',result);print('SOURCE_CT_EXTENT_RESULT',{k:v for k,v in result.items() if k!='rows'},flush=True)

if __name__=='__main__':main()
