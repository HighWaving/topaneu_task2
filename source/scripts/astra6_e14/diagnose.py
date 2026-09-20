import json
import numpy as np
from scripts.astra6_e14.common import RUN
from scripts.astra6_e01.e01_common import P,DATA,load_nifti,write_json
from scripts.analysis.paired_segmentation_diagnostics import diagnose
SHAPE=P/'artifacts/astra6_e04_crop_segmentation_20260909T031146Z';old=json.loads((SHAPE/'evaluation/extended_diagnostics.json').read_text());ids=sorted(old['cases']);results={};candidate_rows=[json.loads(s) for s in (RUN/'candidate_predictions.jsonl').read_text().splitlines()]
for cid in ids:
 gt,aff,_=load_nifti(DATA/f'location_masks/{cid}.nii.gz');pr,pa,_=load_nifti(RUN/f'predictions/mr_center2_k05/{cid}.nii.gz');base,_,_=load_nifti(SHAPE/f'predictions/mr_center2_k05/{cid}.nii.gz');assert np.allclose(aff,pa,atol=1e-4);assert not np.any((pr>0)&(base==0));results[cid]=diagnose(gt,pr,aff);print(cid,flush=True)
lesions=[{**r,'case_id':cid} for cid in ids for r in results[cid]['lesions']];m=[r for r in lesions if r['matched']];summary={'GT_components':len(lesions),'matched':len(m),'recall':len(m)/len(lesions),'FP_per_case':sum(results[c]['FP'] for c in ids)/len(ids),'location_correct':sum(r['class_correct'] for r in m),'location_denominator':len(m),'matched_binary_Dice_mean':float(np.mean([r['binary_dice'] for r in m])) if m else 0,'size':{}}
for b in ['<=3','(3,5]','(5,7]','>7']:
 ls=[r for r in lesions if r['size_bin']==b];ms=[r for r in ls if r['matched']];summary['size'][b]={'total':len(ls),'matched':len(ms),'location_correct':sum(r['class_correct'] for r in ms)}
write_json(RUN/'evaluation/lesion_diagnostics.json',{'E14':summary,'E04':old['versions']['E04'],'selected':len(candidate_rows),'kept':sum(r['filter_keep'] for r in candidate_rows),'cases':results});print(json.dumps(summary,indent=2),flush=True)
