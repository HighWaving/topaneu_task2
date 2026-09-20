import json
import numpy as np
from scripts.astra6_e07.train import RUN,SHAPE
from scripts.analysis.paired_segmentation_diagnostics import diagnose
from scripts.astra6_e01.e01_common import DATA,load_nifti,write_json
ids=json.loads((RUN/'frozen_assignment/eval_case_ids.json').read_text());results={}
for cid in ids:
 gt,aff,_=load_nifti(DATA/f'location_masks/{cid}.nii.gz');pr,pa,_=load_nifti(RUN/f'predictions/mr_center2_k05/{cid}.nii.gz');old,_,_=load_nifti(SHAPE/f'predictions/mr_center2_k05/{cid}.nii.gz');assert np.allclose(aff,pa,atol=1e-4) and np.array_equal(pr>0,old>0);results[cid]=diagnose(gt,pr,aff);print(cid,flush=True)
lesions=[{**l,'case_id':c} for c in ids for l in results[c]['lesions']];m=[l for l in lesions if l['matched']];summary={'GT_components':len(lesions),'matched':len(m),'recall':len(m)/len(lesions),'FP_per_case':sum(results[c]['FP'] for c in ids)/len(ids),'location_correct':sum(l['class_correct'] for l in m),'location_denominator':len(m),'matched_binary_Dice_mean':float(np.mean([l['binary_dice'] for l in m])),'size':{}}
for b in ['<=3','(3,5]','(5,7]','>7']:
 ls=[l for l in lesions if l['size_bin']==b];ms=[l for l in ls if l['matched']];summary['size'][b]={'total':len(ls),'matched':len(ms),'location_correct':sum(l['class_correct'] for l in ms)}
write_json(RUN/'evaluation/lesion_diagnostics.json',{'E07':summary,'E04':json.loads((SHAPE/'evaluation/extended_diagnostics.json').read_text())['versions']['E04'],'binary_foreground_identical_all40':True,'cases':results});print(json.dumps(summary,indent=2))
