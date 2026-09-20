import json
import numpy as np
from scripts.astra6_e09.prepare import RUN,P,DATA
from scripts.astra6_e01.e01_common import load_nifti,write_json
from scripts.analysis.paired_segmentation_diagnostics import diagnose
ids=json.loads((RUN/'source_split.json').read_text())['fixed_CT5'];cases={}
for cid in ids:
 gt,aff,_=load_nifti(DATA/f'location_masks/{cid}.nii.gz');old,_,_=load_nifti(P/f'artifacts/ct_independent5_20260909/E04/{cid}.nii.gz');new,pa,_=load_nifti(RUN/f'predictions_ct/{cid}.nii.gz');assert np.allclose(aff,pa,atol=1e-4) and np.array_equal(old>0,new>0);cases[cid]={'E04':diagnose(gt,old,aff),'E09':diagnose(gt,new,aff)};print(cid,flush=True)
out={}
for v in ['E04','E09']:
 lesions=[r for cid in ids for r in cases[cid][v]['lesions']];m=[r for r in lesions if r['matched']];out[v]={'GT':len(lesions),'matched':len(m),'location_correct':sum(r['class_correct'] for r in m),'FP_per_case':sum(cases[c][v]['FP'] for c in ids)/5,'mean_matched_binary_Dice':float(np.mean([r['binary_dice'] for r in m])) if m else None}
write_json(RUN/'evaluation/lesion_diagnostics.json',{'summary':out,'binary_foreground_identical':True,'cases':cases});print(json.dumps(out,indent=2),flush=True)
