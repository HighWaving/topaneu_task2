import json
import numpy as np
from scripts.astra6_e21.prepare import RUN,P,DATA
from scripts.astra6_e01.e01_common import load_nifti,write_json
from scripts.analysis.paired_segmentation_diagnostics import diagnose
ids=json.loads((RUN/'source_split.json').read_text())['fixed_CT5'];cases={}
for cid in ids:
 gt,aff,_=load_nifti(DATA/f'location_masks/{cid}.nii.gz');old,_,_=load_nifti(P/f'artifacts/astra6_e16_CT_expanded_segmentation_20260909/predictions_ct/{cid}.nii.gz');new,pa,_=load_nifti(RUN/f'predictions_ct/{cid}.nii.gz');assert np.allclose(aff,pa,atol=1e-4);cases[cid]={'E16':diagnose(gt,old,aff),'E21':diagnose(gt,new,aff)};print(cid,flush=True)
out={}
for v in ['E16','E21']:
 lesions=[r for cid in ids for r in cases[cid][v]['lesions']];m=[r for r in lesions if r['matched']];out[v]={'GT':len(lesions),'matched':len(m),'location_correct':sum(r['class_correct'] for r in m),'FP_per_case':sum(cases[c][v]['FP'] for c in ids)/5,'mean_matched_binary_Dice':float(np.mean([r['binary_dice'] for r in m])) if m else None}
write_json(RUN/'evaluation/lesion_diagnostics.json',{'summary':out,'cases':cases});print(json.dumps(out,indent=2),flush=True)

comparison=json.loads((RUN/'evaluation/paired_official_comparison.json').read_text());delta=comparison['delta']
pareto=all(delta[k]>=-1e-10 for k in ['PRECISION','RECALL','MCC','DICE','VOLSIM']) and delta['HD95']<=1e-10 and any(abs(v)>1e-8 for v in delta.values())

size={}
for version in ['E16','E21']:
 lesions=[r for cid in ids for r in cases[cid][version]['lesions']];size[version]={}
 for key in ['<=3','(3,5]','(5,7]','>7']:
  ls=[r for r in lesions if r['size_bin']==key];ms=[r for r in ls if r['matched']]
  size[version][key]={'total':len(ls),'matched':len(ms),'location_correct':sum(r['class_correct'] for r in ms),'matched_Dice':float(np.mean([r['binary_dice'] for r in ms])) if ms else None}
write_json(RUN/'evaluation/size_diagnostics.json',size)
unchanged=[]
for cid in ids:
 old=json.loads((P/f'artifacts/astra6_e16_CT_expanded_segmentation_20260909/predictions_ct/{cid}.nii.json').read_text())['decisions']
 new=json.loads((RUN/f'predictions_ct/{cid}.nii.json').read_text())['decisions']
 assert len(old)==len(new)
 for a,b in zip(old,new):
  assert a['index']==b['index'] and a['keep']==b['keep']
  for k in ['image_probability','anatomical_probability']:assert abs(a[k]-b[k])<2e-5
 unchanged.append(cid)
write_json(RUN/'evaluation/FROZEN_FILTER_DECISIONS_VERIFIED.json',{'cases':unchanged,'thresholds_and_candidate_selection_unchanged':True,'only_segmentation_changed':True})
write_json(RUN/'ADOPTION.json',{'adopt':pareto and out['E21']['matched']>=4 and out['E21']['location_correct']>=3,'official_pareto_gate':pareto,'official_delta':delta,'lesion_gate':out,'source_thresholds_unchanged':True})
