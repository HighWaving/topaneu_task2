"""Describe E26 generalization limits using saved outputs and NIfTI headers only."""
from pathlib import Path
import json,math
import nibabel as nib,numpy as np
P=Path(__file__).resolve().parents[2];D=P.parent/'data_topaneu26';R=P/'artifacts/research_audit_20260909';S=P/'artifacts/astra6_e17_MR_detector_crop_segmentation_20260909';E=P/'artifacts/astra6_e26_MR_segmentation_regularization_20260909'
records=[json.loads(s) for s in (S/'features/train_records.jsonl').read_text().splitlines()];split=json.loads((S/'source_split.json').read_text());headers={}
def header(cid):
 if cid not in headers:
  im=nib.load(str(D/f'images/{cid}_0000.nii.gz'));headers[cid]=(np.linalg.norm(im.affine[:3,:3],axis=0),abs(np.linalg.det(im.affine[:3,:3])))
 return headers[cid]
source=[]
for j,idx in enumerate(split['development_detector_rows']):
 r=records[idx];sp,v=header(r['case_id']);target=dict(np.load(E/f'native_validation/{j:02d}.npz'));n=int(target['GT_voxels']);extent=(np.array(r['high'])-r['low'])*sp
 source.append({'case_id':r['case_id'],'class':r['source_class_id'],'GT_voxels':n,'volume_mm3':n*v,'equivalent_diameter_mm':(6*n*v/math.pi)**(1/3),'spacing_mm':sp.tolist(),'anisotropy':float(sp.max()/sp.min()),'box_extent_mm':extent.tolist(),'input_FOV_mm':(2*extent).tolist(),'box_to_GT_volume':float(np.prod(extent)/(n*v)),'GT_inside_box_fraction':float(target['truth'].sum()/n)})
before=json.loads((S/'evaluation/lesion_diagnostics.json').read_text());after=json.loads((E/'evaluation/lesion_diagnostics.json').read_text());comparison=[]
for cid,b in before['cases'].items():
 sp,v=header(cid)
 for old,new in zip(b['lesions'],after['cases'][cid]['lesions']):
  assert (old['class'],old['voxels'])==(new['class'],new['voxels'])
  if old['matched'] and new['matched']:comparison.append({'case_id':cid,'class':old['class'],'volume_mm3':old['volume_mm3'],'equivalent_diameter_mm':old['equivalent_sphere_diameter_mm'],'spacing_mm':sp.tolist(),'anisotropy':float(sp.max()/sp.min()),'E17_relative_volume_error':old['relative_volume_error'],'E26_relative_volume_error':new['relative_volume_error'],'Dice_delta':new['binary_dice']-old['binary_dice']})
def summary(rows):
 return {'n_lesions':len(rows),'n_cases':len({r['case_id'] for r in rows}),'quantiles_10_50_90':{key:np.percentile([r[key] for r in rows],[10,50,90]).tolist() for key in ['equivalent_diameter_mm','volume_mm3','anisotropy']},'spacing_mm_quantiles_10_50_90':np.percentile([r['spacing_mm'] for r in rows],[10,50,90],axis=0).tolist()}
result={'source31':summary(source),'MR40_common_matched53':summary(comparison),'source_rows':source,'comparison_rows':comparison,'source_selection_stability':json.loads((R/'E26_SELECTION_STABILITY.json').read_text()),'limitations':['Different centers and proposal exposure are confounded; differences describe observed domains, not a causal attribution','Source31 uses development weights and in-sample D boxes; MR53 uses final weights and held-out D images','No MR40 threshold or epoch search, no image/GT used to fit','GT_inside_box and FOV quantified for source only; paired OOF-box comparison awaits E23'],'next_capability_test':'On fixed source development lesions compare old versus OOF candidate geometry with identical S17 development weights; examine boundaries/shape errors before selecting candidate-robust segmentation representation.'}
(R/'E26_DOMAIN_EVIDENCE.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['source31','MR40_common_matched53']},indent=2))
