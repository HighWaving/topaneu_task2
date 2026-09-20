"""Compare frozen raw-package outputs with the locked evaluation artifacts."""
import json,time,pickle
import numpy as np,SimpleITK as sitk
from scripts.astra6_e01.e01_common import P,DATA,sha256_file,write_json,select_candidates,load_boxes
RUN=P/'artifacts/r2_raw_cohort_parity_20260909'
def main():
 start=time.monotonic();plan=json.loads((RUN/'PLAN.json').read_text());results={}
 for case in plan['cases']:
  mod,cid=case['modality'],case['case_id'];resource=RUN/f'{mod}_resources.json'
  while True:
   try:state=json.loads(resource.read_text())
   except (OSError,json.JSONDecodeError):state={}
   if state.get('complete') and 'exit_code' in state:break
   assert time.monotonic()-start<7200,'Two-case raw validation exceeded its budget';time.sleep(5)
  assert state['exit_code']==0,(mod,'raw inference failed')
  for k,path in case['reference_paths'].items():assert sha256_file(P/path)==case['reference_sha256'][k]
  raw=sitk.ReadImage(str(DATA/f'images/{cid}_0000.nii.gz'));output=sitk.ReadImage(str(RUN/f'{mod}.mha'));ref=sitk.ReadImage(str(P/case['reference_paths']['prediction']));arr=sitk.GetArrayFromImage(output);before=sitk.GetArrayFromImage(ref)
  geometry=output.GetSize()==raw.GetSize() and np.allclose(output.GetSpacing(),raw.GetSpacing()) and np.allclose(output.GetOrigin(),raw.GetOrigin()) and np.allclose(output.GetDirection(),raw.GetDirection());difference=int(np.count_nonzero(arr!=before));assert arr.shape==before.shape
  with (RUN/f'{mod}_work/boxes/case_boxes.pkl').open('rb') as f:newboxes=pickle.load(f)
  with (P/case['reference_paths']['boxes']).open('rb') as f:oldboxes=pickle.load(f)
  boxstats={}
  for key in ['pred_boxes','pred_scores']:
   a,b=np.asarray(newboxes[key]),np.asarray(oldboxes[key]);same=a.shape==b.shape;boxstats[key]={'new_shape':list(a.shape),'old_shape':list(b.shape),'max_absolute_difference':float(np.max(abs(a-b))) if same and a.size else None,'exact':bool(same and np.array_equal(a,b))}
  nb,ns,_=load_boxes(RUN/f'{mod}_work/boxes/case_boxes.pkl');ob,oss,_=load_boxes(P/case['reference_paths']['boxes']);deployed_new=[{'score':score,'low':lo.tolist(),'high':hi.tolist()} for _,score,lo,hi in select_candidates(nb,ns)];deployed_old=[{'score':score,'low':lo.tolist(),'high':hi.tolist()} for _,score,lo,hi in select_candidates(ob,oss)];boxstats['deployed_pool']={'exact':deployed_new==deployed_old,'new':deployed_new,'old':deployed_old}
  newv=sitk.GetArrayFromImage(sitk.ReadImage(str(RUN/f'{mod}_work/predicted_vessel.nii.gz')));oldv=sitk.GetArrayFromImage(sitk.ReadImage(str(P/case['reference_paths']['vessel'])));vessel_delta=int(np.count_nonzero(newv!=oldv)) if newv.shape==oldv.shape else None
  results[mod]={'case_id':cid,'geometry_verified':bool(geometry),'uint8_labels0to52':output.GetPixelID()==sitk.sitkUInt8 and int(arr.max())<=52,'mask_different_voxels':difference,'mask_exact':difference==0,'detector':boxstats,'predicted_vessel_different_voxels':vessel_delta,'runtime':json.loads((RUN/f'{mod}_work/runtime.json').read_text()),'resources':{'peak':state['peak'],'limitations':state['limitations']},'output_sha256':sha256_file(RUN/f'{mod}.mha'),'reference_sha256':case['reference_sha256'],'GT_not_used':True};write_json(RUN/'RESULT.json',results);print('RAW_PARITY',mod,'mask_delta',difference,'vessel_delta',vessel_delta,flush=True)
 write_json(RUN/'DONE.json',{'all_exact':all(r['mask_exact'] and r['geometry_verified'] and r['uint8_labels0to52'] for r in results.values()),'n_cases':2,'scope':'Two frozen comparison cases, not a fresh generalization estimate','results':results})
if __name__=='__main__':main()
