"""Invoke pinned, unmodified 2026-09-09 official evaluator on saved predictions."""
import argparse,contextlib,hashlib,importlib.util,io,json,os,sys,time
from pathlib import Path
import numpy as np,SimpleITK as sitk
P=Path(__file__).resolve().parents[2];V=P.parent;ROOT=P/'vendor/official_task2_20260909';sys.path.insert(0,str(ROOT/'metrics'))
spec=importlib.util.spec_from_file_location('current_official',ROOT/'challenge/eval/task2/evaluate.py');official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
RUNS={'E01':P/'artifacts/astra6_e01_vessel_signature_52class_20260908T042902Z/predictions/mr_center2_k05','E02':P/'artifacts/astra6_e02_multiscale_vessel_signature_20260908T173916Z/predictions/mr_center2_k05','E04':P/'artifacts/astra6_e04_crop_segmentation_20260909T031146Z/predictions/mr_center2_k05','CT_E02':P/'artifacts/ct_independent5_20260909/E02','CT_E04':P/'artifacts/ct_independent5_20260909/E04'}
RUNS['E05']=P/'artifacts/astra6_e05_learned_location_splits_20260909/predictions/mr_center2_k05'
RUNS['E06']=P/'artifacts/astra6_e06_image_fp_filter_20260909/predictions/mr_center2_k05'
RUNS['E07']=P/'artifacts/astra6_e07_hierarchical_location_20260909/predictions/mr_center2_k05'
RUNS['CT_E06']=P/'artifacts/astra6_e06_image_fp_filter_20260909/predictions_ct'
RUNS['E08']=P/'artifacts/astra6_e08_mr_only_fp_filter_20260909/predictions/mr_center2_k05'
RUNS['CT_E09']=P/'artifacts/astra6_e09_CT_expanded_location_20260909/predictions_ct'
RUNS['E08D']=P/'artifacts/astra6_e08_deployed_pool_calibration_20260909/predictions/mr_center2_k05'
RUNS['CT_E11']=P/'artifacts/astra6_e11_CT_expanded_fp_filter_20260909/predictions_ct'
RUNS['E12']=P/'artifacts/astra6_e12_MR_expanded_fp_filter_20260909/predictions/mr_center2_k05'
RUNS['CT_E13']=P/'artifacts/astra6_e13_CT_anatomical_fp_filter_20260909/predictions_ct'
RUNS['E14']=P/'artifacts/astra6_e14_MR_filter_seed_stability_20260909/predictions/mr_center2_k05'
RUNS['CT_E15']=P/'artifacts/astra6_e15_CT_predicted_vessel_location_20260909/predictions_ct'
RUNS['CT_E22']=P/'artifacts/astra6_e22_CT_only_segmentation_20260909/predictions_ct'
RUNS['CT_E21']=P/'artifacts/astra6_e21_CT_native_checkpoint_selection_20260909/predictions_ct'
RUNS['CT_E20']=P/'artifacts/astra6_e20_CT_detector_crop_segmentation_20260909/predictions_ct'
RUNS['CT_E16']=P/'artifacts/astra6_e16_CT_expanded_segmentation_20260909/predictions_ct'
RUNS['E19']=P/'artifacts/astra6_e19_MR_anatomical_fp_filter_20260909/predictions/mr_center2_k05'
RUNS['E18']=P/'artifacts/astra6_e18_MR_hierarchical_C_with_current_F_20260909/predictions/mr_center2_k05'
RUNS['E17']=P/'artifacts/astra6_e17_MR_detector_crop_segmentation_20260909/predictions/mr_center2_k05'
RUNS['E10']=P/'artifacts/astra6_e10_mask_contact_location_20260909/predictions/mr_center2_k05'
RUNS['E25']=P/'artifacts/astra6_e25_oof_shape_fp_filter_20260909/predictions/mr_center2_k05'
RUNS['E27']=P/'artifacts/astra6_e27_junction_location_20260909/predictions/mr_center2_k05'
RUNS['E26']=P/'artifacts/astra6_e26_MR_segmentation_regularization_20260909/predictions/mr_center2_k05'
RUNS['E28']=P/'artifacts/astra6_e28_joint_anatomy_location_20260909/predictions/mr_center2_k05'
RUNS['E29_joint']=P/'artifacts/astra6_e29_staged_anatomy_location_20260910/joint/predictions/mr_center2_k05'
RUNS['E29_staged']=P/'artifacts/astra6_e29_staged_anatomy_location_20260910/staged/predictions/mr_center2_k05'
RUNS['E30']=P/'artifacts/astra6_e30_expanded_CT_supervision_for_MR_20260910/predictions/mr_center2_k05'
RUNS['E31_normalized']=P/'artifacts/astra6_e31_physical_candidate_segmentation_20260910/normalized/predictions/mr_center2_k05'
RUNS['E31_physical']=P/'artifacts/astra6_e31_physical_candidate_segmentation_20260910/physical/predictions/mr_center2_k05'
RUNS['E32_normalized']=P/'artifacts/astra6_e32_image_only_segmentation_20260910/normalized/predictions/mr_center2_k05'
RUNS['E33']=P/'artifacts/astra6_e33_source_only_detector_20260910/predictions/mr_center2_k05'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n');os.replace(temp,p)
def aggregate(rows):
 with contextlib.redirect_stdout(io.StringIO()):
  pc=official.evaluation_aggregation(rows);avg=official.evaluation_average(pc)
 return {'overall':avg,'per_class':pc}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--version',choices=list(RUNS),required=True);a=ap.parse_args();out=P/'artifacts/current_official_20260909'/a.version;out.mkdir(parents=True,exist_ok=True);start=time.monotonic();files=sorted(RUNS[a.version].glob('*.nii.gz'));assert len(files)==(5 if a.version.startswith('CT') else 40);rows=[]
 for i,p in enumerate(files,1):
  cid=p.name.removesuffix('.nii.gz');gt=V/f'data_topaneu26/location_masks/{cid}.nii.gz';dest=out/f'cases/{cid}.json'
  if a.version.startswith('CT'):
   original_gt=gt;gt=P/f'artifacts/current_official_20260909/normalized_CT_GT/{cid}.nii.gz';gt.parent.mkdir(parents=True,exist_ok=True)
   if not gt.exists():
    image=sitk.ReadImage(str(original_gt));arr=sitk.GetArrayViewFromImage(image);assert arr.min()>=0 and arr.max()<=52;sitk.WriteImage(sitk.Cast(image,sitk.sitkUInt8),str(gt),True)
  hashes={'prediction':sha(p),'gt':sha(gt),'evaluator':sha(ROOT/'challenge/eval/task2/evaluate.py')}
  if dest.exists():
   r=json.loads(dest.read_text());assert r['hashes']==hashes
  else:
   reused=None
   for candidate in sorted(out.parent.glob('*/cases/'+cid+'.json')):
    cached=json.loads(candidate.read_text())
    if cached.get('hashes')==hashes:
     reused=(candidate,cached);break
   if reused is not None:
    origin,cached=reused;r={**cached,'seconds':0.,'reused_from':str(origin.relative_to(P)),'original_compute_seconds':cached.get('original_compute_seconds',cached.get('seconds'))};write(dest,r)
   else:
    pred=sitk.ReadImage(str(p));t=time.monotonic()
    with contextlib.redirect_stdout(io.StringIO()):raw=official.evaluation_function(pred,gt,execute_in_docker=False)
    r={'case_id':cid,'raw':raw,'hashes':hashes,'seconds':time.monotonic()-t};write(dest,r)
  rows.append(r);print(a.version,i,len(files),cid,r['seconds'],flush=True)
 write(out/'per_case.json',rows);result=aggregate([r['raw'] for r in rows]);write(out/'official.json',result);write(out/'DONE.json',{'seconds':time.monotonic()-start,'n_cases':len(rows),'provenance':json.loads((ROOT/'provenance.json').read_text()),'numpy':np.__version__,'SimpleITK':sitk.Version_VersionString(),'implementation':'unmodified upstream evaluation_function/aggregation/average; execute_in_docker=False'});print(result['overall'],flush=True)
if __name__=='__main__':main()
