"""MR raw-image pipeline: frozen nnDetection -> organizer TA36 -> E02/E04.
Runs each GPU model in a sequential subprocess to release all device memory.
"""
import argparse,json,os,pickle,subprocess,sys,time,resource
from pathlib import Path
import nibabel as nib,numpy as np,SimpleITK as sitk
from nibabel.processing import resample_from_to
P=Path(__file__).resolve().parents[2];V=P.parent

def run(argv,cwd,env,log):
 try:
  with log.open('w') as f:subprocess.run([str(x) for x in argv],cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 except subprocess.CalledProcessError as e:
  if log.exists():
   sys.stderr.write(f"\n=== SUBPROCESS FAILED: {' '.join(str(x) for x in argv)} ===\n")
   sys.stderr.write(f"Log ({log}):\n{log.read_text()}\n=== END SUBPROCESS LOG ===\n")
   sys.stderr.flush()
  raise

def main():
 p=argparse.ArgumentParser();p.add_argument('--fp-filter',type=Path);p.add_argument('--fp-threshold',type=float);p.add_argument('--cache-ta36-preprocessing',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--runtime-config',type=Path);p.add_argument('--modality',choices=['MR','CT'],default='MR');p.add_argument('--image',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--segmentation',type=Path);p.add_argument('--gpu',default='GPU-f7491bbb-3972-1264-755a-63dec96b4a7d');a=p.parse_args();a.work=a.work.resolve();a.output=a.output.resolve();a.work.mkdir(parents=True,exist_ok=False);start=time.monotonic();times={};cfg=json.loads(a.runtime_config.read_text()) if a.runtime_config else {}
 (a.work/'runtime_configuration.json').write_text(json.dumps({'config':cfg,'arguments':vars(a)},default=str,indent=2))
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1',GIT_PYTHON_REFRESH='quiet');raw=a.work/'case_0000.nii.gz';sitk.WriteImage(sitk.ReadImage(str(a.image)),str(raw),True)
 det=Path(cfg.get('detector_python',V/'conda_envs/nndet/bin/python'));seg=Path(cfg.get('refinement_python',V/'conda_envs/nnunet_v100/bin/python'));checkpoint=Path(cfg.get(a.modality+'_detector',P/('artifacts/fold1_checkpoint_snapshots/epoch60' if a.modality=='MR' else 'artifacts/checkpoint_safety_backups/ct_fold2_FINAL_20260816T183639')));detcode=Path(cfg.get('detector_code',V/'new_aneurysms'))
 denv=env.copy();denv['PYTHONPATH']=os.pathsep.join([str(detcode.parent),str(detcode)]);denv['det_data']=str(V/'nndet_data');denv['det_models']=str(V/'nndet_models');t=time.monotonic()
 run([det,P/'scripts/delivery/preprocess_detector.py','--image',raw,'--plan',checkpoint/'plan.pkl','--out',a.work/'preprocessed'],P,denv,a.work/'detector_prepare.log')
 run([det,detcode/'src/scripts/infer_nndet_adam.py','--training-dir',checkpoint,'--source-dir',a.work/'preprocessed','--output-dir',a.work/'boxes','--checkpoint','last','--device','cuda:0','--num-tta','1','--batch-size','1','--max-detections','200','--case-id','case'],detcode,denv,a.work/'detector.log');times['detector_including_preprocessing']=time.monotonic()-t
 (a.work/'runtime_partial.json').write_text(json.dumps(times,indent=2))
 with (a.work/'boxes/case_boxes.pkl').open('rb') as f:proposal=pickle.load(f)
 if not np.any(np.asarray(proposal['pred_scores'])>=.3):
  ref=sitk.ReadImage(str(raw));out=sitk.Image(ref.GetSize(),sitk.sitkUInt8);out.CopyInformation(ref);a.output.parent.mkdir(parents=True,exist_ok=True);sitk.WriteImage(out,str(a.output),True)
  times.update(total_seconds=time.monotonic()-start,empty_selection=True,child_peak_rss_kb=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,T4_validated=False)
  (a.work/'runtime.json').write_text(json.dumps(times,indent=2));print(times);return
 app=Path(cfg.get('ta36_code',V/'topaneu-task1-sanity-submission'));sys.path.insert(0,str(app/'ta36'));from reorient_nii import reorient_nii
 ti=a.work/'ta36_input';to=a.work/'ta36_output';ti.mkdir();to.mkdir();reorient_nii(nib.load(raw),targ_aff='LPS').to_filename(ti/'case_0000.nii.gz')
 tenv=env.copy();tenv['PYTHONPATH']=os.pathsep.join([str(P/'vendor/delivery_runtime_deps'),str(app/'vendor')]);tenv['TOPANEU_MODEL_ROOT']=str(cfg.get('ta36_models',V/'topaneu-task1-algorithm-model/ta36_models'));t=time.monotonic()
 ta36_command=[seg,P/'scripts/delivery/ta36_cached_preprocessing.py',app/'ta36/run_inference.py'] if a.cache_ta36_preprocessing else [seg,app/'ta36/run_inference.py']
 run(ta36_command+['--input',ti,'--output',to,'--suffix','_0000.nii.gz','--sequential','--n_infer_workers','1','--n_pre_post_workers','1'],app,tenv,a.work/'ta36.log')
 original=nib.load(raw);vessel=resample_from_to(nib.load(to/'case.nii.gz'),original,order=0);vpath=a.work/'predicted_vessel.nii.gz';nib.Nifti1Image(np.asarray(vessel.dataobj,dtype=np.uint8),original.affine).to_filename(vpath);times['ta36_including_reorientation']=time.monotonic()-t
 if not np.any(np.asarray(vessel.dataobj)>0):
  ref=sitk.ReadImage(str(raw));out=sitk.Image(ref.GetSize(),sitk.sitkUInt8);out.CopyInformation(ref);a.output.parent.mkdir(parents=True,exist_ok=True);sitk.WriteImage(out,str(a.output),True)
  times.update(total_seconds=time.monotonic()-start,empty_vessel=True,child_peak_rss_kb=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,T4_validated=False)
  (a.work/'runtime.json').write_text(json.dumps(times,indent=2));print(times);return
 filter_path=a.fp_filter or cfg.get(a.modality+'_fp_filter',cfg.get('fp_filter'));filter_threshold=a.fp_threshold if a.fp_threshold is not None else cfg.get(a.modality+'_fp_threshold',cfg.get('fp_threshold'))
 anatomical_filter=cfg.get(a.modality+'_anatomical_filter');anatomical_threshold=cfg.get(a.modality+'_anatomical_threshold')
 if anatomical_filter and (not filter_path or anatomical_threshold is None):raise ValueError('anatomical filter requires image filter and threshold')
 filter_architecture=cfg.get(a.modality+'_filter_architecture')
 if filter_architecture not in (None,'E25_OOF_image_shape_CNN'):raise ValueError('unknown configured filter architecture')
 shape_filter=filter_architecture=='E25_OOF_image_shape_CNN'
 if shape_filter and (a.modality!='MR' or anatomical_filter or not filter_path):raise ValueError('E25 requires MR image-shape filter without anatomical cascade')
 segmentation_architecture=cfg.get(a.modality+'_segmentation_architecture')
 if segmentation_architecture not in (None,'E32_image_candidate_48'):raise ValueError('unknown segmentation architecture')
 image_candidate_segmentation=segmentation_architecture=='E32_image_candidate_48'
 if image_candidate_segmentation and (a.modality!='MR' or not filter_path or anatomical_filter or shape_filter or not cfg.get('MR_fallback_segmentation')):raise ValueError('E32 requires MR image filter and explicit learned S17 fallback')
 refinement_module='scripts.delivery.refine_shape_filter' if shape_filter else ('scripts.astra6_e13.infer' if anatomical_filter else ('scripts.delivery.refine_with_filter' if filter_path else 'scripts.delivery.refine'))
 if image_candidate_segmentation:refinement_module='scripts.astra6_e32.predict_native'
 cmd=[seg,'-m',refinement_module,'--modality',a.modality,'--image',raw,'--predicted-vessel',vpath,'--boxes',a.work/'boxes/case_boxes.pkl','--classifier',Path(cfg.get(a.modality+'_location_classifier',cfg.get('location_classifier',P/'artifacts/astra6_e02_multiscale_vessel_signature_20260908T173916Z/model/classifier.joblib'))),'--device','cuda:0','--output',a.output]
 if image_candidate_segmentation:
  del cmd[3:5]
  cmd+=['--arm','normalized','--fallback-segmentation',Path(cfg['MR_fallback_segmentation']).resolve()]
 if filter_path:
  if filter_threshold is None:raise ValueError('fp_threshold is required with fp_filter')
  cmd+=['--fp-filter',Path(filter_path).resolve(),'--fp-threshold',float(filter_threshold)]
 if anatomical_filter:cmd+=['--anatomical-filter',Path(anatomical_filter).resolve(),'--anatomical-threshold',float(anatomical_threshold)]
 segmentation_path=cfg.get(a.modality+'_segmentation',a.segmentation)
 if segmentation_path:cmd+=['--segmentation',Path(segmentation_path).resolve()]
 t=time.monotonic();run(cmd,P,env,a.work/'refine.log');times['refine_and_output']=time.monotonic()-t
 times['total_seconds']=time.monotonic()-start;times['child_peak_rss_kb']=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss;times['within720_seconds_on_V100']=times['total_seconds']<=720;times['T4_validated']=False;times['cached_ta36_preprocessing']=a.cache_ta36_preprocessing
 (a.work/'runtime.json').write_text(json.dumps(times,indent=2));print(times)
if __name__=='__main__':main()
