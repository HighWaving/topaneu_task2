"""Raw MR K pipeline with frozen source output policy; serial GPU execution."""
import argparse,json,os,sys,time,shutil,resource
from pathlib import Path
import numpy as np,nibabel as nib
from nibabel.processing import resample_from_to
from scripts.delivery.raw_pipeline import run
P=Path(__file__).resolve().parents[2];V=P.parent

def main():
 p=argparse.ArgumentParser()
 for k in ['image','output','work']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--bundle',type=Path,default=P/'submission_models_MRE32_CTE16_20260910_r3');p.add_argument('--classifier',type=Path);p.add_argument('--model-dir',type=Path)
 p.add_argument('--policy',choices=['dense_fusion','detector_control'],default='dense_fusion');p.add_argument('--gpu',required=True);p.add_argument('--detector-python',type=Path,default=V/'conda_envs/nndet/bin/python');p.add_argument('--refinement-python',type=Path,default=V/'conda_envs/nnunet_v100/bin/python')
 p.add_argument('--calibration',type=Path,default=P/'artifacts/mr_dense_search_S32_refinement_20260912/SOURCE_CALIBRATION_LOCKED.json');a=p.parse_args()
 a.work=a.work.resolve();a.output=a.output.resolve();a.bundle=a.bundle.resolve();a.work.mkdir(parents=True,exist_ok=False);began=time.monotonic();times={}
 mdir=(a.model_dir or (a.bundle/'models' if (a.bundle/'models').exists() else a.bundle)).resolve()
 cfg=json.loads(a.calibration.read_text());q=cfg['arms']['fixed_control_plus_refined']['threshold'];d=cfg['control_D_threshold'];assert cfg['source45_only'];classifier=(a.classifier or mdir/'location_MR.joblib').resolve()
 raw=a.work/'case_0000.nii.gz';shutil.copyfile(a.image,raw)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',GIT_PYTHON_REFRESH='quiet')
 detcode=a.bundle/'source/dependencies/detector_app';denv=env.copy();denv['PYTHONPATH']=os.pathsep.join([str(P),str(detcode.parent),str(detcode)]);denv['det_data']=str(V/'nndet_data');denv['det_models']=str(V/'nndet_models');checkpoint=mdir/'detector_MR'
 def stage(name,cmd,cwd,environment):
  t=time.monotonic();run(cmd,cwd,environment,a.work/(name+'.log'));times[name]=time.monotonic()-t;(a.work/'runtime_partial.json').write_text(json.dumps(times,indent=2)+'\n')
 stage('detector_preprocessing',[a.detector_python,P/'scripts/delivery/preprocess_detector.py','--image',raw,'--plan',checkpoint/'plan.pkl','--out',a.work/'preprocessed'],P,denv)
 stage('detector_boxes',[a.detector_python,detcode/'src/scripts/infer_nndet_adam.py','--training-dir',checkpoint,'--source-dir',a.work/'preprocessed','--output-dir',a.work/'boxes','--checkpoint','last','--device','cuda:0','--num-tta','1','--batch-size','1','--max-detections','200','--case-id','case'],detcode,denv)
 if a.policy=='dense_fusion':
  stage('dense_independent_search',[a.detector_python,'-m','scripts.delivery.dense_auxiliary_explicit','--image',raw,'--preprocessed',a.work/'preprocessed','--checkpoint',checkpoint,'--output',a.work/'dense/candidates.json'],P,denv)
 else:
  ref=nib.load(str(raw));(a.work/'dense').mkdir();(a.work/'dense/candidates.json').write_text(json.dumps({'candidates':[],'native_shape':list(ref.shape),'affine':ref.affine.tolist(),'disabled_by_fixed_output_policy':True}));np.savez_compressed(a.work/'dense/candidates.npz')
 # Fusion mode searches the full image even without boxes.
 app=a.bundle/'source/dependencies/ta36_app';sys.path.insert(0,str(app/'ta36'));from reorient_nii import reorient_nii
 ti=a.work/'ta36_input';to=a.work/'ta36_output';ti.mkdir();to.mkdir();reorient_nii(nib.load(raw),targ_aff='LPS').to_filename(ti/'case_0000.nii.gz')
 tenv=env.copy();tenv['PYTHONPATH']=os.pathsep.join([str(P/'vendor/delivery_runtime_deps'),str(app/'vendor')]);tenv['TOPANEU_MODEL_ROOT']=str(mdir/'ta36')
 stage('ta36',[a.refinement_python,P/'scripts/delivery/ta36_cached_preprocessing.py',app/'ta36/run_inference.py','--input',ti,'--output',to,'--suffix','_0000.nii.gz','--sequential','--n_infer_workers','1','--n_pre_post_workers','1'],app,tenv)
 original=nib.load(raw);vessel=resample_from_to(nib.load(to/'case.nii.gz'),original,order=0);vpath=a.work/'predicted_vessel.nii.gz';nib.Nifti1Image(np.asarray(vessel.dataobj,dtype=np.uint8),original.affine,original.header.copy()).to_filename(vpath)
 stage('refinement',[a.refinement_python,'-m','scripts.delivery.refine_dense_fusion','--image',raw,'--predicted-vessel',vpath,'--boxes',a.work/'boxes/case_boxes.pkl','--dense',a.work/'dense/candidates.json','--classifier',classifier,'--segmentation',str(mdir/'segmentation_MR.pt'),'--detector-threshold',d,'--dense-threshold',q,'--policy',a.policy,'--output',a.output],P,env)
 times.update(total_seconds=time.monotonic()-began,within720_seconds_on_V100=time.monotonic()-began<=720,child_peak_rss_kb=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,T4_validated=False,container_validated=False,source_policy={'detector':d,'dense':q},classifier=str(classifier),inference_policy=a.policy,image_only_no_GT=True)
 (a.work/'runtime.json').write_text(json.dumps(times,indent=2)+'\n');print(times,flush=True)
if __name__=='__main__':main()
