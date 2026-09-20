"""Image-only native preprocessing and one fixed final D, no checkpoint ensemble."""
import json,pickle,time,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import partial
import numpy as np,torch
from omegaconf import OmegaConf
from nndet.preprocessing.preprocessor import GenericPreprocessor
from nndet.inference.helper import predict_dir
from nndet.inference.loading import load_final_model
from scripts.astra6_e33.prepare import P,RUN,PREP,TASK,DATA,sha,write
from scripts.astra6_e33.train import MODEL
OUT=RUN/'comparison_image_preprocessing'
def prepare(cid):
 start=time.monotonic();marker=OUT/(cid+'.json');planfile=PREP/'D3V001_3d.pkl';ph=sha(planfile)
 if marker.exists():
  r=json.loads(marker.read_text());assert r['plan_sha256']==ph;return r
 plan=pickle.loads(planfile.read_bytes());pre=GenericPreprocessor(norm_scheme_per_modality=plan['normalization_schemes'],use_mask_for_norm=plan['use_mask_for_norm'],transpose_forward=plan['transpose_forward'],intensity_properties=None,resample_anisotropy_threshold=plan['resample_anisotropy_threshold']);image=DATA/f'images/{cid}_0000.nii.gz';x,_,props=pre.preprocess_test_case([str(image)],plan['target_spacing']);assert np.isfinite(x).all()
 for suffix,value in [('.npy',x),('.pkl',props)]:
  target=OUT/(cid+suffix);temp=target.with_suffix(target.suffix+'.tmp')
  with temp.open('wb') as f:
   if suffix=='.npy':np.save(f,value,allow_pickle=False)
   else:pickle.dump(value,f)
  os.replace(temp,target)
 r={'case_id':cid,'image_sha256':sha(image),'plan_sha256':ph,'GT_not_read':True,'image_only_preprocessed_shape':list(x.shape),'seconds':time.monotonic()-start};write(marker,r);return r

def main():
 torch.set_num_threads(1);lock=json.loads((RUN/'TRAINED.json').read_text());fold=RUN/'models'/TASK/MODEL/'fold0';assert sha(fold/'model_last.ckpt')==lock['sha256'];assert sha(fold/'plan.pkl')==lock['plan_sha256'];ids=json.loads((P/'artifacts/astra6_e02_multiscale_vessel_signature_20260908T173916Z/eval_case_ids.json').read_text());assert len(ids)==40;OUT.mkdir(exist_ok=True);dest=RUN/'comparison_boxes';dest.mkdir(exist_ok=True)
 if (RUN/'DETECTOR_INFERENCE_COMPLETE.json').exists():return
 with ProcessPoolExecutor(max_workers=2) as pool:
  for f in as_completed([pool.submit(prepare,c) for c in ids]):r=f.result();print('E33 COMPARISON IMAGE',r['case_id'],r['seconds'],flush=True)
 from src.inference.cpu_nms_fallback import cuda_nms_is_usable,install_cpu_nms_fallback
 from src.inference.nndet_inference import detection_cap_override
 cpu_nms=not cuda_nms_is_usable()
 if cpu_nms:install_cpu_nms_fallback()
 cfg=OmegaConf.load(fold/'config.yaml');plan=pickle.loads((fold/'plan.pkl').read_bytes());plan['batch_size']=1;plan['inference_plan']={**plan.get('inference_plan',{}),**detection_cap_override(200)};plan['architecture']['detections_per_img']=200
 metadata={'checkpoint_sha256':lock['sha256'],'plan_sha256':lock['plan_sha256'],'cases':ids,'num_models':1,'num_tta':1,'max_detections':200,'native_restore':True,'no_MR40_postprocessing_sweep':True,'GT_not_read':True,'nms_backend':'CPU fallback' if cpu_nms else 'CUDA nndet._C'};marker=RUN/'DETECTOR_INFERENCE_FROZEN.json'
 if marker.exists():assert json.loads(marker.read_text())==metadata
 else:write(marker,metadata)
 pending=[c for c in ids if not (dest/(c+'_boxes.pkl')).exists()];start=time.monotonic()
 if pending:predict_dir(source_dir=OUT,target_dir=dest,cfg=cfg,plan=plan,source_models=fold,model_fn=partial(load_final_model,identifier='last'),num_models=1,num_tta_transforms=1,restore=True,case_ids=pending,save_state=False,device='cuda:0',ensemble_on_device=True)
 write(RUN/'DETECTOR_INFERENCE_COMPLETE.json',{**metadata,'boxes_sha256':{c:sha(dest/(c+'_boxes.pkl')) for c in ids},'seconds_current_inference':time.monotonic()-start});print('E33 DETECTOR INFERENCE COMPLETE',flush=True)
if __name__=='__main__':main()
