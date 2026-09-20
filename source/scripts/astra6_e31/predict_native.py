"""Explicit-file MR refinement entry; no dataset IDs, GT or candidate caches required.
Runs unchanged D candidate selection, C02 and F14 with an E31 S checkpoint.
The detector/TA36 stages remain supplied by the existing raw-image pipeline.
"""
import argparse,json,resource,time
from pathlib import Path
import numpy as np,torch,nibabel as nib,joblib,SimpleITK as sitk
from scipy.ndimage import map_coordinates
from scripts.astra6_e31.common import input_crop,native_foreground
from scripts.astra6_e31.train import Model
from scripts.astra6_e01.e01_common import load_boxes,select_candidates,compute_feature,sha256_file
from scripts.astra6_e02.run_e02 import multiscale
from scripts.delivery.geometry import vessel_geometry_fast
from scripts.astra6_e06.common import crops,load_filter
from scripts.astra6_e04.run_e04 import Segmenter,crop_volume,PRIOR,largest

def main():
 p=argparse.ArgumentParser()
 for name in ['image','predicted-vessel','boxes','classifier','segmentation','fallback-segmentation','fp-filter','output']:p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--fp-threshold',type=float,required=True);p.add_argument('--arm',choices=['normalized','physical'],required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();assert 0<=a.fp_threshold<=1;torch.set_num_threads(2);start=time.monotonic()
 image=nib.load(str(a.image));aff=image.affine;arr=image.get_fdata(dtype=np.float32);assert np.isfinite(arr).all();sub=arr[::4,::4,::4];sub=sub[sub!=0];assert len(sub);lo,hi=np.percentile(sub,[.5,99.5]);arr-=float(lo);arr/=max(float(hi-lo),1e-6);np.clip(arr,0,1,out=arr)
 vi=nib.load(str(a.predicted_vessel));assert vi.shape==arr.shape and np.allclose(vi.affine,aff,atol=1e-4);vessel=np.asanyarray(vi.dataobj);assert vessel.min()>=0 and vessel.max()<=36
 geom=vessel_geometry_fast(a.predicted_vessel,arr.shape,aff);classifier=joblib.load(a.classifier);classifier.n_jobs=2;assert classifier.n_features_in_==943
 filter_model=load_filter(a.fp_filter,a.device);net=Model().to(a.device).eval();state=torch.load(a.segmentation,map_location='cpu',weights_only=False);assert state['arm']==a.arm;net.load_state_dict(state['state_dict']);backup=Segmenter().to(a.device).eval();backup.load_state_dict(torch.load(a.fallback_segmentation,map_location='cpu',weights_only=False)['state_dict'])
 boxes,scores,_=load_boxes(a.boxes);selected=select_candidates(boxes,scores);mask=np.zeros(arr.shape,np.uint8);ledger=[]
 for idx,score,low,high in reversed(selected):
  with torch.inference_mode():fp=float(filter_model(torch.from_numpy(crops(arr,aff,low,high)[None]).to(a.device)).sigmoid().cpu()[0])
  keep=fp>=a.fp_threshold;r={'original_index':idx,'score':score,'low':low.tolist(),'high':high.tolist(),'filter_probability':fp,'filter_keep':keep}
  if keep:
   feature=np.concatenate([compute_feature(geom,aff,low,high,'MR'),multiscale(geom,aff,low,high)]).astype(np.float32);cl=int(classifier.predict(feature[None])[0]);x,origin,step=input_crop(arr,vessel,low,high,aff,a.arm)
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=a.device.startswith('cuda')):pr=net(torch.from_numpy(x[None].astype(np.float32)).to(a.device)).float().sigmoid().cpu().numpy()[0,0]
   base,fg=native_foreground(pr,origin,step,arr.shape);fallback=not fg.any()
   if fallback:
    bx=np.stack([crop_volume(arr,low,high,1),PRIOR])[None].astype(np.float32)
    with torch.inference_mode():prob=backup(torch.from_numpy(bx).to(a.device)).sigmoid().cpu().numpy()[0,0]
    base=np.maximum(np.floor(low).astype(int),0);end=np.minimum(np.ceil(high).astype(int),arr.shape);grid=np.stack(np.meshgrid(*[np.arange(v,w) for v,w in zip(base,end)],indexing='ij'));coords=(grid-((low+high)/2)[:,None,None,None])/(2*np.maximum(high-low,1))[:,None,None,None]*32+15.5;fg=largest(map_coordinates(prob,coords.reshape(3,-1),order=1,mode='constant',cval=0,prefilter=False).reshape(tuple(end-base))>=.5)
   if fg.any():mask[tuple(slice(int(v),int(v+n)) for v,n in zip(base,fg.shape))][fg]=cl
   r.update(predicted_class_id=cl,learned_S17_fallback=fallback,empty_after_learned_fallback=not fg.any(),native_candidate_voxels=int(fg.sum()))
  ledger.append(r)
 ref=sitk.ReadImage(str(a.image));out=sitk.GetImageFromArray(mask.transpose(2,1,0));out.CopyInformation(ref);a.output.parent.mkdir(parents=True,exist_ok=True);sitk.WriteImage(out,str(a.output),True);check=sitk.ReadImage(str(a.output));assert check.GetSize()==ref.GetSize() and check.GetPixelID()==sitk.sitkUInt8
 corners=np.stack(np.meshgrid(*[[0,n-1] for n in ref.GetSize()],indexing='ij')).reshape(3,-1).T
 corner_error=max(np.linalg.norm(np.array(check.TransformIndexToPhysicalPoint(tuple(int(v) for v in c)))-ref.TransformIndexToPhysicalPoint(tuple(int(v) for v in c))) for c in corners);assert corner_error<1e-4,('NIfTI physical geometry changed',corner_error)
 a.output.with_suffix('.json').write_text(json.dumps({'seconds':time.monotonic()-start,'peak_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'candidates':ledger,'no_GT_or_dataset_cache_read':True,'no_ellipse_fallback':True,'geometry_verified':True,'max_corner_geometry_error_mm':float(corner_error),'arm':a.arm,'model_sha256':sha256_file(a.segmentation),'classifier_sha256':sha256_file(a.classifier),'filter_sha256':sha256_file(a.fp_filter),'fallback_sha256':sha256_file(a.fallback_segmentation)},indent=2)+'\n')
if __name__=='__main__':main()
