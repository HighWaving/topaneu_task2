"""Frozen E02/E04 inference on arbitrary native images and predicted vessels.
No location ground truth, cohort IDs or source vessels are read by this entry point.
"""
import argparse,time,resource,json
from pathlib import Path
import joblib,nibabel as nib,numpy as np,SimpleITK as sitk,torch
from scipy.ndimage import map_coordinates
from scripts.astra6_e01.e01_common import load_boxes,select_candidates,vessel_geometry,compute_feature,fill_ellipsoid,sha256_file
from scripts.astra6_e02.run_e02 import multiscale
from scripts.delivery.geometry import vessel_geometry_fast
from scripts.astra6_e04.run_e04 import Segmenter,crop_volume,PRIOR,largest

def predict(image,vessel,boxes,classifier,segmentation=None,device='cpu',modality='MR'):
 raw=nib.load(str(image));aff=raw.affine;arr=raw.get_fdata(dtype=np.float32);shape=arr.shape
 if not np.isfinite(arr).all():raise ValueError('nonfinite image')
 geom=vessel_geometry_fast(vessel,shape,aff);clf=joblib.load(classifier);bx,sc,_=load_boxes(boxes);selected=select_candidates(bx,sc)
 feature_version=getattr(clf,'feature_version',None)
 if feature_version not in (None,'mask_contact_v1','E27_junction_v1'):raise ValueError('unknown classifier feature version')
 junctions=None
 if feature_version=='E27_junction_v1':
  from scripts.astra6_e27.common import anchors,features as junction_features
  junctions=anchors(geom)
 contact=feature_version=='mask_contact_v1'
 if contact and (segmentation is None or sha256_file(segmentation)!=clf.required_segmenter_sha256):raise ValueError('contact classifier requires its frozen segmentation checkpoint')
 model=None
 if segmentation:
  model=Segmenter().to(device);model.load_state_dict(torch.load(segmentation,map_location=device,weights_only=False)['state_dict']);model.eval()
  sub=arr[::4,::4,::4];sub=sub[sub!=0]
  if not len(sub):raise ValueError('empty image')
  low,high=np.percentile(sub,[.5,99.5]);arr-=float(low);arr/=max(float(high-low),1e-6);np.clip(arr,0,1,out=arr)
 mask=np.zeros(shape,np.uint8);ledger=[]
 for idx,score,low,high in reversed(selected):
  features=np.concatenate([compute_feature(geom,aff,low,high,modality),multiscale(geom,aff,low,high)]).astype(np.float32)
  if junctions is not None:features=np.concatenate([features,junction_features(junctions,aff,low,high)])
  cl=None if contact else int(clf.classes_[np.argmax(clf.predict_proba(features[None])[0])]);fallback=False
  if model is None:fill_ellipsoid(mask,low,high,cl)
  else:
   x=np.stack([crop_volume(arr,low,high,1),PRIOR])[None].astype(np.float32)
   with torch.inference_mode():prob=model(torch.from_numpy(x).to(device)).sigmoid().cpu().numpy()[0,0]
   if contact:
    from scripts.astra6_e10.features import contact_features,mask_from_probability
    extra=contact_features(geom,aff,low,high,mask_from_probability(prob));full=np.concatenate([features,extra]);cl=int(clf.classes_[np.argmax(clf.predict_proba(full[None])[0])])
   lo=np.maximum(np.floor(low).astype(int),0);hi=np.minimum(np.ceil(high).astype(int),shape)
   if np.any(hi<=lo):raise ValueError('candidate outside native image')
   native=np.stack(np.meshgrid(*[np.arange(a,b) for a,b in zip(lo,hi)],indexing='ij'))
   coords=(native-((low+high)/2)[:,None,None,None])/(2*np.maximum(high-low,1))[:,None,None,None]*32+15.5
   fg=largest(map_coordinates(prob,coords.reshape(3,-1),order=1,mode='constant',cval=0,prefilter=False).reshape(tuple(hi-lo))>=.5)
   fallback=not fg.any()
   if fallback:fill_ellipsoid(mask,low,high,cl)
   else:mask[tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))][fg]=cl
  ledger.append({'index':idx,'score':score,'class':cl,'ellipsoid_empty_fallback':fallback})
 return mask,ledger

def main():
 p=argparse.ArgumentParser();p.add_argument('--modality',choices=['MR','CT'],default='MR');p.add_argument('--image',type=Path,required=True);p.add_argument('--predicted-vessel',type=Path,required=True);p.add_argument('--boxes',type=Path,required=True);p.add_argument('--classifier',type=Path,required=True);p.add_argument('--segmentation',type=Path);p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args();start=time.monotonic()
 mask,ledger=predict(a.image,a.predicted_vessel,a.boxes,a.classifier,a.segmentation,a.device,a.modality)
 ref=sitk.ReadImage(str(a.image));out=sitk.GetImageFromArray(mask.transpose(2,1,0));out.CopyInformation(ref);a.output.parent.mkdir(parents=True,exist_ok=True);sitk.WriteImage(out,str(a.output),True)
 check=sitk.ReadImage(str(a.output));assert check.GetSize()==ref.GetSize() and check.GetSpacing()==ref.GetSpacing() and np.allclose(check.GetDirection(),ref.GetDirection()) and np.allclose(check.GetOrigin(),ref.GetOrigin()) and check.GetPixelID()==sitk.sitkUInt8
 a.output.with_suffix('.json').write_text(json.dumps({'seconds':time.monotonic()-start,'peak_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'candidates':ledger,'geometry_verified':True,'raw_image_pipeline':False},indent=2))
if __name__=='__main__':main()
