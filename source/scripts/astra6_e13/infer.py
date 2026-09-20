"""Generic native-image entry for the experimental CT anatomy filter."""
from pathlib import Path
import argparse,json,time,pickle,tempfile,resource
import numpy as np,nibabel as nib,torch,joblib,SimpleITK as sitk
from scripts.astra6_e13.prepare import RUN,PARENT,P,DATA
from scripts.astra6_e06.common import Filter,crops
from scripts.astra6_e01.e01_common import load_boxes,select_candidates,compute_feature,sha256_file,sha256_tree,write_json
from scripts.astra6_e02.run_e02 import multiscale
from scripts.delivery.geometry import vessel_geometry_fast
from scripts.delivery.refine import predict

def native(image,vessel,boxes,classifier,segmentation,image_filter,image_threshold,anatomical_filter,anatomical_threshold,output,device):
 start=time.monotonic();torch.set_num_threads(4);clf=joblib.load(anatomical_filter);assert clf.feature_version=='anatomical_fp_v1' and sha256_file(image_filter)==clf.required_image_filter_sha256
 im=nib.load(str(image));arr=im.get_fdata(dtype=np.float32);aff=im.affine;geom=vessel_geometry_fast(vessel,arr.shape,aff);sub=arr[::4,::4,::4];sub=sub[sub!=0];assert len(sub) and np.isfinite(arr).all();lo,hi=np.percentile(sub,[.5,99.5]);arr-=float(lo);arr/=max(float(hi-lo),1e-6);np.clip(arr,0,1,out=arr)
 net=Filter().to(device);net.load_state_dict(torch.load(image_filter,map_location='cpu',weights_only=False)['state_dict']);net.eval();bx,sc,obj=load_boxes(boxes);filtered=np.zeros_like(sc);decisions=[]
 for idx,score,low,high in select_candidates(bx,sc):
  with torch.inference_mode():prob=float(net(torch.from_numpy(crops(arr,aff,low,high)[None]).to(device)).sigmoid().cpu()[0])
  feature=np.concatenate([compute_feature(geom,aff,low,high,'CT'),multiscale(geom,aff,low,high),[prob]]).astype(np.float32);q=float(clf.predict_proba(feature[None])[0,list(clf.classes_).index(1)]);keep=prob>=image_threshold and q>=anatomical_threshold
  if keep:filtered[idx]=score
  decisions.append({'index':idx,'image_probability':prob,'anatomical_probability':q,'keep':keep})
 del arr,im,geom,net,clf;torch.cuda.empty_cache() if device!='cpu' else None;obj['pred_scores']=filtered
 with tempfile.TemporaryDirectory(prefix='task2-anatomy-filter-') as td:
  path=Path(td)/'boxes.pkl';path.write_bytes(pickle.dumps(obj));mask,ledger=predict(image,vessel,path,classifier,segmentation,device,'CT')
 ref=sitk.ReadImage(str(image));out=sitk.GetImageFromArray(mask.transpose(2,1,0));out.CopyInformation(ref);output.parent.mkdir(parents=True,exist_ok=True);sitk.WriteImage(out,str(output),True);write_json(output.with_suffix('.json'),{'seconds':time.monotonic()-start,'peak_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'decisions':decisions,'candidates':ledger,'no_refill':True,'native_geometry':True});return mask

def main():
 p=argparse.ArgumentParser();p.add_argument('--image',type=Path);p.add_argument('--predicted-vessel',type=Path);p.add_argument('--boxes',type=Path);p.add_argument('--output',type=Path);p.add_argument('--classifier',type=Path,default=P/'artifacts/astra6_e09_CT_expanded_location_20260909/model/classifier.joblib');p.add_argument('--segmentation',type=Path,default=P/'artifacts/astra6_e04_crop_segmentation_20260909T031146Z/model/final_last.pt');p.add_argument('--image-filter','--fp-filter',dest='image_filter',type=Path,default=PARENT/'model/final_last.pt');p.add_argument('--image-threshold','--fp-threshold',dest='image_threshold',type=float);p.add_argument('--modality',choices=['CT'],default='CT');p.add_argument('--anatomical-filter',type=Path,default=RUN/'model/final.joblib');p.add_argument('--anatomical-threshold',type=float);p.add_argument('--device',default='cuda:0');a=p.parse_args();threshold=json.loads((RUN/'model/THRESHOLD.json').read_text()) if a.image_threshold is None or a.anatomical_threshold is None else {};image_threshold=a.image_threshold if a.image_threshold is not None else threshold['parent_image_threshold'];anatomical_threshold=a.anatomical_threshold if a.anatomical_threshold is not None else threshold['threshold']
 if a.image:
  assert a.predicted_vessel and a.boxes and a.output;native(a.image,a.predicted_vessel,a.boxes,a.classifier,a.segmentation,a.image_filter,image_threshold,a.anatomical_filter,anatomical_threshold,a.output,a.device);return
 ids=json.loads((RUN/'source_split.json').read_text())['fixed_CT5'];lock=json.loads((RUN/'model/LOCKED.json').read_text());assert sha256_file(a.anatomical_filter)==lock['model_sha256'];start=time.time()
 for cid in ids:
  native(DATA/f'images/{cid}_0000.nii.gz',P/f'artifacts/ta36_ct_output/{cid}.nii.gz',P/f'artifacts/ct_fold2_all109_boxes/{cid}_boxes.pkl',a.classifier,a.segmentation,a.image_filter,image_threshold,a.anatomical_filter,anatomical_threshold,RUN/f'predictions_ct/{cid}.nii.gz',a.device);print('CT_E13',cid,flush=True)
 write_json(RUN/'PREDICTIONS_LOCKED.json',{'prediction_tree_sha256':sha256_tree(RUN/'predictions_ct'),'GT_not_read':True,'cases':ids,'seconds':time.time()-start,'model_sha256':lock['model_sha256']})
if __name__=='__main__':main()
