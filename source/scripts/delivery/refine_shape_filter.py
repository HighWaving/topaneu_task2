"""E25 path-based inference: image + predicted S probability, no GT access."""
import argparse,json,pickle,tempfile,time,resource
from pathlib import Path
import nibabel as nib,numpy as np,torch,SimpleITK as sitk
from scripts.astra6_e25.common import Filter
from scripts.astra6_e06.common import crops
from scripts.astra6_e04.run_e04 import Segmenter,PRIOR,SUPPORT
from scripts.astra6_e01.e01_common import load_boxes,select_candidates,sha256_file
from scripts.delivery.refine import predict

EXPECTED_S='ec1d12c8028843d9ea2cc932f4025bf1247c87135bdb0266c1d5cfbfb20fa14d'
def main():
 p=argparse.ArgumentParser()
 for name in ['image','predicted-vessel','boxes','classifier','segmentation','fp-filter','output']:p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--fp-threshold',type=float,required=True);p.add_argument('--modality',choices=['MR'],default='MR');p.add_argument('--device',default='cpu');a=p.parse_args();assert 0<=a.fp_threshold<=1;assert sha256_file(a.segmentation)==EXPECTED_S,'E25 was calibrated for frozen E17 S; another S requires a new validation';torch.set_num_threads(2);start=time.monotonic()
 ck=torch.load(a.fp_filter,map_location='cpu',weights_only=False);assert ck['architecture']=='E25_OOF_image_shape_CNN';model=Filter().to(a.device);model.load_state_dict(ck['state_dict']);model.eval();s=Segmenter().to(a.device);s.load_state_dict(torch.load(a.segmentation,map_location='cpu',weights_only=False)['state_dict']);s.eval()
 im=nib.load(str(a.image));arr=im.get_fdata(dtype=np.float32);aff=im.affine;assert np.isfinite(arr).all();sample=arr[::4,::4,::4];sample=sample[sample!=0];assert len(sample);lo,hi=np.percentile(sample,[.5,99.5]);arr-=float(lo);arr/=max(float(hi-lo),1e-6);np.clip(arr,0,1,out=arr)
 bx,sc,obj=load_boxes(a.boxes);filtered=np.zeros_like(sc);decisions=[]
 for idx,score,low,high in select_candidates(bx,sc):
  x=crops(arr,aff,low,high);sx=np.stack([x[0],PRIOR])[None].astype(np.float32)
  with torch.inference_mode():
   shape=s(torch.from_numpy(sx).to(a.device)).sigmoid().cpu().numpy()[0,0];fx=np.concatenate([x,(shape*SUPPORT)[None]],axis=0);prob=float(model(torch.from_numpy(fx[None]).to(a.device)).sigmoid().cpu()[0])
  keep=prob>=a.fp_threshold
  if keep:filtered[idx]=score
  decisions.append({'index':idx,'probability':prob,'keep':keep})
 del arr,im,s,model,ck
 if a.device!='cpu':torch.cuda.empty_cache()
 filter_seconds=time.monotonic()-start;obj['pred_scores']=filtered
 with tempfile.TemporaryDirectory(prefix='task2-shape-filter-') as td:
  fp=Path(td)/'boxes.pkl'
  with fp.open('wb') as f:pickle.dump(obj,f)
  mask,ledger=predict(a.image,a.predicted_vessel,fp,a.classifier,a.segmentation,a.device,a.modality)
 ref=sitk.ReadImage(str(a.image));out=sitk.GetImageFromArray(mask.transpose(2,1,0));out.CopyInformation(ref);a.output.parent.mkdir(parents=True,exist_ok=True);sitk.WriteImage(out,str(a.output),True);check=sitk.ReadImage(str(a.output));assert check.GetSize()==ref.GetSize() and check.GetSpacing()==ref.GetSpacing() and np.allclose(check.GetDirection(),ref.GetDirection()) and np.allclose(check.GetOrigin(),ref.GetOrigin()) and check.GetPixelID()==sitk.sitkUInt8
 a.output.with_suffix('.json').write_text(json.dumps({'architecture':'E25_OOF_image_shape_CNN','filter_seconds':filter_seconds,'total_seconds':time.monotonic()-start,'peak_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'filter_threshold':a.fp_threshold,'filter_decisions':decisions,'candidates':ledger,'no_candidate_refill':True,'geometry_verified':True,'segmentation_sha256':EXPECTED_S},indent=2)+'\n')
if __name__=='__main__':main()
