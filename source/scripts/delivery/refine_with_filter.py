"""Optional E06 image-only candidate filter before the frozen refinement pipeline."""
import argparse,json,pickle,tempfile,time,resource
from pathlib import Path
import numpy as np,torch,SimpleITK as sitk
from scripts.astra6_e06.common import Filter,crops,load_image,load_filter
from scripts.astra6_e01.e01_common import load_boxes,select_candidates
from scripts.delivery.refine import predict

def main():
 p=argparse.ArgumentParser();p.add_argument('--image',type=Path,required=True);p.add_argument('--predicted-vessel',type=Path,required=True);p.add_argument('--boxes',type=Path,required=True);p.add_argument('--classifier',type=Path,required=True);p.add_argument('--segmentation',type=Path);p.add_argument('--modality',choices=['MR','CT'],default='MR');p.add_argument('--device',default='cpu');p.add_argument('--fp-filter',type=Path,required=True);p.add_argument('--fp-threshold',type=float,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert 0<=a.fp_threshold<=1;torch.set_num_threads(4);start=time.monotonic()
 # General file-path variant of the exact source per-image normalization.
 import nibabel as nib
 im=nib.load(str(a.image));arr=im.get_fdata(dtype=np.float32);aff=im.affine;sub=arr[::4,::4,::4];sub=sub[sub!=0];assert len(sub)>0 and np.isfinite(arr).all();lo,hi=np.percentile(sub,[.5,99.5]);arr-=float(lo);arr/=max(float(hi-lo),1e-6);np.clip(arr,0,1,out=arr)
 model=load_filter(a.fp_filter,a.device);bx,sc,obj=load_boxes(a.boxes);selected=select_candidates(bx,sc);filtered=np.zeros_like(sc);decisions=[]
 for idx,score,low,high in selected:
  with torch.inference_mode():prob=float(model(torch.from_numpy(crops(arr,aff,low,high)[None]).to(a.device)).sigmoid().cpu()[0])
  keep=prob>=a.fp_threshold
  if keep:filtered[idx]=score
  decisions.append({'index':idx,'probability':prob,'keep':keep})
 del arr,model,im
 if a.device!='cpu':torch.cuda.empty_cache()
 filter_seconds=time.monotonic()-start;obj['pred_scores']=filtered
 # Preserve original top5 selection: lower-ranked candidates are never refilled.
 with tempfile.TemporaryDirectory(prefix='task2-filter-') as td:
  fp=Path(td)/'filtered_boxes.pkl'
  with fp.open('wb') as f:pickle.dump(obj,f)
  mask,ledger=predict(a.image,a.predicted_vessel,fp,a.classifier,a.segmentation,a.device,a.modality)
 ref=sitk.ReadImage(str(a.image));out=sitk.GetImageFromArray(mask.transpose(2,1,0));out.CopyInformation(ref);a.output.parent.mkdir(parents=True,exist_ok=True);sitk.WriteImage(out,str(a.output),True);a.output.with_suffix('.json').write_text(json.dumps({'filter_seconds':filter_seconds,'total_seconds':time.monotonic()-start,'peak_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'filter_threshold':a.fp_threshold,'filter_decisions':decisions,'candidates':ledger,'no_candidate_refill':True},indent=2))
if __name__=='__main__':main()
