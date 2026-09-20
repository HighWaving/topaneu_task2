"""S-only native inference; retain E17 detector, classifier and filter decisions."""
import argparse,json,time,fcntl
import numpy as np,torch,nibabel as nib
from scripts.astra6_e31.common import *
from scripts.astra6_e31.train import Model
from scripts.astra6_e04.run_e04 import load_image
from scripts.astra6_e28.infer import draw,F,S
from scripts.astra6_e01.e01_common import DATA,TA36_DIR,sha256_file,sha256_tree,write_json,nifti_output
BASE=P/'artifacts/astra6_e02_multiscale_vessel_signature_20260908T173916Z'
def _main(arm):
 from scripts.astra6_e31.box_stability import main as stability
 stability()
 out=RUN/arm
 if (out/'PREDICTIONS_HASHED_BEFORE_EVAL_GT.json').exists():return
 torch.set_num_threads(2);lock=json.loads((out/'model/LOCKED.json').read_text());assert lock['full_formal_training_complete'] and sha256_file(out/'model/final_last.pt')==lock['model_sha256'];net=Model().cuda().eval();net.load_state_dict(torch.load(out/'model/final_last.pt',map_location='cpu',weights_only=False)['state_dict'])
 records=[json.loads(r) for r in (S/'candidate_predictions.jsonl').read_text().splitlines()];ids=json.loads((BASE/'eval_case_ids.json').read_text());proof=json.loads((F/'FIXED_UPSTREAM_PARITY.json').read_text());assert proof['complete'];rows=[];checks={};start=time.monotonic()
 for n,cid in enumerate(ids,1):
  image,aff,_=load_image(cid);vimg=nib.load(str(TA36_DIR/f'{cid}.nii.gz'));assert vimg.shape==image.shape and np.allclose(vimg.affine,aff,atol=1e-4);vessel=np.asanyarray(vimg.dataobj);mask=np.zeros(image.shape,np.uint8);oldmask=np.zeros_like(mask)
  cache=F/f'fixed_upstream/{cid}.npz';assert sha256_file(cache)==proof['cases'][cid]['probabilities_sha256'];z=np.load(cache);oldprob={int(i):q for i,q in zip(z['original_index'],z['probability'])};crows=[]
  for old in [r for r in records if r['case_id']==cid]:
   r=dict(old)
   if old['filter_keep']:
    cl=old['predicted_class_id'];low,high=np.array(old['low']),np.array(old['high']);draw(oldmask,oldprob[old['original_index']],low,high,cl)
    x,origin,step=input_crop(image,vessel,low,high,aff,arm)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):pr=net(torch.from_numpy(x[None].astype(np.float32)).cuda()).float().sigmoid().cpu().numpy()[0,0]
    lo,fg=native_foreground(pr,origin,step,image.shape);fallback=not fg.any();r.update(segmentation_arm='E31_'+arm,learned_S17_fallback=fallback,new_segmenter_voxels=int(fg.sum()),sampling_origin=origin.tolist(),sampling_step=step.tolist())
    if fallback:
     assert not old.get('empty_fallback',False),'Baseline ellipse cannot be introduced as a new learned fallback'
     draw(mask,oldprob[old['original_index']],low,high,cl)
    else:mask[tuple(slice(int(a),int(a+n)) for a,n in zip(lo,fg.shape))][fg]=cl
   rows.append(r);crows.append(r)
  ref=nib.load(str(S/f'predictions/mr_center2_k05/{cid}.nii.gz'));assert np.array_equal(oldmask,np.asanyarray(ref.dataobj)) and np.allclose(aff,ref.affine,atol=1e-4)
  nifti_output(mask,DATA/f'images/{cid}_0000.nii.gz',out/f'predictions/mr_center2_k05/{cid}.nii.gz');checks[cid]={'baseline_native_exact':True,'D_C_F_decisions_exact':True,'learned_fallback_count':sum(r.get('learned_S17_fallback',False) for r in crows),'predicted_vessel_sha256':sha256_file(TA36_DIR/f'{cid}.nii.gz')};print('E31',arm,n,40,cid,flush=True)
 (out/'candidate_predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n');write_json(out/'FIXED_UPSTREAM_PARITY.json',{'baseline':'E17','complete':len(checks)==40,'cases':checks,'no_GT_vessel_or_GT_fallback':True});write_json(out/'PREDICTIONS_HASHED_BEFORE_EVAL_GT.json',{'prediction_tree_sha256':sha256_tree(out/'predictions'),'model_sha256':lock['model_sha256'],'baseline':'E17','GT_not_read':True,'n_cases':40,'seconds':time.monotonic()-start})
def main(arm):
 with (RUN/arm/'INFERENCE.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  _main(arm)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['normalized','physical'],required=True);main(p.parse_args().arm)
