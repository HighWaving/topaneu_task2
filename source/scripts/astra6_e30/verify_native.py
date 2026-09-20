"""Explicit-input CPU parity/read-scope check on one changed OA case; no new evaluation."""
import json,sys,runpy,hashlib,resource,time
from pathlib import Path
P=Path(__file__).resolve().parents[2];RUN=P/'artifacts/astra6_e30_expanded_CT_supervision_for_MR_20260910';DATA=P.parent/'data_topaneu26';S=P/'artifacts/astra6_e17_MR_detector_crop_segmentation_20260909';F=P/'artifacts/astra6_e14_MR_filter_seed_stability_20260909';cid='topaneu_center2_mr_016'
def main():
 import nibabel as nib,numpy as np
 out=RUN/'native_verification'/f'{cid}.nii.gz';out.parent.mkdir(exist_ok=True)
 image=DATA/f'images/{cid}_0000.nii.gz';vessel=P/f'artifacts/ta36_mr_center2_output/{cid}.nii.gz';boxes=P/f'artifacts/fold1_eval_center2_epoch60/{cid}_boxes.pkl';classifier=(RUN/'model/classifier.joblib').resolve();seg=S/'model/final_last.pt';fp=F/'model/final_last.pt'
 threshold_file=F/'model/THRESHOLD.json';lock=json.loads((F/'model/LOCKED.json').read_text());assert hashlib.sha256(threshold_file.read_bytes()).hexdigest()==lock['threshold_sha256'];threshold=json.loads(threshold_file.read_text())['threshold']
 allowed={q.resolve() for q in [image,vessel,boxes,classifier,seg,fp]};reads=set();active=[True]
 def audit(event,args):
  if not active[0] or event!='open' or not isinstance(args[0],(str,bytes)):return
  path=Path(args[0]).resolve();mode=args[1];write=isinstance(mode,str) and any(x in mode for x in ['w','a','+'])
  if not write and (path.is_relative_to(DATA) or path.is_relative_to(P/'artifacts')):
   if path not in allowed and not path.is_relative_to(out.parent):raise RuntimeError('Unexpected data/artifact read: '+str(path))
   reads.add(str(path))
 sys.addaudithook(audit);start=time.monotonic();sys.argv=['refine_with_filter','--image',str(image),'--predicted-vessel',str(vessel),'--boxes',str(boxes),'--classifier',str(classifier),'--segmentation',str(seg),'--fp-filter',str(fp),'--fp-threshold',str(threshold),'--modality','MR','--device','cpu','--output',str(out)];runpy.run_module('scripts.delivery.refine_with_filter',run_name='__main__');active[0]=False
 a=nib.load(str(out));b=nib.load(str(RUN/f'predictions/mr_center2_k05/{cid}.nii.gz'));assert np.allclose(a.affine,b.affine,atol=1e-4) and np.array_equal(np.asanyarray(a.dataobj),np.asanyarray(b.dataobj));result={'passed':True,'case_id':cid,'candidate_parity_native_exact':True,'explicit_input_only_inference_read_scope':True,'allowed_reads':sorted(reads),'CPU_functional_check_not_T4_runtime_validation':True,'seconds':time.monotonic()-start,'peak_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'research_candidate_only_r2_unchanged':True};(out.parent/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
