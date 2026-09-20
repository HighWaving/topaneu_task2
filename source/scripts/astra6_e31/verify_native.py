"""One explicit-file GPU native refinement/read-scope check; compare after research inference."""
import argparse,json,runpy,sys,time
from pathlib import Path
import nibabel as nib,numpy as np
from scripts.astra6_e31.common import P,RUN
from scripts.astra6_e01.e01_common import sha256_file

def main(arm):
 cid='topaneu_center2_mr_016';outdir=RUN/arm/'native_verification';outdir.mkdir(exist_ok=True);out=outdir/f'{cid}.nii.gz';data=P.parent/'data_topaneu26';image=data/f'images/{cid}_0000.nii.gz';vessel=P/f'artifacts/ta36_mr_center2_output/{cid}.nii.gz';boxes=P/f'artifacts/fold1_eval_center2_epoch60/{cid}_boxes.pkl';classifier=P/'artifacts/astra6_e02_multiscale_vessel_signature_20260908T173916Z/model/classifier.joblib';seg=RUN/arm/'model/final_last.pt';backup=P/'artifacts/astra6_e17_MR_detector_crop_segmentation_20260909/model/final_last.pt';fr=P/'artifacts/astra6_e14_MR_filter_seed_stability_20260909/model';fp=fr/'final_last.pt';threshold=json.loads((fr/'THRESHOLD.json').read_text())['threshold'];assert sha256_file(fr/'THRESHOLD.json')==json.loads((fr/'LOCKED.json').read_text())['threshold_sha256'];assert (RUN/arm/'model/LOCKED.json').exists();scope=outdir/'READ_SCOPE.json'
 if not scope.exists():
  allowed={x.resolve() for x in [image,vessel,boxes,classifier,seg,backup,fp]};active=[True];reads=set()
  def audit(event,args):
   if not active[0] or event!='open' or not isinstance(args[0],(str,bytes)):return
   path=Path(args[0]).resolve();mode=args[1];write=isinstance(mode,str) and any(c in mode for c in ['w','a','+'])
   if not write and (path.is_relative_to(data) or path.is_relative_to(P/'artifacts')):
    if path not in allowed and not path.is_relative_to(outdir):raise RuntimeError('Unexpected data/artifact read: '+str(path))
    reads.add(str(path))
  sys.addaudithook(audit);sys.argv=['predict_native','--image',str(image),'--predicted-vessel',str(vessel),'--boxes',str(boxes),'--classifier',str(classifier),'--segmentation',str(seg),'--fallback-segmentation',str(backup),'--fp-filter',str(fp),'--fp-threshold',str(threshold),'--arm',arm,'--device','cuda:0','--output',str(out)];runpy.run_module('scripts.astra6_e31.predict_native',run_name='__main__');active[0]=False;scope.write_text(json.dumps({'passed':True,'explicit_input_only':True,'reads':sorted(reads),'model_sha256':sha256_file(seg),'output_sha256':sha256_file(out),'TA36_and_D_provided_as_explicit_stage_inputs':True,'not_T4_or_whole_raw_pipeline_runtime':True},indent=2)+'\n')
 reference=RUN/arm/f'predictions/mr_center2_k05/{cid}.nii.gz'
 if reference.exists():
  a=nib.load(str(out));b=nib.load(str(reference));same=np.allclose(a.affine,b.affine,atol=1e-4) and np.array_equal(np.asanyarray(a.dataobj),np.asanyarray(b.dataobj));assert same,'Explicit native entry differs from frozen research pipeline';(outdir/'PARITY.json').write_text(json.dumps({'passed':True,'native_exact':True,'case_id':cid,'GT_not_read':True,'arm':arm})+'\n')
 print('E31 explicit native entry',arm,'read-scope passed','parity',reference.exists(),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['normalized','physical'],required=True);main(p.parse_args().arm)
