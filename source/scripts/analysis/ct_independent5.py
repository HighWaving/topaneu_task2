"""Five CT center2 cases excluded from detector and E02/E04 source fitting."""
from pathlib import Path
import json,time
import numpy as np,nibabel as nib
from scripts.delivery.refine import predict
from scripts.astra6_e01.e01_common import P,DATA,write_json,sha256_tree,sha256_file,sitk_array
from scripts.astra6_e03.run_e03 import BASE,read
from scripts.local_scoring_arena import score_case,aggregate
R=P/'artifacts/ct_independent5_20260909';R.mkdir(exist_ok=True);ids=[x for x in read(P/'CT_SPLIT_AUDIT_20260909.json')['2']['validation_ids'] if 'center2' in x]
source={json.loads(x)['case_id'] for x in (BASE/'features/train_records.jsonl').read_text().splitlines()};assert len(ids)==5 and not source&set(ids)
write_json(R/'cohort.json',{'cases':ids,'source_overlap':False,'only5_cases':True,'selection':'all center2 CT cases in preexisting detector fold2 val; center4 excluded due downstream source fitting','not_pristine_test':True})
seg=P/'artifacts/astra6_e04_crop_segmentation_20260909T031146Z/model/final_last.pt';start=time.monotonic()
for version,s in [('E02',None),('E04',seg)]:
 for cid in ids:
  out=R/f'{version}/{cid}.nii.gz';out.parent.mkdir(exist_ok=True)
  if out.exists():continue
  image=DATA/f'images/{cid}_0000.nii.gz';vessel=P/f'artifacts/ta36_ct_output/{cid}.nii.gz';boxes=P/f'artifacts/ct_fold2_all109_boxes/{cid}_boxes.pkl'
  mask,ledger=predict(image,vessel,boxes,BASE/'model/classifier.joblib',s,'cuda:0','CT');im=nib.load(image);nib.Nifti1Image(mask.astype(np.uint8),im.affine).to_filename(out);write_json(out.with_suffix('.json'),ledger);print(version,cid,flush=True)
write_json(R/'PREDICTIONS_LOCKED.json',{v:sha256_tree(R/v) for v in ['E02','E04']})
for version in ['E02','E04']:
 pc=[{'case_id':cid,'raw':score_case(sitk_array(R/f'{version}/{cid}.nii.gz'),cid)} for cid in ids];write_json(R/f'{version}_per_case.json',pc);write_json(R/f'{version}_official.json',aggregate([x['raw'] for x in pc]))
write_json(R/'runtime.json',{'seconds':time.monotonic()-start,'includes':'cached boxes and predicted vessels refinement+scoring; not upstream full pipeline'})
