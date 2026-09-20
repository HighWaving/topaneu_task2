import json,subprocess,sys,time
from pathlib import Path
from scripts.astra6_e06.common import RUN,P,DATA
from scripts.astra6_e01.e01_common import sha256_tree,sha256_file,write_json
root=P/'artifacts/ct_independent5_20260909';ids=sorted(p.name.removesuffix('.nii.gz') for p in (root/'E04').glob('*.nii.gz'));assert len(ids)==5 and all('center2_ct' in c for c in ids);threshold=json.loads((RUN/'model/THRESHOLD.json').read_text())['threshold'];dest=RUN/'predictions_ct';dest.mkdir(exist_ok=True);start=time.time()
for cid in ids:
 output=dest/f'{cid}.nii.gz'
 if output.exists() and output.with_suffix('.json').exists():continue
 cmd=[sys.executable,'-m','scripts.delivery.refine_with_filter','--modality','CT','--image',str(DATA/f'images/{cid}_0000.nii.gz'),'--predicted-vessel',str(P/f'artifacts/ta36_ct_output/{cid}.nii.gz'),'--boxes',str(P/f'artifacts/ct_fold2_all109_boxes/{cid}_boxes.pkl'),'--classifier',str(P/'artifacts/astra6_e02_multiscale_vessel_signature_20260908T173916Z/model/classifier.joblib'),'--segmentation',str(P/'artifacts/astra6_e04_crop_segmentation_20260909T031146Z/model/final_last.pt'),'--fp-filter',str(RUN/'model/final_last.pt'),'--fp-threshold',str(threshold),'--device','cuda:0','--output',str(output)];subprocess.run(cmd,cwd=P,check=True);print('CT done',cid,flush=True)
write_json(RUN/'CT_PREDICTIONS_LOCKED.json',{'n_cases':5,'cases':ids,'prediction_tree_sha256':sha256_tree(dest),'model_sha256':sha256_file(RUN/'model/final_last.pt'),'seconds':time.time()-start,'no_GT_access':True})
