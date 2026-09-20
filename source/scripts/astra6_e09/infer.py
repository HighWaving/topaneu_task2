import json,subprocess,sys,time
from scripts.astra6_e09.prepare import RUN,P,DATA
from scripts.astra6_e01.e01_common import sha256_file,sha256_tree,write_json
ids=json.loads((RUN/'source_split.json').read_text())['fixed_CT5'];dest=RUN/'predictions_ct';start=time.time();lock=json.loads((RUN/'model/LOCKED.json').read_text());assert sha256_file(RUN/'model/classifier.joblib')==lock['model_sha256']
for cid in ids:
 cmd=[sys.executable,'-m','scripts.delivery.refine','--modality','CT','--image',str(DATA/f'images/{cid}_0000.nii.gz'),'--predicted-vessel',str(P/f'artifacts/ta36_ct_output/{cid}.nii.gz'),'--boxes',str(P/f'artifacts/ct_fold2_all109_boxes/{cid}_boxes.pkl'),'--classifier',str(RUN/'model/classifier.joblib'),'--segmentation',str(P/'artifacts/astra6_e04_crop_segmentation_20260909T031146Z/model/final_last.pt'),'--device','cuda:0','--output',str(dest/f'{cid}.nii.gz')];subprocess.run(cmd,cwd=P,check=True);print('CT inference',cid,flush=True)
write_json(RUN/'PREDICTIONS_LOCKED.json',{'n_cases':5,'predictions_sha256':sha256_tree(dest),'classifier_sha256':lock['model_sha256'],'seconds':time.time()-start,'GT_not_read':True})
