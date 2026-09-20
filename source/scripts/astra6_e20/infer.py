import json,time
from scripts.astra6_e20.prepare import RUN,P,DATA
from scripts.astra6_e13.infer import native
from scripts.astra6_e01.e01_common import sha256_tree,sha256_file,write_json
R13=P/'artifacts/astra6_e13_CT_anatomical_fp_filter_20260909';F11=P/'artifacts/astra6_e11_CT_expanded_fp_filter_20260909';C=P/'artifacts/astra6_e09_CT_expanded_location_20260909/model/classifier.joblib'
def main():
 ids=json.loads((RUN/'source_split.json').read_text())['fixed_CT5'];threshold=json.loads((R13/'model/THRESHOLD.json').read_text());lock=json.loads((RUN/'model/LOCKED.json').read_text());assert sha256_file(RUN/'model/final_last.pt')==lock['model_sha256'];start=time.time()
 write_json(RUN/'INFERENCE_INPUTS_LOCKED.json',{'segmentation_sha256':lock['model_sha256'],'classifier_sha256':sha256_file(C),'image_filter_sha256':sha256_file(F11/'model/final_last.pt'),'anatomical_filter_sha256':sha256_file(R13/'model/final.joblib'),'threshold_sha256':sha256_file(R13/'model/THRESHOLD.json'),'boxes':{c:sha256_file(P/f'artifacts/ct_fold2_all109_boxes/{c}_boxes.pkl') for c in ids},'predicted_vessels':{c:sha256_file(P/f'artifacts/ta36_ct_output/{c}.nii.gz') for c in ids},'GT_not_input':True})
 for cid in ids:
  native(DATA/f'images/{cid}_0000.nii.gz',P/f'artifacts/ta36_ct_output/{cid}.nii.gz',P/f'artifacts/ct_fold2_all109_boxes/{cid}_boxes.pkl',C,RUN/'model/final_last.pt',F11/'model/final_last.pt',threshold['parent_image_threshold'],R13/'model/final.joblib',threshold['threshold'],RUN/f'predictions_ct/{cid}.nii.gz','cuda:0');print('CT_E20',cid,flush=True)
 write_json(RUN/'PREDICTIONS_LOCKED.json',{'prediction_tree_sha256':sha256_tree(RUN/'predictions_ct'),'GT_not_read':True,'cases':ids,'seconds':time.time()-start,'segmentation_sha256':lock['model_sha256']})
if __name__=='__main__':main()
