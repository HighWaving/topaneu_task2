import json
import numpy as np
from scripts.astra6_e11.common import *
from scripts.astra6_e01.e01_common import load_boxes,load_nifti_geometry,sha256_file,write_json
split=json.loads((RUN/'source_split.json').read_text());ids=split['train_cases']+split['development_cases'];out={}
for cid in ids:
 folder=P/('artifacts/fold1_eval_center1_epoch60' if '_mr_' in cid else 'artifacts/ct_fold2_all109_boxes');box=folder/f'{cid}_boxes.pkl';_,_,o=load_boxes(box);img=DATA/f'images/{cid}_0000.nii.gz';gt=DATA/f'location_masks/{cid}.nii.gz';a,shape=load_nifti_geometry(img);g,gsh=load_nifti_geometry(gt);assert shape==gsh and tuple(o['original_size_of_raw_data'])==shape[::-1],cid
 old=np.eye(4);old[:3,:3]=np.reshape(o['itk_direction'],(3,3))@np.diag(o['itk_spacing']);old[:3,3]=o['itk_origin'];old=np.diag([-1,-1,1,1])@old
 xyz=np.array(np.meshgrid(*[(0,d-1) for d in shape],indexing='ij')).reshape(3,-1).T;errors={k:float(np.linalg.norm(xyz@(a[:3,:3]-b[:3,:3]).T+a[:3,3]-b[:3,3],axis=1).max()) for k,b in [('GT',g),('cached_detector',old)]};assert max(errors.values())<=.001,(cid,errors)
 out[cid]={'corner_error_mm':errors,'label_sha256':sha256_file(gt),'box_sha256':sha256_file(box),'image_size_bytes':img.stat().st_size,'image_mtime_ns':img.stat().st_mtime_ns,'image_affine':a.tolist(),'shape':shape}
write_json(RUN/'SOURCE_GEOMETRY_VALIDATED.json',{'n_cases':len(ids),'max_allowed_corner_error_mm':.001,'max_observed_error_mm':max(max(r['corner_error_mm'].values()) for r in out.values()),'cases':out});print('All source cached detector and GT coordinates validated',len(ids),flush=True)
