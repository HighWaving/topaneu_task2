"""Reprocess an updated source label in a private tree with the frozen detector plan."""
import argparse,pickle,json
from pathlib import Path
import numpy as np
from nndet.preprocessing.preprocessor import GenericPreprocessor
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args();v=a.root.parents[2];data=v/'data_topaneu26';old=v/'nndet_data/Task030FG_TopAneuMR';out=a.root/'reprocessed'/a.case;out.mkdir(parents=True,exist_ok=True);plan=pickle.loads((old/'preprocessed/D3V001_3d.pkl').read_bytes());pre=GenericPreprocessor(norm_scheme_per_modality=plan['normalization_schemes'],use_mask_for_norm=plan['use_mask_for_norm'],transpose_forward=plan['transpose_forward'],intensity_properties=plan['dataset_properties']['intensity_properties'],resample_anisotropy_threshold=plan['resample_anisotropy_threshold']);x,y,props=pre.preprocess_test_case([str(data/f'images/{a.case}_0000.nii.gz')],plan['target_spacing'],seg_file=str(a.root/f'corrected_native_labels/{a.case}.nii.gz'));props['use_nonzero_mask_for_norm']=pre.use_mask_for_norm;boxes=pre.compute_candidates(data=x,seg=y,properties=props);np.savez_compressed(out/f'{a.case}.npz',data=x,seg=y);np.save(out/f'{a.case}.npy',x);np.save(out/f'{a.case}_seg.npy',y)
for suffix,item in [('.pkl',props),('_boxes.pkl',boxes)]:
 with (out/f'{a.case}{suffix}').open('wb') as f:pickle.dump(item,f)
(out/'READY.json').write_text(json.dumps({'case_id':a.case,'shape':list(x.shape),'n_instances':len(boxes['instances']),'source_only':True,'existing_files_untouched':True},indent=2)+'\n');print('E23_REPROCESSED',a.case,x.shape,flush=True)
