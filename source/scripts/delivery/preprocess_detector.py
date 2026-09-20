"""Image-only nnDetection preprocessing using the frozen training plan."""
import argparse,pickle
from pathlib import Path
import numpy as np
from nndet.preprocessing.preprocessor import GenericPreprocessor
p=argparse.ArgumentParser();p.add_argument('--image',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
plan=pickle.load(a.plan.open('rb'));pre=GenericPreprocessor(norm_scheme_per_modality=plan['normalization_schemes'],use_mask_for_norm=plan['use_mask_for_norm'],transpose_forward=plan['transpose_forward'],intensity_properties=plan['dataset_properties']['intensity_properties'],resample_anisotropy_threshold=plan['resample_anisotropy_threshold'])
data,_,props=pre.preprocess_test_case([str(a.image)],plan['target_spacing']);a.out.mkdir(parents=True,exist_ok=True)
np.savez_compressed(a.out/'case.npz',data=data)
with (a.out/'case.pkl').open('wb') as f:pickle.dump(props,f)
