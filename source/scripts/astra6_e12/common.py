from scripts.astra6_e06.common import *
RUN=P/'artifacts/astra6_e12_MR_expanded_fp_filter_20260909'

def nearest_label_identity(image_affine,label_affine,shape):
 # An affine coordinate difference is bounded on a rectangular grid by its eight corners.
 # If each coordinate stays strictly within half a voxel of the same integer,
 # nearest-neighbor physical resampling produces exactly the unchanged label array.
 corners=np.array(np.meshgrid(*[(0,d-1) for d in shape],indexing='ij')).reshape(3,-1).T
 transform=np.linalg.solve(label_affine,image_affine);delta=corners@(transform[:3,:3]-np.eye(3)).T+transform[:3,3];bound=np.max(np.abs(delta),axis=0)
 return bool(np.all(bound<.499999)),bound.tolist()
