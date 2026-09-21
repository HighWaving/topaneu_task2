"""Equivalent E02 vessel geometry from one sparse foreground scan."""
import numpy as np
from scipy.spatial import cKDTree
from scripts.astra6_e01.e01_common import load_nifti,affine_world

def vessel_geometry_fast(vessel_path,image_shape,image_affine,allow_empty=True):
 arr,aff,shape=load_nifti(vessel_path)
 if shape!=image_shape or not np.allclose(aff,image_affine,atol=1e-4):raise ValueError('vessel/image geometry mismatch')
 coords=np.argwhere(arr>0)
 if not len(coords):
  if allow_empty:return None
  raise ValueError('empty predicted vessel union')
 labels=arr[tuple(coords.T)];world=affine_world(aff,coords);p01,p99=np.percentile(world,[1,99],axis=0);gm=(p01+p99)/2;gs=np.maximum(p99-p01,1.)
 trees={};sm={};ss={}
 for v in range(1,37):
  c=coords[labels==v]
  if len(c):
   pts=affine_world(aff,c);trees[v]=cKDTree(pts);lo,hi=np.percentile(pts,[5,95],axis=0);sm[v]=(lo+hi)/2;ss[v]=np.maximum(hi-lo,1.)
  else:trees[v]=None;sm[v]=np.zeros(3);ss[v]=np.ones(3)
 return {'global_m':gm,'global_s':gs,'trees':trees,'segment_m':sm,'segment_s':ss}
