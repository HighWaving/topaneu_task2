from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
P=Path(__file__).resolve().parents[2];RUN=P/'artifacts/astra6_e32_image_only_segmentation_20260910';N=48

def grid_geometry(lo,hi,aff,arm):
 lo,hi=np.asarray(lo,float),np.asarray(hi,float);center=(lo+hi)/2;extent=np.maximum(hi-lo,1);spacing=np.linalg.norm(aff[:3,:3],axis=0)
 directions=aff[:3,:3]/spacing;assert np.allclose(directions.T@directions,np.eye(3),atol=1e-4),'Sheared image requires an explicit world-coordinate implementation'
 if arm=='normalized':field=2*extent
 else:
  assert arm=='physical';mm=max(19.2,9.6*np.ceil(1.5*np.max(extent*spacing)/9.6));field=np.repeat(mm,3)/spacing
 step=field/N;origin=center-(N-1)/2*step
 return origin,step

def coords_for(origin,step):
 return np.stack(np.meshgrid(*[origin[i]+np.arange(N)*step[i] for i in range(3)],indexing='ij'))
def input_crop(image,vessel,lo,hi,aff,arm):
 origin,step=grid_geometry(lo,hi,aff,arm);coords=coords_for(origin,step);flat=coords.reshape(3,-1)
 im=map_coordinates(image,flat,order=1,mode='constant',cval=0,prefilter=False).reshape((N,)*3)
 ve=map_coordinates(vessel,flat,order=0,mode='constant',cval=0,prefilter=False).reshape((N,)*3)>0
 center=(np.array(lo)+hi)/2;sigma=np.maximum((np.array(hi)-lo)/2,1)
 cue=np.exp(-.5*np.sum(((coords-center[:,None,None,None])/sigma[:,None,None,None])**2,axis=0))
 return np.stack([im,ve,cue]).astype(np.float16),origin,step

def target_crop(component,origin,step):
 lo=component.min(0)-1;hi=component.max(0)+2;small=np.zeros(tuple(hi-lo),np.uint8);small[tuple((component-lo).T)]=1
 c=coords_for(origin,step)-lo[:,None,None,None]
 return map_coordinates(small,c.reshape(3,-1),order=0,mode='constant',cval=0,prefilter=False).reshape((N,)*3).astype(np.uint8)

def native_foreground(prob,origin,step,image_shape):
 from scripts.astra6_e04.run_e04 import largest
 lo=np.maximum(0,np.ceil(origin).astype(int));hi=np.minimum(image_shape,np.floor(origin+(N-1)*step).astype(int)+1)
 if np.any(hi<=lo):return lo,np.zeros((0,0,0),bool)
 coords=np.stack(np.meshgrid(*[(np.arange(lo[i],hi[i])-origin[i])/step[i] for i in range(3)],indexing='ij'))
 pr=map_coordinates(prob,coords.reshape(3,-1),order=1,mode='constant',cval=0,prefilter=False).reshape(tuple(hi-lo))
 return lo,largest(pr>=.5)
