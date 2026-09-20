"""396 geometric features from predicted tumor surface and predicted vessel labels."""
import numpy as np
from scipy.ndimage import binary_erosion
from scripts.astra6_e01.e01_common import affine_world
from scripts.astra6_e04.run_e04 import PRIOR,SUPPORT,largest
FEATURE_VERSION='mask_contact_v1'
FIELDS=['surface_min_distance50','surface_p10_distance50','surface_mean_distance50','surface_fraction_within1mm','surface_fraction_within2mm','contact_offset_x','contact_offset_y','contact_offset_z','contact_segment_relative_x','contact_segment_relative_y','contact_segment_relative_z']
def mask_from_probability(prob):
 mask=largest((prob>=.5)&(SUPPORT>0));return mask if mask.any() else PRIOR.astype(bool)
def contact_features(geometry,affine,low,high,mask):
 pts=np.argwhere(mask & ~binary_erosion(mask))
 if not len(pts):pts=np.argwhere(PRIOR>0)
 if len(pts)>256:pts=pts[np.linspace(0,len(pts)-1,256,dtype=int)]
 center=(np.asarray(low)+np.asarray(high))/2;extent=np.maximum(np.asarray(high)-np.asarray(low),1);vox=center+(pts-15.5)/32*2*extent;world=affine_world(affine,vox);q=affine_world(affine,center);features=[]
 for v in range(1,37):
  tree=geometry['trees'][v]
  if tree is None:features.extend([1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.]);continue
  dist,index=tree.query(world);mind=float(dist.min());near=dist<=mind+1.;contact=np.mean(tree.data[np.asarray(index)[near]],axis=0);off=np.clip(contact-q,-50,50)/50;rel=np.clip((contact-geometry['segment_m'][v])/geometry['segment_s'][v],-1.5,1.5);features.extend([min(mind,50)/50,min(float(np.percentile(dist,10)),50)/50,float(np.minimum(dist,50).mean())/50,float(np.mean(dist<=1)),float(np.mean(dist<=2)),*off,*rel])
 result=np.asarray(features,np.float32);assert result.shape==(396,) and np.isfinite(result).all();return result

def mirror_features(f,vessel_pair):
 a=np.asarray(f).reshape(36,11);out=a[[vessel_pair[v]-1 for v in range(1,37)]].copy();out[:,5]*=-1;out[:,8]*=-1;return out.reshape(-1)

def geometry_test():
 from scipy.spatial import cKDTree
 rng=np.random.default_rng(90210);pair={v:(v+1 if v%2 else v-1) for v in range(1,37)};g={'trees':{},'segment_m':{},'segment_s':{}}
 for v in range(1,37):
  points=rng.normal(0,8,(80,3));g['trees'][v]=cKDTree(points);g['segment_m'][v]=points.mean(0);g['segment_s'][v]=points.std(0)+1
 aff=np.eye(4);lo=np.array([-3.,-4.,-5.]);hi=-lo;f=contact_features(g,aff,lo,hi,PRIOR>0);s=np.array([-1.,1.,1.]);gm={'trees':{},'segment_m':{},'segment_s':{}}
 for v in range(1,37):
  src=pair[v];gm['trees'][v]=cKDTree(g['trees'][src].data*s);gm['segment_m'][v]=g['segment_m'][src]*s;gm['segment_s'][v]=g['segment_s'][src]
 fm=contact_features(gm,np.diag([-1.,1.,1.,1.]),lo,hi,PRIOR>0);assert np.allclose(fm,mirror_features(f,pair),atol=1e-6);assert np.array_equal(f,mirror_features(mirror_features(f,pair),pair));return {'physical_RAS_reflection_verified':True,'mirror_involution':True,'features':396,'max_surface_points':256}
