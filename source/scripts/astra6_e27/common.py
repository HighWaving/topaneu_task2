"""Predicted-vessel junction anchors; pure inference geometry, no GT inputs."""
from pathlib import Path
import numpy as np
from scripts.astra6_e01.e01_common import affine_world
P=Path(__file__).resolve().parents[2];RUN=P/'artifacts/astra6_e27_junction_location_20260909'
VPAIR={1:1,10:10,15:15,16:16}
for a,b in [(2,3),(4,6),(5,7),(8,9),(11,12),(13,14),(17,19),(18,20),(21,22),(23,24),(25,26),(27,28),(29,30),(31,32),(33,34),(35,36)]:VPAIR[a]=b;VPAIR[b]=a
RIGHT=[(23,1),(23,29),(1,27),(1,25),(1,2),(2,21),(35,4),(4,33),(4,8),(4,31),(4,5),(4,11),(10,11),(11,13),(5,17),(17,18)]
def mirror_pair(pair):return tuple(VPAIR[v] for v in pair)
PAIRS=RIGHT+[mirror_pair(p) for p in RIGHT]+[(10,15),(15,16)]
RP=[((23,1),(1,25)),((1,25),(1,2)),((35,4),(4,33)),((4,33),(4,8)),((4,8),(4,31)),((4,31),(4,5)),((35,4),(4,5)),((4,5),(5,17)),((4,11),(10,11)),((10,11),(11,13)),((1,2),(2,21))]
PATHS=RP+[(mirror_pair(a),mirror_pair(b)) for a,b in RP]
N_FEATURES=6*len(PAIRS)+4*len(PATHS)
assert N_FEATURES==292 and len(set(PAIRS))==34
PERM=[];SIGN=[]
for pair in PAIRS:
 j=PAIRS.index(mirror_pair(pair));PERM.extend(range(6*j,6*j+6));SIGN.extend([1,1,-1,1,1,1])
for a,b in PATHS:
 j=PATHS.index((mirror_pair(a),mirror_pair(b)));PERM.extend(range(6*len(PAIRS)+4*j,6*len(PAIRS)+4*j+4));SIGN.extend([1]*4)
PERM=np.asarray(PERM);SIGN=np.asarray(SIGN,np.float32);assert np.array_equal(PERM[PERM],np.arange(N_FEATURES));assert np.all(SIGN*SIGN[PERM]==1)

def anchors(geometry):
 result={}
 for parent,child in PAIRS:
  a=geometry['trees'][parent];b=geometry['trees'][child]
  if a is None or b is None:result[parent,child]=None;continue
  points=b.data;dist=a.query(points,workers=1)[0];k=min(64,len(dist));cut=float(np.partition(dist,k-1)[k-1]);ix=dist<=cut+1e-6
  # Include all boundary-distance ties for reflection equivariance. Only child
  # coordinates are averaged, avoiding arbitrary nearest-parent voxel ties.
  d=dist[ix];w=1/(d+.25)**2;origin=(points[ix]*w[:,None]).sum(0)/w.sum();result[parent,child]=(origin,float(dist.min()))
 return result

def features(junctions,aff,low,high):
 q=affine_world(aff,(np.asarray(low)+np.asarray(high))/2);out=[]
 for pair in PAIRS:
  anchor=junctions[pair]
  if anchor is None:out.extend([0,1,0,0,0,1]);continue
  origin,gap=anchor;delta=q-origin;out.extend([1,min(gap,10)/10,*np.clip(delta/50,-1,1),min(float(np.linalg.norm(delta)),50)/50])
 for a,b in PATHS:
  aa,bb=junctions[a],junctions[b]
  if aa is None or bb is None:out.extend([0,0,0,1]);continue
  start,end=aa[0],bb[0];axis=end-start;length=float(np.linalg.norm(axis))
  if length<.5:out.extend([0,0,0,1]);continue
  projection=float(np.dot(q-start,axis)/(length*length));perpendicular=float(np.linalg.norm(q-start-projection*axis));out.extend([1,min(length,50)/50,float(np.clip(projection,-2,3)),min(perpendicular,50)/50])
 out=np.asarray(out,np.float32);assert out.shape==(292,) and np.isfinite(out).all();return out

def mirrored(x):return x[PERM]*SIGN

def synthetic_geometry_check():
 from scipy.spatial import cKDTree
 rng=np.random.default_rng(20260909);trees={v:cKDTree(rng.normal(size=(80,3))*3+np.array([v%3,0,0])) for v in range(1,37)};trees[33]=None
 geom={'trees':trees};a=anchors(geom);aff=np.eye(4);lo=np.array([-.5,1.,2.]);hi=lo+2;x=features(a,aff,lo,hi)
 reflected={'trees':{v:(cKDTree(trees[VPAIR[v]].data*np.array([-1.,1.,1.])) if trees[VPAIR[v]] is not None else None) for v in range(1,37)}};la,ha=lo.copy(),hi.copy();la[0],ha[0]=-hi[0],-lo[0];y=features(anchors(reflected),aff,la,ha);assert np.allclose(y,mirrored(x),atol=2e-6),(np.abs(y-mirrored(x)).max());assert np.array_equal(mirrored(mirrored(x)),x)
 return {'physical_reflection_equivariance':True,'mirror_involution':True,'missing_branch_handled':True,'features':292,'junction_pairs':len(PAIRS),'anchor_spans':len(PATHS)}
