"""Read-only oblique thin-slab inspection of present versus disconnected OA."""
from pathlib import Path
import json,os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np,nibabel as nib
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[2];D=P.parent/'data_topaneu26';R=P/'artifacts/research_audit_20260909/neck_slices';R.mkdir(exist_ok=True);out=[]
for cid,cl,branch,mother in [('topaneu_center2_mr_016',25,34,[6,36]),('topaneu_center2_mr_029',24,33,[4,35])]:
 image=nib.load(str(D/f'images/{cid}_0000.nii.gz'));a=image.get_fdata(dtype=np.float32);aff=image.affine;inv=np.linalg.inv(aff);g=np.asanyarray(nib.load(str(D/f'location_masks/{cid}.nii.gz')).dataobj);v=np.asanyarray(nib.load(str(P/f'artifacts/ta36_mr_center2_output/{cid}.nii.gz')).dataobj);pts=np.argwhere(g==cl);center=pts.mean(0)@aff[:3,:3].T+aff[:3,3];b=np.argwhere(v==branch);world=b@aff[:3,:3].T+aff[:3,3];near=world[np.linalg.norm(world-center,axis=1).argmin()];u=(near-center)/np.linalg.norm(near-center);seed=np.array([0.,0.,1.]) if abs(u[2])<.9 else np.array([0.,1.,0.]);normal=np.cross(u,seed);normal/=np.linalg.norm(normal);w=np.cross(normal,u);q=(np.arange(160)-79.5)*.25;xx,yy=np.meshgrid(q,q,indexing='ij');base=center[:,None,None]+u[:,None,None]*xx+w[:,None,None]*yy;fig,axes=plt.subplots(2,3,figsize=(12,8));allplanes=[]
 for j,offset in enumerate([-1.,0.,1.]):
  planes=[];labels=[]
  for slab in [-.5,0.,.5]:
   coords=(inv[:3,:3]@(base+normal[:,None,None]*(offset+slab)).reshape(3,-1)+inv[:3,3,None]);planes.append(map_coordinates(a,coords,order=1,mode='constant',cval=0,prefilter=False).reshape(160,160));labels.append((map_coordinates(g,coords,order=0,mode='constant',cval=0,prefilter=False).reshape(160,160),map_coordinates(v,coords,order=0,mode='constant',cval=0,prefilter=False).reshape(160,160)))
  mip=np.max(planes,axis=0);allplanes.append((mip,labels));nz=mip[mip>0];lo,hi=np.percentile(nz,[30,99.95])
  for k in [0,1]:
   ax=axes[k,j];ax.imshow(mip.T,origin='lower',cmap='gray',vmin=lo,vmax=hi,extent=[-20,20,-20,20]);ax.set_title(f'Plane offset {offset:+.0f}mm; slab1mm');ax.set_xlabel('toward nearest predicted OA (mm)');ax.set_ylabel('orthogonal in-plane (mm)')
   if k:
    for kind,color in [('aneurysm','red'),('OA','cyan'),('parent','orange')]:
     mask=np.any([gl==cl if kind=='aneurysm' else vl==branch if kind=='OA' else np.isin(vl,mother) for gl,vl in labels],axis=0)
     if mask.any() and not mask.all():ax.contour(mask.T,levels=[.5],colors=[color],linewidths=.8,origin='lower',extent=[-20,20,-20,20])
  axes[0,j].plot([0],[0],marker='+',color='red',markersize=5)
 fig.suptitle(f'{cid}, GT class{cl}; raw / GT aneurysm red + predicted OA cyan + parent orange\nOblique plane toward nearest predicted OA; projected proximity is not clinical neck adjudication',fontsize=10);fig.tight_layout(rect=[0,0,1,.93]);path=R/f'{cid}.png';fig.savefig(path,dpi=140);plt.close(fig);out.append({'case_id':cid,'GT_class':cl,'branch_id':branch,'nearest_predicted_OA_center_distance_mm':float(np.linalg.norm(near-center)),'image':str(path),'read_only':True,'no_GT_vessel':True});print(cid,flush=True)
(R/'RESULT.json').write_text(json.dumps({'rows':out,'limitation':'Two targeted repeated-MR40 diagnostic images, not training data or independent clinical annotation validation'},indent=2))
