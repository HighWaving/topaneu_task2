"""Read-only bounded MR40 error inspection, never training or relabeling."""
from pathlib import Path
import json,os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np,nibabel as nib
from scipy.ndimage import label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[2];D=P.parent/'data_topaneu26';R=P/'artifacts/research_audit_20260909/location_images';R.mkdir(parents=True,exist_ok=True)
rows=json.loads((P/'reports/MR_E17_LOCATION_BOTTLENECK_20260909.json').read_text())['rows'];rows=[r for r in rows if (22<=r['GT_class']<=35 and 22<=r['predicted_class']<=35) or (45<=r['GT_class']<=52 and 45<=r['predicted_class']<=52)]
plan={'hypothesis':'C02 geometric summaries may omit visible fine branch/neck context; TA36 absence may destroy the required information. Inspect raw image and predicted anatomy before selecting representation.','cases':sorted({r['case_id'] for r in rows}),'n_error_lesions':len(rows),'read_only_repeated_development_evidence':True,'no_GT_vessel':True,'outputs':'32/80mm raw-image MIPs with GT aneurysm and predicted-vessel overlays; branch counts/distances. MIP contact is not proof of actual neck attachment.','budget':'one CPU worker, no GPU, one pass per case'}
(R/'PLAN.json').write_text(json.dumps(plan,indent=2));results=[]
for cid in plan['cases']:
 im=nib.load(str(D/f'images/{cid}_0000.nii.gz'));a=im.get_fdata(dtype=np.float32);aff=im.affine;spacing=np.linalg.norm(aff[:3,:3],axis=0);gim=nib.load(str(D/f'location_masks/{cid}.nii.gz'));vim=nib.load(str(P/f'artifacts/ta36_mr_center2_output/{cid}.nii.gz'));g=np.asanyarray(gim.dataobj);v=np.asanyarray(vim.dataobj);assert a.shape==g.shape==v.shape and np.allclose(aff,gim.affine,atol=1e-4) and np.allclose(aff,vim.affine,atol=1e-4)
 sub=a[::4,::4,::4];nz=sub[sub!=0];vmin,vmax=np.percentile(nz,[.5,99.5]);case_rows=[r for r in rows if r['case_id']==cid]
 for row in case_rows:
  cl=row['GT_class'];points=np.argwhere(g==cl);cc,n=label(g==cl,np.ones((3,3,3)));assert n==1,f'Multiple same-class lesions need explicit matching {cid} {cl}'
  center=points.mean(0);branches=([33,34,31,32,4,6,35,36,8,9] if cl<45 else [5,7,17,19,18,20]);stats={}
  for b in branches:
   xyz=np.argwhere(v==b);delta=(xyz-center)@aff[:3,:3].T if len(xyz) else np.empty((0,3));stats[str(b)]={'whole_volume_voxels':len(xyz),'nearest_center_mm':float(np.linalg.norm(delta,axis=1).min()) if len(xyz) else None,'within32mm_cube_voxels':int(np.all(np.abs((xyz-center)*spacing)<=16,axis=1).sum()),'within80mm_cube_voxels':int(np.all(np.abs((xyz-center)*spacing)<=40,axis=1).sum())}
  fig,axes=plt.subplots(2,3,figsize=(13,9));colors={b:plt.cm.tab10(i%10) for i,b in enumerate(branches)}
  for j,width in enumerate([32,80]):
   lo=np.maximum(np.floor(center-width/2/spacing).astype(int),0);hi=np.minimum(np.ceil(center+width/2/spacing).astype(int)+1,a.shape);sl=tuple(slice(x,y) for x,y in zip(lo,hi));crop=a[sl];gc=g[sl]==cl;vc=v[sl]
   for axis in range(3):
    ax=axes[j,axis];remaining=[i for i in range(3) if i!=axis];extent=[0,crop.shape[remaining[0]]*spacing[remaining[0]],0,crop.shape[remaining[1]]*spacing[remaining[1]]];ax.imshow(crop.max(axis=axis).T,origin='lower',cmap='gray',vmin=vmin,vmax=vmax,extent=extent)
    def outline(mask,color):
     m=mask.any(axis=axis).T
     if m.any() and not m.all():ax.contour(m,levels=[.5],colors=[color],linewidths=.65,origin='lower',extent=extent)
    outline(gc,'red')
    for b in branches:outline(vc==b,colors[b])
    ax.set_title(f'{width}mm FOV; native axis {axis} MIP');ax.set_xlabel('mm');ax.set_ylabel('mm')
  fig.suptitle(f'{cid}: GT {cl} -> C02 {row["predicted_class"]}\nGT aneurysm red; predicted vessel IDs '+', '.join(map(str,branches))+' (tab10 order)\nNative axes; projected overlap does not establish neck attachment',fontsize=10);fig.tight_layout(rect=[0,0,1,.92]);path=R/f'{cid}_GT{cl}.png';fig.savefig(path,dpi=120);plt.close(fig)
  results.append({**row,'spacing_mm':spacing.tolist(),'GT_voxels':len(points),'GT_center_native':center.tolist(),'branch_evidence':stats,'image':str(path),'raw_branch_visibility':'requires visual review; do not infer absence from TA36 mask','annotation_class_present':True})
 (R/'RESULT.json').write_text(json.dumps({'complete':len(results)==len(rows),'plan':plan,'rows':results},indent=2));print(cid,len(results),len(rows),flush=True)
 del a,g,v,cc
