"""Inspect two distinct saved CT failure mechanisms; no fit or new inference."""
from pathlib import Path
import json
import numpy as np,nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.analysis.paired_segmentation_diagnostics import components
from scripts.astra6_e01.e01_common import load_boxes,box_to_native_bounds
P=Path(__file__).resolve().parents[2];D=P.parent/'data_topaneu26';E=P/'artifacts/astra6_e16_CT_expanded_segmentation_20260909';R=P/'artifacts/research_audit_20260909/ct_targeted_images';R.mkdir(exist_ok=True)
rows=[]
for cid in ['topaneu_center2_ct_125','topaneu_center2_ct_190']:
 im=nib.load(str(D/f'images/{cid}_0000.nii.gz'));arr=im.get_fdata(dtype=np.float32);aff=im.affine;sp=np.linalg.norm(aff[:3,:3],axis=0)
 gi=nib.load(str(D/f'location_masks/{cid}.nii.gz'));pi=nib.load(str(E/f'predictions_ct/{cid}.nii.gz'));vi=nib.load(str(P/f'artifacts/ta36_ct_output/{cid}.nii.gz'));assert all(np.allclose(x.affine,aff,atol=1e-4) and x.shape==im.shape for x in [gi,pi,vi]);gt=np.asanyarray(gi.dataobj);pred=np.asanyarray(pi.dataobj);vessel=np.asanyarray(vi.dataobj);gc=components(gt);pc=components(pred);assert len(gc)==1;g=gc[0];overlaps=[len(np.intersect1d(g['flat'],c['flat'],assume_unique=True)) for c in pc];matched=pc[int(np.argmax(overlaps))] if max(overlaps,default=0)>0 else None
 center=g['coords'].mean(0);lo=np.maximum(np.floor(center-12/sp).astype(int),0);hi=np.minimum(np.ceil(center+12/sp).astype(int)+1,arr.shape);sl=tuple(slice(int(x),int(y)) for x,y in zip(lo,hi));image=arr[sl];gm=gt[sl]>0;pm=np.zeros(image.shape,bool)
 if matched is not None:
  pts=matched['coords'];pts=pts[np.all((pts>=lo)&(pts<hi),axis=1)];pm[tuple((pts-lo).T)]=True
 boxes,scores,_=load_boxes(P/f'artifacts/ct_fold2_all109_boxes/{cid}_boxes.pkl');boxmask=np.zeros(image.shape,bool)
 if matched is None:
  idx=89;bl,bh=box_to_native_bounds(boxes[idx]);grid=np.stack(np.meshgrid(*[np.arange(a,b) for a,b in zip(lo,hi)],indexing='ij'));boxmask=np.all((grid>=bl[:,None,None,None])&(grid<bh[:,None,None,None]),axis=0)
 limits=np.percentile(image,[10,99.5]);fig,axes=plt.subplots(2,3,figsize=(12,8))
 for axis in range(3):
  other=[j for j in range(3) if j!=axis];extent=[0,image.shape[other[0]]*sp[other[0]],0,image.shape[other[1]]*sp[other[1]]];idx=int(round(center[axis]-lo[axis]));ss=[slice(None)]*3;ss[axis]=slice(max(idx-1,0),min(idx+2,image.shape[axis]));ss=tuple(ss)
  for row in range(2):
   plane=image.max(axis) if row==0 else image[ss].max(axis);ax=axes[row,axis];ax.imshow(plane.T,origin='lower',cmap='gray',vmin=limits[0],vmax=limits[1],extent=extent)
   for mask,color in [(gm,'red'),(pm,'cyan'),(boxmask,'yellow')]:
    q=mask.any(axis) if row==0 else mask[ss].any(axis)
    if q.any() and not q.all():ax.contour(q.T,levels=[.5],colors=[color],linewidths=.9,origin='lower',extent=extent)
   ax.set_title(f'Native axis{axis}: '+('24mm MIP' if row==0 else '3 native slices'));ax.set_xlabel('mm');ax.set_ylabel('mm')
 title=f'{cid}: GT red; matched E16 cyan; rank90 box yellow if missed';fig.suptitle(title,fontsize=10);fig.tight_layout(rect=[0,0,1,.95]);file=R/f'{cid}.png';fig.savefig(file,dpi=130);plt.close(fig)
 fp=np.setdiff1d(matched['flat'],g['flat'],assume_unique=True) if matched is not None else np.array([],dtype=int);v=vessel.reshape(-1)[fp];rows.append({'case_id':cid,'image':str(file),'class':g['class'],'matched_prediction':matched is not None,'matched_component_nonGT_voxels':len(fp),'nonGT_predicted_vessel_overlap':float(np.mean(v>0)) if len(v) else None,'display_native_intensity_percentiles_10_99_5':limits.tolist(),'diagnostic_only':True});print(cid,flush=True)
(R/'RESULT.json').write_text(json.dumps({'rows':rows,'selection':'One saved candidate miss and one saved large volume overprediction; illustrative, not unbiased cohort estimate. No model or operating threshold change; no GT vessel input. Predicted vessel overlap is not clinical adjudication.'},indent=2)+'\n')
