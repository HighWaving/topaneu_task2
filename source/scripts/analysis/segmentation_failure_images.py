"""Saved E26 failure morphology, read-only, no new inference or threshold search."""
from pathlib import Path
import json,os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np,nibabel as nib
from scipy.ndimage import binary_dilation
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.analysis.paired_segmentation_diagnostics import components
P=Path(__file__).resolve().parents[2];D=P.parent/'data_topaneu26';B=P/'artifacts/astra6_e17_MR_detector_crop_segmentation_20260909';A=P/'artifacts/astra6_e26_MR_segmentation_regularization_20260909';R=P/'artifacts/research_audit_20260909/segmentation_images';R.mkdir(exist_ok=True)
errors=json.loads((A/'evaluation/DECISION.json').read_text())['largest_Dice_declines'];selected=[]
for row in errors:
 if row['case'] not in {r['case'] for r in selected}:selected.append(row)
 if len(selected)==3:break

def assignment(g,p):
 inter=np.array([[len(np.intersect1d(x['flat'],y['flat'],assume_unique=True)) for y in p] for x in g]);dice=2*inter/(np.array([len(x['flat']) for x in g])[:,None]+np.array([len(x['flat']) for x in p])[None,:]);return {int(i):int(j) for i,j in zip(*linear_sum_assignment(-dice)) if inter[i,j]>0}
results=[]
for row in selected:
 cid=row['case'];im=nib.load(str(D/f'images/{cid}_0000.nii.gz'));a=im.get_fdata(dtype=np.float32);aff=im.affine;sp=np.linalg.norm(aff[:3,:3],axis=0);gt=np.asanyarray(nib.load(str(D/f'location_masks/{cid}.nii.gz')).dataobj);before=np.asanyarray(nib.load(str(B/f'predictions/mr_center2_k05/{cid}.nii.gz')).dataobj);after=np.asanyarray(nib.load(str(A/f'predictions/mr_center2_k05/{cid}.nii.gz')).dataobj);v=np.asanyarray(nib.load(str(P/f'artifacts/ta36_mr_center2_output/{cid}.nii.gz')).dataobj);gc,bc,ac=components(gt),components(before),components(after);bi,ai=assignment(gc,bc),assignment(gc,ac);i=row['lesion_index'];g,b,c=gc[i],bc[bi[i]],ac[ai[i]];assert g['class']==row['class'];center=g['coords'].mean(0);lo=np.maximum(np.floor(center-12/sp).astype(int),0);hi=np.minimum(np.ceil(center+12/sp).astype(int)+1,a.shape);sl=tuple(slice(x,y) for x,y in zip(lo,hi));crop=a[sl];masks=[]
 for comp in [g,b,c]:
  mask=np.zeros(tuple(hi-lo),bool);pts=comp['coords'];pts=pts[np.all((pts>=lo)&(pts<hi),axis=1)];mask[tuple((pts-lo).T)]=True;masks.append(mask)
 added=np.setdiff1d(c['flat'],b['flat'],assume_unique=True);added_wrong=np.setdiff1d(added,g['flat'],assume_unique=True);removed_GT=np.intersect1d(np.setdiff1d(b['flat'],c['flat'],assume_unique=True),g['flat'],assume_unique=True);vl=v.reshape(-1)[added_wrong];hist={str(int(k)):int(n) for k,n in zip(*np.unique(vl,return_counts=True))};frac=float(np.mean(vl>0)) if len(vl) else None
 fig,axes=plt.subplots(2,3,figsize=(12,8));nonzero=crop[crop>0];vmin,vmax=np.percentile(nonzero,[30,99.9])
 for axis in range(3):
  others=[x for x in range(3) if x!=axis];extent=[0,crop.shape[others[0]]*sp[others[0]],0,crop.shape[others[1]]*sp[others[1]]];idx=int(round(center[axis]-lo[axis]));slices=[slice(None)]*3;slices[axis]=slice(max(0,idx-1),min(crop.shape[axis],idx+2));slices=tuple(slices)
  for j in [0,1]:
   plane=crop.max(axis=axis) if j==0 else crop[slices].max(axis=axis);axes[j,axis].imshow(plane.T,origin='lower',cmap='gray',vmin=vmin,vmax=vmax,extent=extent)
   for mask,color in zip(masks,['red','cyan','yellow']):
    z=mask.any(axis=axis) if j==0 else mask[slices].any(axis=axis)
    if z.any() and not z.all():axes[j,axis].contour(z.T,levels=[.5],colors=[color],linewidths=.8,origin='lower',extent=extent)
   axes[j,axis].set_title(f'Native axis{axis}; '+('24mm MIP' if j==0 else '3 native slices'));axes[j,axis].set_xlabel('mm');axes[j,axis].set_ylabel('mm')
 fig.suptitle(f'{cid} class{g["class"]}: GT red / E17 cyan / E26 yellow\nSaved predictions only; added nonGT voxels{len(added_wrong)}, TA36 overlap{frac:.2f}' if frac is not None else cid,fontsize=10);fig.tight_layout(rect=[0,0,1,.94]);path=R/f'{cid}.png';fig.savefig(path,dpi=130);plt.close(fig);results.append({**row,'added_nonGT_voxels':len(added_wrong),'removed_GT_voxels':len(removed_GT),'added_nonGT_predicted_vessel_fraction':frac,'added_nonGT_vessel_histogram':hist,'image':str(path),'interpretation':'Predicted-vessel alignment is a morphology proxy; not proof of clinical parent-vessel leakage. Review raw image/neck uncertainty.'});print(cid,flush=True)
(R/'RESULT.json').write_text(json.dumps({'rows':results,'no_fit_or_new_inference':True,'no_GT_vessel':True,'selection':'First3distinct cases in saved largestDice-decline list; illustrative worst errors, not unbiased cohort estimates'},indent=2))
