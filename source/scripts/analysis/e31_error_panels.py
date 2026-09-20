"""Fixed post-evaluation source of error panels; no model or GT edits."""
import json
from pathlib import Path
import numpy as np,nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.astra6_e31.common import P,RUN
from scripts.astra6_e04.run_e04 import load_image,component_records_fast
from scripts.astra6_e01.e01_common import DATA,TA36_DIR,write_json

def main():
 dest=RUN/'error_panels';dest.mkdir(exist_ok=True);diag=json.loads((RUN/'physical/evaluation/lesion_diagnostics.json').read_text());rows=diag['paired_rows'];chosen=[]
 for cid in ['topaneu_center2_mr_016','topaneu_center2_mr_079','topaneu_center2_mr_080']:
  rr=[r for r in rows if r['case_id']==cid and r['before']['matched']]
  target=min(rr,key=lambda r:r['after'].get('binary_dice',0)-r['before']['binary_dice']);chosen.append(target)
 manifest=[]
 for target in chosen:
  cid=target['case_id'];image,aff,_=load_image(cid);gt=np.asanyarray(nib.load(str(DATA/f'location_masks/{cid}.nii.gz')).dataobj);cs=component_records_fast(gt);comp=next(c for c in cs if c['class_id']==target['class'] and len(c['coords'])==target['before']['voxels']);center=np.rint((comp['coords'].min(0)+comp['coords'].max(0))/2).astype(int);spacing=np.linalg.norm(aff[:3,:3],axis=0);radius=np.ceil(12/spacing).astype(int);lo=np.maximum(0,center-radius);hi=np.minimum(image.shape,center+radius+1);slices=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi));im=image[slices];truth=gt[slices]>0;vessel=np.asanyarray(nib.load(str(TA36_DIR/f'{cid}.nii.gz')).dataobj)[slices]>0;preds={}
  for name,folder in [('E17',P/'artifacts/astra6_e17_MR_detector_crop_segmentation_20260909'),('normalized',RUN/'normalized'),('physical',RUN/'physical')]:preds[name]=np.asanyarray(nib.load(str(folder/f'predictions/mr_center2_k05/{cid}.nii.gz')).dataobj)[slices]>0
  fig,axes=plt.subplots(3,4,figsize=(12,9),facecolor='white');index=center-lo
  for axis in range(3):
   left=[i for i in range(3) if i!=axis];idx=int(index[axis]);gray=np.take(im,idx,axis=axis).T;t=np.take(truth,idx,axis=axis).T;ext=[-.5*gray.shape[1]*spacing[left[0]],.5*gray.shape[1]*spacing[left[0]],-.5*gray.shape[0]*spacing[left[1]],.5*gray.shape[0]*spacing[left[1]]]
   for j,name in enumerate(['TA36','E17','normalized','physical']):
    ax=axes[axis,j];ax.imshow(gray,cmap='gray',vmin=0,vmax=1,origin='lower',extent=ext);mask=np.take(vessel if name=='TA36' else preds[name],idx,axis=axis).T
    if mask.any() and not mask.all():ax.contour(mask,levels=[.5],colors=['orange' if name=='TA36' else 'magenta'],linewidths=.8,origin='lower',extent=ext)
    if t.any() and not t.all():ax.contour(t,levels=[.5],colors=['lime'],linewidths=.8,origin='lower',extent=ext)
    ax.set_aspect('equal');ax.set_title(name if axis==0 else '',fontsize=11);ax.set_xlabel('native-plane mm');ax.set_ylabel('axis '+str(axis)+' slice' if j==0 else '')
  fig.suptitle(cid+' | GT green; predicted vessel orange; segmentation magenta\nNative oblique planes, physical aspect; target class '+str(target['class'])+', '+str(target['before']['voxels'])+' GT voxels',fontsize=11);fig.tight_layout(rect=[0,0,1,.94]);path=dest/f'{cid}.png';fig.savefig(path,dpi=150);plt.close(fig);manifest.append({'case_id':cid,'target':target,'center_native_voxel':center.tolist(),'spacing':spacing.tolist(),'panel':str(path.relative_to(P))});print(path,flush=True)
 write_json(dest/'MANIFEST.json',{'cases':manifest,'selection':'Largest physical-arm degradation in MR016/MR080 and lost lesion in MR079 after completed evaluation; diagnostic only','GT_never_used_as_model_input':True})
if __name__=='__main__':main()
