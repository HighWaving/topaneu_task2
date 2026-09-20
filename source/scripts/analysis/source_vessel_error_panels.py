"""Descriptive source images for low predicted-vessel/GT overlap, no relabeling."""
from pathlib import Path
import json
import numpy as np,nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.astra6_e01.e01_common import P,DATA,write_json
from scripts.astra6_e04.run_e04 import component_records_fast
OUT=P/'artifacts/review_current_model_20260910/source_vessel_panels'
def main():
 OUT.mkdir(exist_ok=True);rows=[]
 for cid in ['topaneu_center1_mr_800','topaneu_center5_mr_016','topaneu_center5_mr_478']:
  im=nib.load(str(DATA/f'images/{cid}_0000.nii.gz'));image=im.get_fdata(dtype=np.float32);gt=np.asanyarray(nib.load(str(DATA/f'location_masks/{cid}.nii.gz')).dataobj);ves=np.asanyarray(nib.load(str(P/f'artifacts/source_MR_TA36_distribution_audit_20260909/{cid}/predicted_vessel.nii.gz')).dataobj);c=max(component_records_fast(gt),key=lambda x:len(x['coords']));coords=c['coords'];center=np.rint(coords.mean(0)).astype(int);spacing=np.linalg.norm(im.affine[:3,:3],axis=0);field=max(24,float(np.max((coords.max(0)-coords.min(0))*spacing)*1.5));sub=image[::4,::4,::4];sub=sub[sub!=0];lo,hi=np.percentile(sub,[.5,99.5]);image=np.clip((image-lo)/max(hi-lo,1e-6),0,1);target=np.zeros(gt.shape,bool);target[tuple(coords.T)]=True;fig,axs=plt.subplots(1,3,figsize=(15,5))
  for axis,ax in enumerate(axs):
   others=[i for i in range(3) if i!=axis];slices=[slice(max(0,int(center[i]-field/2/spacing[i])),min(image.shape[i],int(center[i]+field/2/spacing[i])+1)) for i in range(3)];slices[axis]=int(center[axis]);key=tuple(slices);img=image[key].T;g=target[key].T;v=(ves[key]>0).T;extent=[]
   for i in others:extent.extend([(slices[i].start-center[i])*spacing[i],(slices[i].stop-center[i])*spacing[i]])
   ax.imshow(img,cmap='gray',origin='lower',vmin=0,vmax=1,extent=extent,aspect='equal')
   if g.any():ax.contour(g,levels=[.5],colors=['lime'],linewidths=1,origin='lower',extent=extent)
   if v.any():ax.contour(v,levels=[.5],colors=['orange'],linewidths=.7,origin='lower',extent=extent)
   ax.set_title(f'Native axis {axis}');ax.set_xlabel('native-plane mm')
  overlap=float((ves[tuple(coords.T)]>0).mean());diam=(6*len(coords)*abs(np.linalg.det(im.affine[:3,:3]))/np.pi)**(1/3);fig.suptitle(f'{cid} | class {c["class_id"]} | equivalent diameter {diam:.2f} mm\nGT green, TA36 orange; GT overlap {overlap:.3f}');fig.tight_layout();fig.savefig(OUT/(cid+'.png'),dpi=120);plt.close(fig);rows.append({'case_id':cid,'class':c['class_id'],'diameter_mm':diam,'GT_voxels':len(coords),'TA36_GT_overlap':overlap,'normalized_image_median_on_GT':float(np.median(image[tuple(coords.T)]))});print('SOURCE PANEL',cid,flush=True)
 write_json(OUT/'MANIFEST.json',{'cases':rows,'not_clinical_label_adjudication':True,'no_data_or_model_change':True})
if __name__=='__main__':main()
