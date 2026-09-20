"""Class-independent one-to-one lesion diagnosis; never changes fitted models."""
import argparse,json,math
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import label,find_objects,binary_erosion
from scipy.spatial import cKDTree
from scripts.astra6_e01.e01_common import DATA,load_nifti,write_json,sha256_tree,affine_world
from scripts.astra6_e03.run_e03 import BASE,read,METRICS
from scripts.local_scoring_arena import aggregate

def components(mask):
 out=[]
 for cl in np.unique(mask):
  if not cl:continue
  xyz=np.argwhere(mask==cl);lo=xyz.min(0);hi=xyz.max(0)+1;crop=mask[tuple(slice(a,b) for a,b in zip(lo,hi))]==cl;cc,n=label(crop,np.ones((3,3,3)))
  for i,sl in enumerate(find_objects(cc),1):
   if sl is None:continue
   pts=np.argwhere(cc[sl]==i)+lo+np.array([s.start for s in sl]);out.append({'class':int(cl),'coords':pts,'flat':np.ravel_multi_index(pts.T,mask.shape)})
 return out

def surface(c,aff):
 pts=c['coords'];lo=pts.min(0)-1;hi=pts.max(0)+2;small=np.zeros(tuple(hi-lo),bool);small[tuple((pts-lo).T)]=1;s=small & ~binary_erosion(small);return affine_world(aff,np.argwhere(s)+lo)

def diagnose(gt,pred,aff):
 g=components(gt);p=components(pred);inter=np.zeros((len(g),len(p)))
 for i,x in enumerate(g):
  for j,y in enumerate(p):inter[i,j]=len(np.intersect1d(x['flat'],y['flat'],assume_unique=True))
 dice=2*inter/np.maximum(1,np.array([len(x['flat']) for x in g])[:,None]+np.array([len(x['flat']) for x in p])[None,:]) if g and p else inter
 matches={i:j for i,j in zip(*linear_sum_assignment(-dice)) if inter[i,j]>0} if dice.size else {}
 spacing_volume=abs(np.linalg.det(aff[:3,:3]));ledger=[]
 for i,x in enumerate(g):
  diameter=(6*len(x['flat'])*spacing_volume/math.pi)**(1/3);bin_name='<=3' if diameter<=3 else '(3,5]' if diameter<=5 else '(5,7]' if diameter<=7 else '>7'
  r={'class':x['class'],'voxels':len(x['flat']),'volume_mm3':len(x['flat'])*spacing_volume,'equivalent_sphere_diameter_mm':diameter,'size_bin':bin_name,'matched':i in matches}
  if i in matches:
   j=matches[i];y=p[j];a=surface(x,aff);b=surface(y,aff);hd=max(np.percentile(cKDTree(a).query(b)[0],95),np.percentile(cKDTree(b).query(a)[0],95));r.update(predicted_class=y['class'],class_correct=x['class']==y['class'],binary_dice=float(dice[i,j]),hd95_mm=float(hd),relative_volume_error=(len(y['flat'])-len(x['flat']))/len(x['flat']))
  ledger.append(r)
 return {'gt':len(g),'predicted':len(p),'matched':len(matches),'FP':len(p)-len(matches),'FN':len(g)-len(matches),'lesions':ledger}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();r=a.run
 lock=read(r/'PREDICTIONS_HASHED_BEFORE_EVAL_GT.json');assert sha256_tree(r/'predictions')==lock['prediction_tree_sha256'];ids=read(BASE/'eval_case_ids.json');results={};before=read(BASE/'evaluation/after_per_case.json');after=read(r/'evaluation/after_per_case.json');assert [x['case_id'] for x in after]==ids
 for cid in ids:
  gt,aff,_=load_nifti(DATA/f'location_masks/{cid}.nii.gz');out={}
  for name,path in [('E02',BASE),('E04',r)]:
   pred,pa,_=load_nifti(path/f'predictions/mr_center2_k05/{cid}.nii.gz');assert np.allclose(aff,pa,atol=1e-4);out[name]=diagnose(gt,pred,aff)
  results[cid]=out;print(cid,flush=True)
  write_json(r/'evaluation/lesion_diagnostics_partial.json',results)
 summaries={}
 for version in ['E02','E04']:
  rows=[results[c][version] for c in ids];lesions=[{**l,'case_id':c} for c in ids for l in results[c][version]['lesions']];m=[l for l in lesions if l['matched']];summary={'GT_components':len(lesions),'matched':len(m),'recall':len(m)/len(lesions),'FP_per_case':sum(x['FP'] for x in rows)/len(ids),'location_correct':sum(x['class_correct'] for x in m),'location_denominator':len(m),'matched_binary_Dice_mean':float(np.mean([x['binary_dice'] for x in m])),'matched_HD95_mm_mean':float(np.mean([x['hd95_mm'] for x in m])),'size':{},'confusion':{}}
  for b in ['<=3','(3,5]','(5,7]','>7']:
   ls=[x for x in lesions if x['size_bin']==b];ms=[x for x in ls if x['matched']];summary['size'][b]={'matched':len(ms),'total':len(ls),'mean_matched_dice':float(np.mean([x['binary_dice'] for x in ms])) if ms else None}
  for x in m:
   k=f"{x['class']}->{x['predicted_class']}";summary['confusion'][k]=summary['confusion'].get(k,0)+1
  summaries[version]=summary
 rng=np.random.default_rng(20260909);samples={k:[] for k in METRICS}
 for _ in range(2000):
  ix=rng.integers(0,len(ids),len(ids));b=aggregate([before[i]['raw'] for i in ix])['overall'];c=aggregate([after[i]['raw'] for i in ix])['overall']
  for k in METRICS:samples[k].append(c[k]-b[k])
 boot={k:{'low':float(np.percentile(v,2.5)),'high':float(np.percentile(v,97.5))} for k,v in samples.items()}
 write_json(r/'evaluation/extended_diagnostics.json',{'matching':'maximum summed binary Dice Hungarian one-to-one, accept only nonzero intersection, per-location 26-connected components','diameter':'volume-equivalent sphere diameter; not maximum Feret diameter','bootstrap_unit':'case (patient linkage unavailable)','source_split':'no center2 refit','versions':summaries,'paired_bootstrap_2000_all_six':boot,'cases':results})
if __name__=='__main__':main()
