"""Source-only target round-trip test for fixed32 versus48 sampling."""
import json
import numpy as np
from scipy.ndimage import map_coordinates
from scripts.astra6_e01.e01_common import P,DATA,load_nifti,write_json
from scripts.astra6_e04.run_e04 import component_records_fast,largest

def main():
 out=P/'artifacts/source_segmentation_resolution_audit_20260909';out.mkdir(exist_ok=True)
 write_json(out/'config.json',{'hypothesis':'32cube target sampling may discard native lesion detail; test same2xcontext and same candidate support at32 and48','source_only':True,'modality_training_gate':'At least0.02mean native target-roundtrip Dice gain and at least20percent of paired source lesions gain>=0.02','scope':'31MR and28CT held-out source lesions, no MR40 or CT5','interpretation':'Target discretization diagnostic, not a mathematical upper bound on learned soft probabilities','no_threshold_sweep':True})
 result={}
 for modality,folder in [('MR','astra6_e17_MR_detector_crop_segmentation_20260909'),('CT','astra6_e20_CT_detector_crop_segmentation_20260909')]:
  run=P/'artifacts'/folder;records=[json.loads(s) for s in (run/'features/train_records.jsonl').read_text().splitlines()];split=json.loads((run/'source_split.json').read_text());groups={};rows=[]
  for i in split['development_detector_rows']:r=records[i];groups.setdefault(r['case_id'],[]).append(r)
  for cid,items in sorted(groups.items()):
   gt,aff,shape=load_nifti(DATA/f'location_masks/{cid}.nii.gz');comps=component_records_fast(gt)
   for r in items:
    original=records[r['matched_GT_base_row']];comp=next(c for c in comps if c['class_id']==r['source_class_id'] and np.array_equal(c['coords'].min(0)-.5,original['low']) and np.array_equal(c['coords'].max(0)+.5,original['high']));low,high=np.array(r['low']),np.array(r['high']);extent=np.maximum(high-low,1);center=(low+high)/2;lo=np.maximum(np.floor(low).astype(int),0);hi=np.minimum(np.ceil(high).astype(int),shape);native=np.stack(np.meshgrid(*[np.arange(a,b) for a,b in zip(lo,hi)],indexing='ij'));points=comp['coords'];inside=points[np.all((points>=lo)&(points<hi),axis=1)];truth=np.zeros(tuple(hi-lo),bool);truth[tuple((inside-lo).T)]=True;gl=points.min(0);gh=points.max(0)+1;small=np.zeros(tuple(gh-gl),np.uint8);small[tuple((points-gl).T)]=1;small=np.pad(small,1);gl-=1;row={'case_id':cid,'detector_index':r['detector_index'],'GT_class':r['source_class_id'],'GT_voxels':len(points),'native_roundtrip_Dice':{}}
    for n in [32,48]:
     grid=np.stack(np.meshgrid(*[(np.arange(n)-(n-1)/2)/n*2]*3,indexing='ij'));coords=center[:,None,None,None]+grid*extent[:,None,None,None];sampled=map_coordinates(small,(coords-gl[:,None,None,None]).reshape(3,-1),order=0,mode='constant',cval=0,prefilter=False).reshape(n,n,n);sampled=sampled*(np.abs(grid).max(0)<=.5);back=(native-center[:,None,None,None])/(2*extent[:,None,None,None])*n+(n-1)/2;fg=largest(map_coordinates(sampled.astype(np.float32),back.reshape(3,-1),order=1,mode='constant',cval=0,prefilter=False).reshape(truth.shape)>=.5);row['native_roundtrip_Dice'][str(n)]=float(2*(fg&truth).sum()/max(1,fg.sum()+len(points)))
    rows.append(row)
   print('SOURCE_RESOLUTION',modality,cid,flush=True)
  d=np.array([r['native_roundtrip_Dice']['48']-r['native_roundtrip_Dice']['32'] for r in rows]);result[modality]={'n':len(rows),'mean_Dice':{str(n):float(np.mean([r['native_roundtrip_Dice'][str(n)] for r in rows])) for n in [32,48]},'mean_gain':float(d.mean()),'n_gain_at_least_02':int((d>=.02).sum()),'training_gate_passed':bool(d.mean()>=.02 and np.mean(d>=.02)>=.2),'rows':rows};write_json(out/'RESULT.json',result);print('RESOLUTION_RESULT',modality,{k:v for k,v in result[modality].items() if k!='rows'},flush=True)
if __name__=='__main__':main()
