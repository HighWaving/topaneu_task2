"""Freeze E20 data and cache source-native decoding targets for epoch selection."""
import json,shutil
import numpy as np
from scripts.astra6_e01.e01_common import P,DATA,load_nifti,write_json,sha256_file
from scripts.astra6_e04.run_e04 import component_records_fast
RUN=P/'artifacts/astra6_e21_CT_native_checkpoint_selection_20260909'
BASE=P/'artifacts/astra6_e20_CT_detector_crop_segmentation_20260909'
def main():
 for d in ['model','evaluation','native_validation','predictions_ct']:(RUN/d).mkdir(parents=True,exist_ok=True)
 pre=json.loads((P/'artifacts/source_CT_native_segmentation_audit_20260909/RESULT.json').read_text());assert pre['criterion_mismatch']
 config={'experiment':'E21','hypothesis':'Native decoding reverses E20 source crop-Dice improvement; select training duration on exact production native decoder','data':'Immutable E20 source bank, no additional cases or errors backfilled','development_epochs':13,'final_epochs':'Epoch with maximum source-native mean Dice within completed13development epochs, earliest tie; complete full selected duration from scratch','source_gate':'Selected source-native Dice must exceed frozen E16 source-native0.8208194051141334 by at least0.005','official_gate':'All-six Pareto vs CT_E16,>=4matched and>=3correct locations; MR untouched','budget_hours':1,'checkpoint':'Each epoch optimizer and all RNG states; separate native validation history; no early stop','no_CT5_or_MR40_fit':True};write_json(RUN/'config.json',config)
 if not (RUN/'features').exists():(RUN/'features').symlink_to(BASE/'features',target_is_directory=True)
 shutil.copy2(BASE/'source_split.json',RUN/'source_split.json')
 if (RUN/'native_validation/READY.json').exists():return
 records=[json.loads(s) for s in (BASE/'features/train_records.jsonl').read_text().splitlines()];split=json.loads((BASE/'source_split.json').read_text());groups={}
 for j,row in enumerate(split['development_detector_rows']):groups.setdefault(records[row]['case_id'],[]).append((j,row,records[row]))
 for cid,items in sorted(groups.items()):
  gt,aff,shape=load_nifti(DATA/f'location_masks/{cid}.nii.gz');comps=component_records_fast(gt)
  for j,row,r in items:
   original=records[r['matched_GT_base_row']];comp=next(c for c in comps if c['class_id']==r['source_class_id'] and np.array_equal(c['coords'].min(0)-.5,original['low']) and np.array_equal(c['coords'].max(0)+.5,original['high']));low,high=np.array(r['low']),np.array(r['high']);lo=np.maximum(np.floor(low).astype(int),0);hi=np.minimum(np.ceil(high).astype(int),shape);native=np.stack(np.meshgrid(*[np.arange(a,b) for a,b in zip(lo,hi)],indexing='ij'));coords=(native-((low+high)/2)[:,None,None,None])/(2*np.maximum(high-low,1))[:,None,None,None]*32+15.5;points=comp['coords'];inside=points[np.all((points>=lo)&(points<hi),axis=1)];truth=np.zeros(tuple(hi-lo),bool);truth[tuple((inside-lo).T)]=True;center=(low+high)/2;radius=np.maximum((high-low)/2,.5);ellipse=np.sum(((native-center[:,None,None,None])/radius[:,None,None,None])**2,axis=0)<=1
   np.savez_compressed(RUN/f'native_validation/{j:02d}.npz',coords=coords,truth=truth,ellipse=ellipse,GT_voxels=len(points),row=row)
  print('E21_NATIVE_TARGET',cid,flush=True)
 files=list((RUN/'native_validation').glob('*.npz'));assert len(files)==28
 write_json(RUN/'native_validation/READY.json',{'n':28,'source_E20_split_sha256':sha256_file(BASE/'source_split.json'),'files_sha256':{f.name:sha256_file(f) for f in files},'source_only':True})
if __name__=='__main__':main()
