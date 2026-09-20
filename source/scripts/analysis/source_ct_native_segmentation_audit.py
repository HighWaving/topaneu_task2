"""Does crop Dice improvement survive the actual native-coordinate decoder?"""
import json
import numpy as np,torch
from scipy.ndimage import map_coordinates
from scripts.astra6_e01.e01_common import P,DATA,load_nifti,write_json
from scripts.astra6_e04.run_e04 import Segmenter,PRIOR,largest,component_records_fast

def main():
 torch.set_num_threads(4);out=P/'artifacts/source_CT_native_segmentation_audit_20260909';out.mkdir(exist_ok=True)
 write_json(out/'config.json',{'hypothesis':'Fixed32crop validation may disagree with native-coordinate segmentation quality; compare same28source CT lesions under production decoder','source_only':True,'no_threshold_sweep':True,'next_gate':'If E20 crop improvement disappears in native decoding, replace crop-only source checkpoint criterion with native lesion Dice before any next full training'})
 audit=P/'artifacts/source_CT_detector_crop_dice_audit_20260909';pairs=json.loads((audit/'RESULT.json').read_text())['pairs'];data=np.load(audit/'paired_detector_crops.npz');x=np.stack([data['x'],np.broadcast_to(PRIOR,data['x'].shape)],axis=1).astype(np.float32);runs={'E16':P/'artifacts/astra6_e16_CT_expanded_segmentation_20260909','E20':P/'artifacts/astra6_e20_CT_detector_crop_segmentation_20260909'};probs={}
 for version,run in runs.items():
  net=Segmenter().eval();net.load_state_dict(torch.load(run/'model/development_last.pt',map_location='cpu',weights_only=False)['state_dict'])
  with torch.inference_mode():probs[version]=net(torch.from_numpy(x)).sigmoid().numpy()[:,0]
 records=[json.loads(s) for s in (runs['E16']/'features/train_records.jsonl').read_text().splitlines()];groups={};rows=[]
 for i,r in enumerate(pairs):groups.setdefault(r['case_id'],[]).append((i,r))
 for cid,items in groups.items():
  gt,aff,shape=load_nifti(DATA/f'location_masks/{cid}.nii.gz');comps=component_records_fast(gt)
  for i,r in items:
   g=records[r['GT_row']];comp=next(c for c in comps if c['class_id']==g['class_id'] and np.array_equal(c['coords'].min(0)-.5,g['low']) and np.array_equal(c['coords'].max(0)+.5,g['high']));low,high=np.array(r['low']),np.array(r['high']);lo=np.maximum(np.floor(low).astype(int),0);hi=np.minimum(np.ceil(high).astype(int),shape);native=np.stack(np.meshgrid(*[np.arange(a,b) for a,b in zip(lo,hi)],indexing='ij'));coords=(native-((low+high)/2)[:,None,None,None])/(2*np.maximum(high-low,1))[:,None,None,None]*32+15.5;truth=np.zeros(tuple(hi-lo),bool);points=comp['coords'];inside=points[np.all((points>=lo)&(points<hi),axis=1)];truth[tuple((inside-lo).T)]=True;row={'case_id':cid,'detector_index':r['detector_index'],'GT_voxels':len(points),'GT_class':g['class_id']}
   for version in runs:
    fg=largest(map_coordinates(probs[version][i],coords.reshape(3,-1),order=1,mode='constant',cval=0,prefilter=False).reshape(tuple(hi-lo))>=.5);empty=not fg.any()
    if empty:
     center=(low+high)/2;radius=np.maximum((high-low)/2,.5);fg=np.sum(((native-center[:,None,None,None])/radius[:,None,None,None])**2,axis=0)<=1
    row[version]={'native_Dice':float(2*(fg&truth).sum()/max(1,fg.sum()+len(points))),'predicted_voxels':int(fg.sum()),'empty_fallback':empty}
   rows.append(row)
  print('SOURCE_NATIVE_CT',cid,flush=True)
 means={v:float(np.mean([r[v]['native_Dice'] for r in rows])) for v in runs};write_json(out/'RESULT.json',{'n':len(rows),'native_Dice':means,'delta':means['E20']-means['E16'],'criterion_mismatch':means['E20']<means['E16'],'rows':rows,'no_CT5_or_MR40_access':True});print('SOURCE_NATIVE_RESULT',means,flush=True)
if __name__=='__main__':main()
