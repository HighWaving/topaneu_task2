"""Complete paired error attribution from saved diagnostics; no model fitting."""
from pathlib import Path
import json,time,math
import numpy as np
from scripts.astra6_e26.prepare import RUN,P
from scripts.astra6_e01.e01_common import write_json

def main():
 files=[RUN/'evaluation/paired_official_comparison.json',RUN/'evaluation/lesion_diagnostics.json']
 while not all(f.exists() for f in files):time.sleep(10)
 comparison=json.loads(files[0].read_text());after=json.loads(files[1].read_text());before=json.loads((P/'artifacts/astra6_e17_MR_detector_crop_segmentation_20260909/evaluation/lesion_diagnostics.json').read_text());rows=[];counts=[]
 for cid,b in before['cases'].items():
  a=after['cases'][cid];assert len(a['lesions'])==len(b['lesions']);counts.append({'case':cid,'FP_before':b['FP'],'FP_after':a['FP'],'matched_before':b['matched'],'matched_after':a['matched']})
  for i,(old,new) in enumerate(zip(b['lesions'],a['lesions'])):
   assert (old['class'],old['voxels'])==(new['class'],new['voxels']);r={'case':cid,'lesion_index':i,'class':old['class'],'size_bin':old['size_bin'],'matched_before':old['matched'],'matched_after':new['matched']}
   if old['matched'] and new['matched']:r.update(dice_before=old['binary_dice'],dice_after=new['binary_dice'],dice_delta=new['binary_dice']-old['binary_dice'],relative_volume_error_before=old['relative_volume_error'],relative_volume_error_after=new['relative_volume_error'],location_before=old['class_correct'],location_after=new['class_correct'])
   rows.append(r)
 pairs=[r for r in rows if 'dice_delta' in r];size={}
 for name in ['<=3','(3,5]','(5,7]','>7']:
  rr=[r for r in pairs if r['size_bin']==name];size[name]={'paired_n':len(rr),'mean_Dice_before':float(np.mean([r['dice_before'] for r in rr])) if rr else None,'mean_Dice_after':float(np.mean([r['dice_after'] for r in rr])) if rr else None,'mean_relative_volume_error_before':float(np.mean([r['relative_volume_error_before'] for r in rr])) if rr else None,'mean_relative_volume_error_after':float(np.mean([r['relative_volume_error_after'] for r in rr])) if rr else None}
 result={'decision':'not_adopted','reason':'Source-native gain did not transfer: official DICE/VOLSIM lower and HD95 higher, three detection metrics identical. Preserve E17/r2.','official':comparison,'before':before['E17'],'after':after['E26'],'paired_size':size,'largest_Dice_declines':sorted(pairs,key=lambda r:r['dice_delta'])[:10],'largest_Dice_gains':sorted(pairs,key=lambda r:r['dice_delta'],reverse=True)[:10],'case_count_changes':[r for r in counts if r['FP_before']!=r['FP_after'] or r['matched_before']!=r['matched_after']],'all_paired_lesions':rows,'no_training_from_comparison_errors':True};write_json(RUN/'evaluation/DECISION.json',result);print(json.dumps({k:result[k] for k in ['decision','reason','before','after','paired_size','case_count_changes']},indent=2),flush=True)
if __name__=='__main__':main()
