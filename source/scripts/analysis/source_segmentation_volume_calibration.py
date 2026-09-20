"""One analytic volume calibration, evaluated by source-case leave-one-out."""
from pathlib import Path
import json,math
import numpy as np,torch
from scipy.ndimage import map_coordinates
from scripts.astra6_e04.run_e04 import Segmenter,SegData,largest
from scripts.astra6_e26.prepare import RUN as CACHE,BASE,P
from scripts.astra6_e01.e01_common import write_json,sha256_file
R=P/'artifacts/source_MR_segmentation_volume_calibration_20260909'
def decode(prob,t,threshold):
 fg=largest(prob>=threshold);fallback=not fg.any()
 if fallback:fg=t['ellipse']
 inter=int((fg&t['truth']).sum());g=int(t['GT_voxels']);n=int(fg.sum());return {'dice':2*inter/max(n+g,1),'relative_volume_error':(n-g)/g,'overlap':inter>0,'fallback':fallback,'predicted_voxels':n}
def main():
 R.mkdir(exist_ok=True);config={'hypothesis':'Source-independent assessment of volume bias in frozen E17 S, motivated by E26 oversegmentation analysis. MR40 does not set or select any threshold.','model':'E17 development13 for source assessment; E17 final13 retained for possible deployment','calibration':'For each eligible source lesion use midpoint between native probability order statistics that enclose GT volume, then median within case and across cases. No threshold sweep. Eligible: >=90% GT inside detectorbox, 0<GTvolume<boxvolume, finite probabilities.','validation':'Leave one source case out of analytic threshold calculation, apply held-out-case threshold through exact production largestCC/emptyellipse decoder; no image/GT fitting of S','source_gate':'At least20 eligiblecases; source baseline median relative volume error >=.20; LOCO native meanDice gain>=.005, median absolute relative volume error improves>=.05, no lost baseline-positive GT overlap','inference_scope_if_pass':'One frozen all-source scalar only; C/F/D/S weights unchanged. NoMR40 threshold selection.','upstream_limitation':'Original source detector proposals were in-sample; not all-stage leave-one-case-out.'};write_json(R/'PLAN.json',config);torch.set_num_threads(1);split=json.loads((BASE/'source_split.json').read_text());ix=split['development_detector_rows'];records=[json.loads(s) for s in (BASE/'features/train_records.jsonl').read_text().splitlines()];assert len(ix)==31 and all('center2' not in records[i]['case_id'] for i in ix);net=Segmenter().cuda();net.load_state_dict(torch.load(BASE/'model/development_last.pt',map_location='cpu',weights_only=False)['state_dict']);net.eval();data=SegData(CACHE,ix);probs=[]
 with torch.inference_mode():
  for start in range(0,len(ix),8):probs.extend(net(torch.stack([data[i][0] for i in range(start,min(start+8,len(ix)))]).cuda()).sigmoid().cpu().numpy()[:,0])
 targets=[dict(np.load(CACHE/f'native_validation/{j:02d}.npz')) for j in range(31)];native=[map_coordinates(p,t['coords'].reshape(3,-1),order=1,mode='constant',cval=0,prefilter=False).reshape(t['truth'].shape) for p,t in zip(probs,targets)];case_thresholds={};rows=[]
 for j,(prob,t,i) in enumerate(zip(native,targets,ix)):
  cid=records[i]['case_id'];g=int(t['GT_voxels']);flat=prob.ravel();eligible=bool(t['truth'].sum()/g>=.9 and 0<g<len(flat));q=None
  if eligible:
   k=len(flat)-g;v=np.partition(flat,[k-1,k]);q=float((float(v[k-1])+float(v[k]))/2);case_thresholds.setdefault(cid,[]).append(q)
  rows.append({'case_id':cid,'source_row':i,'eligible':eligible,'volume_matching_threshold':q,'baseline':decode(prob,t,.5)})
 case_thresholds={c:float(np.median(v)) for c,v in case_thresholds.items()};assert len(case_thresholds)>1
 for r,prob,t in zip(rows,native,targets):
  threshold=float(np.median([v for c,v in case_thresholds.items() if c!=r['case_id']]));r['LOCO_threshold']=threshold;r['LOCO_result']=decode(prob,t,threshold)
 b=np.mean([r['baseline']['dice'] for r in rows]);a=np.mean([r['LOCO_result']['dice'] for r in rows]);bias=float(np.median([r['baseline']['relative_volume_error'] for r in rows]));bv=float(np.median([abs(r['baseline']['relative_volume_error']) for r in rows]));av=float(np.median([abs(r['LOCO_result']['relative_volume_error']) for r in rows]));lost=sum(r['baseline']['overlap'] and not r['LOCO_result']['overlap'] for r in rows);passed=bool(len(case_thresholds)>=20 and bias>=.2 and a-b>=.005 and bv-av>=.05 and lost==0);threshold=float(np.median(list(case_thresholds.values())));result={'source_gate_passed':passed,'n_lesions':31,'n_cases':len({r['case_id'] for r in rows}),'eligible_cases':len(case_thresholds),'baseline_native_Dice':float(b),'LOCO_native_Dice':float(a),'Dice_gain':float(a-b),'baseline_median_relative_volume_error':bias,'baseline_median_absolute_relative_volume_error':bv,'LOCO_median_absolute_relative_volume_error':av,'lost_baseline_overlap':int(lost),'final_analytic_threshold':threshold,'rows':rows,'case_thresholds':case_thresholds,'source_model_sha256':sha256_file(BASE/'model/development_last.pt'),'target_manifest_sha256':sha256_file(CACHE/'native_validation/READY.json'),'no_MR40_CT5_opened':True,'no_new_network_training':True};assert abs(b-.7504814451169158)<1e-8;write_json(R/'RESULT.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['rows','case_thresholds']},indent=2),flush=True)
if __name__=='__main__':main()
