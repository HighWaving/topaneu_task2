"""Mechanism diagnostic on S-held-out source cases; old-D vs OOF-D geometry.
The old detector fitted these source cases: this is a box-sensitivity comparison,
not an independent detector evaluation or a source model-selection rule.
"""
import json
import numpy as np,torch,nibabel as nib
from scripts.astra6_e31.common import *
from scripts.astra6_e31.train import Model
from scripts.astra6_e04.run_e04 import load_image
from scripts.astra6_e01.e01_common import DATA,load_boxes,select_candidates,write_json,sha256_file
F=P/'artifacts/astra6_e25_oof_shape_fp_filter_20260909'
def direct_score(prob,origin,step,r,truth):
 lo,fg=native_foreground(prob,origin,step,r['image_shape']);c=truth-lo;ok=np.all((c>=0)&(c<np.array(fg.shape)),axis=1);c=c[ok];inter=int(fg[tuple(c.T)].sum()) if len(c) else 0
 return {'Dice':2*inter/max(1,int(fg.sum())+len(truth)),'predicted_voxels':int(fg.sum()),'relative_volume_error':float(fg.sum()/len(truth)-1),'empty':not fg.any()}
def main():
 if (RUN/'BOX_STABILITY.json').exists():return
 torch.set_num_threads(2);split=json.loads((RUN/'source_split.json').read_text());records=[json.loads(s) for s in (RUN/'features/records.jsonl').read_text().splitlines()];rows=[];missing=[];models={}
 for arm in ['normalized','physical']:
  assert (RUN/arm/'model/LOCKED.json').exists();m=Model().cuda().eval();m.load_state_dict(torch.load(RUN/arm/'model/development_best.pt',map_location='cpu',weights_only=False)['state_dict']);models[arm]=m
 for cid in sorted({records[i]['case_id'] for i in split['validation_rows']}):
  image,aff,_=load_image(cid);vessel=np.asanyarray(nib.load(str(DATA/f'vessel_masks/{cid}.nii.gz')).dataobj);meta=json.loads((F/f'features/cases/{cid}.provenance.json').read_text())['old_source_detector'];assert sha256_file(Path(meta['boxes_path']))==meta['boxes_sha256'];bx,sc,_=load_boxes(Path(meta['boxes_path']));old=select_candidates(bx,sc)
  for i in [i for i in split['validation_rows'] if records[i]['case_id']==cid]:
   r=records[i];truth=np.load(RUN/f'native_validation/{i}.npz')['coords'];bounds=(truth.min(0)-.5,truth.max(0)+.5);matches=[]
   for index,score,lo,hi in old:
    coverage=float(np.all((truth>=lo)&(truth<hi),axis=1).mean());inter=np.maximum(0,np.minimum(hi,bounds[1])-np.maximum(lo,bounds[0])).prod();iou=inter/max(1e-12,(hi-lo).prod()+(bounds[1]-bounds[0]).prod()-inter)
    if coverage>=.9:matches.append((float(iou),index,lo,hi,coverage))
   ooflo,oofhi=np.array(r['low']),np.array(r['high']);coverage=float(np.all((truth>=ooflo)&(truth<oofhi),axis=1).mean())
   if not matches or coverage<.9:missing.append({'row':i,'case_id':cid,'OOF_coverage':coverage,'old_coverage_qualified_candidates':len(matches)});continue
   _,index,oldlo,oldhi,oldcoverage=max(matches,key=lambda x:(x[0],-x[1]));result={'row':i,'case_id':cid,'old_index':index,'OOF_index':r['candidate_index'],'old_coverage':oldcoverage,'OOF_coverage':coverage,'box_volume_ratio':float((oofhi-ooflo).prod()/(oldhi-oldlo).prod()),'arms':{}}
   for arm,net in models.items():
    values={}
    for name,lo,hi in [('old',oldlo,oldhi),('OOF',ooflo,oofhi)]:
     x,origin,step=input_crop(image,vessel,lo,hi,aff,arm)
     with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):prob=net(torch.from_numpy(x[None].astype(np.float32)).cuda()).float().sigmoid().cpu().numpy()[0,0]
     values[name]=direct_score(prob,origin,step,r,truth)
    result['arms'][arm]=values
   rows.append(result)
  print('E31 source box stability',cid,flush=True)
 summary={}
 for arm in models:
  eligible=[r for r in rows if r['arms'][arm]['old']['predicted_voxels'] and r['arms'][arm]['OOF']['predicted_voxels']];x=np.log([r['box_volume_ratio'] for r in eligible]);y=np.log([r['arms'][arm]['OOF']['predicted_voxels']/r['arms'][arm]['old']['predicted_voxels'] for r in eligible]);summary[arm]={'n_pairs':len(rows),'n_nonempty_pairs':len(eligible),'mean_OOF_minus_old_Dice':float(np.mean([r['arms'][arm]['OOF']['Dice']-r['arms'][arm]['old']['Dice'] for r in rows])) if rows else None,'mean_absolute_log_predicted_volume_change':float(np.abs(y).mean()) if len(y) else None,'log_box_vs_prediction_volume_Pearson':float(np.corrcoef(x,y)[0,1]) if len(x)>2 and np.std(x)>0 and np.std(y)>0 else None}
 write_json(RUN/'BOX_STABILITY.json',{'rows':rows,'excluded':missing,'summary':summary,'direct_new_model_output_before_learned_fallback':True,'source_S_selection_cases_only':True,'old_detector_in_sample':True,'OOF_detector_excludes_cases_from_gradients_but_not_legacy_planning':True,'not_a_selection_or_acceptance_metric':True})
if __name__=='__main__':main()
