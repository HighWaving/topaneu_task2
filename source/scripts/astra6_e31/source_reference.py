import json,numpy as np
from scripts.astra6_e31.train import RUN,source_score
from scripts.astra6_e01.e01_common import write_json

def main():
 if (RUN/'SOURCE_REFERENCE.json').exists():return
 split=json.loads((RUN/'source_split.json').read_text());rows=[json.loads(s) for s in (RUN/'features/records.jsonl').read_text().splitlines()];geo=np.load(RUN/'normalized/features/geometry.npz');out=[]
 for i in split['validation_rows']:
  score,used,volume=source_score(np.zeros((48,48,48),np.float32),rows[i],geo['origin'][i],geo['step'][i],dict(np.load(RUN/f'native_validation/{i}.npz')));assert used;out.append({'row':i,'case_id':rows[i]['case_id'],'Dice':score,'predicted_voxels':volume})
 audit={}
 for arm in ['normalized','physical']:
  y=np.load(RUN/arm/'features/targets.npy',mmap_mode='r');g=np.load(RUN/arm/'features/geometry.npz');scores=[]
  for i in split['validation_rows']:
   score,used,volume=source_score(y[i].astype(np.float32),rows[i],g['origin'][i],g['step'][i],dict(np.load(RUN/f'native_validation/{i}.npz')));assert not used,('GT roundtrip lost target',arm,i);scores.append({'row':i,'case_id':rows[i]['case_id'],'native_GT_roundtrip_Dice':score,'reconstructed_voxels':volume,'full_GT_voxels':rows[i]['GT_voxels']})
  audit[arm]={'rows':scores,'mean_Dice':float(np.mean([r['native_GT_roundtrip_Dice'] for r in scores]))}
 write_json(RUN/'SOURCE_GEOMETRY_AUDIT.json',{'arms':audit,'scope':'Source-only GT roundtrip of representation; includes crop field coverage; not model Dice or a claimed learned upper bound'})
 write_json(RUN/'SOURCE_REFERENCE.json',{'case_mean_Dice':float(np.mean([np.mean([r['Dice'] for r in out if r['case_id']==c]) for c in sorted({r['case_id'] for r in out})])),'rows':out,'model':'E23 heldout S1 learned source fallback','GT_vessel_input':False,'training_data_differs_from_E31':True})
if __name__=='__main__':main()
