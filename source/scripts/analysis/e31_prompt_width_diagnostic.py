"""Causal source diagnostic: change only Gaussian prompt width, freeze image/grid.
No MR40, retraining, output threshold choice or adoption based on this probe.
"""
import json,numpy as np,torch
from scripts.astra6_e31.common import RUN
from scripts.astra6_e31.train import Model
from scripts.astra6_e31.box_stability import direct_score
from scripts.astra6_e01.e01_common import write_json,sha256_file

def main():
 dest=RUN/'PROMPT_WIDTH_DIAGNOSTIC.json'
 if dest.exists():return
 assert all((RUN/a/'model/LOCKED.json').exists() for a in ['normalized','physical'])
 plan={'hypothesis':'Even physical sampling still supplies detector width via the Gaussian prompt; this may preserve a shortcut from box size to mask volume.','intervention':'Multiply Gaussian sigma by 0.9 or 1.1, leaving image, vessel, crop field, center, weights and threshold unchanged.','cases':'All 26 frozen source selection candidates, source development weights only','no_MR40_read_or_model_change':True,'interpretation':'Source sensitivity diagnostic; not an additional validation or replacement selection rule.'};write_json(RUN/'PROMPT_WIDTH_DIAGNOSTIC_PLAN.json',plan)
 torch.set_num_threads(2);arm=RUN/'physical';model=Model().cuda().eval();model.load_state_dict(torch.load(arm/'model/development_best.pt',map_location='cpu',weights_only=False)['state_dict']);split=json.loads((RUN/'source_split.json').read_text());records=[json.loads(s) for s in (RUN/'features/records.jsonl').read_text().splitlines()];x=np.load(arm/'features/images.npy',mmap_mode='r');g=np.load(arm/'features/geometry.npz');selection={r['row']:r for r in json.loads((arm/'model/SOURCE_SELECTION.json').read_text())['rows']};rows=[]
 for i in split['validation_rows']:
  r=records[i];truth=np.load(RUN/f'native_validation/{i}.npz')['coords'];values={}
  for scale in [1.,.9,1.1]:
   inp=x[i:i+1].astype(np.float32);inp[:,2]=np.power(inp[:,2],1/(scale*scale))
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):pr=model(torch.from_numpy(inp).cuda()).float().sigmoid().cpu().numpy()[0,0]
   values[str(scale)]=direct_score(pr,g['origin'][i],g['step'][i],r,truth)
  assert not values['1.0']['empty'] and values['1.0']['predicted_voxels']==selection[i]['predicted_voxels'],'Fixed input source prediction did not reproduce'
  rows.append({'row':i,'case_id':r['case_id'],'variants':values})
 summary={}
 for scale in [.9,1.1]:
  pairs=[(r['variants']['1.0'],r['variants'][str(scale)]) for r in rows];summary[str(scale)]={'mean_relative_volume_change':float(np.mean([b['predicted_voxels']/a['predicted_voxels']-1 for a,b in pairs])),'median_relative_volume_change':float(np.median([b['predicted_voxels']/a['predicted_voxels']-1 for a,b in pairs])),'mean_Dice_change':float(np.mean([b['Dice']-a['Dice'] for a,b in pairs])),'new_empty':sum(b['empty'] for a,b in pairs)}
 write_json(dest,{'plan':plan,'summary':summary,'rows':rows,'source_checkpoint_sha256':sha256_file(arm/'model/development_best.pt'),'baseline_source_volumes_exact':True});print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
