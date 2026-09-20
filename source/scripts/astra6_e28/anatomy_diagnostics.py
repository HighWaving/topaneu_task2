"""Source-only anatomy capability audit after C epoch selection; no refitting."""
import json
import numpy as np
import torch
from scripts.astra6_e28.common import RUN
from scripts.astra6_e28.representation import JointAnatomyLocation
from scripts.astra6_e01.e01_common import DATA,sha256_file,write_json

def metrics(matrix,names):
 actual=matrix.sum(1);predicted=matrix.sum(0);tp=np.diag(matrix)
 return {str(k):{'name':names[k],'reference_voxels':int(actual[k]),'predicted_voxels':int(predicted[k]),'true_positive_voxels':int(tp[k]),'Dice':float(2*tp[k]/(actual[k]+predicted[k])) if actual[k]+predicted[k] else None,'recall':float(tp[k]/actual[k]) if actual[k] else None} for k in range(1,37)}

def run(records,split,diagnostic_last=False):
 checkpoint=RUN/('model/development_last.pt' if diagnostic_last else 'model/development_best.pt');model=JointAnatomyLocation().cuda().eval();saved=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(saved['state_dict']);names={int(v):k for k,v in json.loads((DATA/'vessel_mapping.json').read_text())['labels'].items()};arrays={k:np.load(RUN/f'features/{k}.npy',mmap_mode='r') for k in ['local','context','silver']};dev=set(split['development_cases']);fit={records[i]['case_id'] for i in split['train_rows']};assert not fit&dev;groups={'classification_validation_crops':split['validation_rows'],'anatomy_only_validation_crops':[i for i,r in enumerate(records) if r['case_id'] in dev and r['kind']=='silver_anatomy_only']};result={}
 for pool,indices in groups.items():
  assert all(records[i]['case_id'] in dev for i in indices);bycase={}
  for start in range(0,len(indices),4):
   ix=indices[start:start+4];local=torch.from_numpy(np.array(arrays['local'][ix],np.float32))[:,None].cuda();context=torch.from_numpy(np.array(arrays['context'][ix],np.float32))[:,None].cuda()
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):out=model(local,context,torch.zeros_like(local),True)['vessel_logits']
   assert torch.isfinite(out).all();pred=out.argmax(1).to(torch.uint8).cpu().numpy();truth=np.asarray(arrays['silver'][ix]);assert truth.min()>=0 and truth.max()<=36
   for j,i in enumerate(ix):
    cid=records[i]['case_id'];mat=np.bincount(37*truth[j].astype(np.int64).ravel()+pred[j].ravel(),minlength=37*37).reshape(37,37);bycase.setdefault(cid,np.zeros((37,37),np.int64));bycase[cid]+=mat
  total=sum(bycase.values(),np.zeros((37,37),np.int64));percase={c:metrics(m,names) for c,m in bycase.items()};pooled=metrics(total,names)
  for k,row in pooled.items():
   ds=[v[k]['Dice'] for v in percase.values() if v[k]['Dice'] is not None];row['case_mean_Dice']=float(np.mean(ds)) if ds else None;row['cases_with_reference']=sum(v[k]['reference_voxels']>0 for v in percase.values())
  result[pool]={'n_crops':len(indices),'n_cases':len(bycase),'pooled_confusion_reference_rows_prediction_columns':total.tolist(),'per_class':pooled,'per_case':percase}
 output='SOURCE_ANATOMY_LAST_DIAGNOSTICS.json' if diagnostic_last else 'SOURCE_ANATOMY_DIAGNOSTICS.json'
 write_json(RUN/'evaluation'/output,{'development_checkpoint_sha256':sha256_file(checkpoint),'selected_C_epoch':None if diagnostic_last else saved['epoch'],'checkpoint_epoch':saved['epoch'],'checkpoint_role':'unselected_last_diagnostic_only' if diagnostic_last else 'source_selected','source_split_sha256':sha256_file(RUN/'source_split.json'),'groups':result,'image_only_auxiliary_head':True,'not_used_to_select_epoch_or_threshold':True,'interpretation':'Agreement with organizer-predicted vessel silver, not true-vessel accuracy or official52class Task2 metrics. Source C epoch-selection cases are reused for diagnosis, not independent acceptance. Random vessel-centered anatomy crops and D-centered positive crops reported separately; no MR40/CT5 images or labels used.'});print('E28 source anatomy diagnostics complete',output,flush=True)

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--last',action='store_true');args=parser.parse_args()
 # The training chain still calls the unchanged default selected-checkpoint path.
 # A last-checkpoint audit is a single post-training diagnosis, never selection.
 if args.last:assert (RUN/'model/LOCKED.json').exists(),'Finish formal training before this additional diagnostic'
 torch.set_num_threads(1)
 records=[json.loads(s) for s in (RUN/'features/records.jsonl').read_text().splitlines()]
 run(records,json.loads((RUN/'source_split.json').read_text()),diagnostic_last=args.last)
