"""Full-budget source-only multi-task C training with exact-state resume."""
import json,os,time,random,math
from pathlib import Path
from collections import Counter
import numpy as np,torch,joblib
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from sklearn.base import clone
from scripts.astra6_e28.common import P,RUN,BASE,SEED
from scripts.astra6_e28.representation import JointAnatomyLocation,training_loss,refine_prediction
from scripts.astra6_e01.e01_common import DATA,write_json,sha256_file,lr_pair_map

def save(path,state):
 tmp=path.with_suffix('.tmp');torch.save(state,tmp);os.replace(tmp,path)
def mappings():
 maps=[]
 for name in ['location_mapping.json','vessel_mapping.json']:
  names={int(v):k for k,v in json.loads((DATA/name).read_text())['labels'].items()};pairs=lr_pair_map({k:v for k,v in names.items() if k});lut=np.arange(max(names)+1);lut[list(pairs)]=list(pairs.values());assert np.array_equal(lut[lut],np.arange(len(lut)));maps.append(torch.from_numpy(lut).cuda())
 return maps
class Crops(Dataset):
 def __init__(self,indices,records):
  self.indices=indices;self.records=records;self.arrays={k:np.load(RUN/f'features/{k}.npy',mmap_mode='r') for k in ['local','context','shape','silver']}
 def __len__(self):return len(self.indices)
 def __getitem__(self,j):
  i=self.indices[j];return {k:torch.from_numpy(np.array(a[i],copy=True)) for k,a in self.arrays.items()},self.records[i]['class'],i

def fit(stage,indices,records,dev=None,epochs=160):
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
 model=JointAnatomyLocation().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=160,eta_min=1e-5);scaler=torch.amp.GradScaler('cuda');locmap,vmap=mappings();labels=np.array([records[i]['class'] for i in indices]);positive=labels>0;assert positive.any();component_counts=Counter((records[i]['case_id'],records[i]['component_index']) for i in indices if records[i]['class']>0);component_class={(records[i]['case_id'],records[i]['component_index']):records[i]['class'] for i in indices if records[i]['class']>0};class_counts=Counter(component_class.values());mirror=locmap.cpu().numpy();weights=np.array([1/component_counts[records[i]['case_id'],records[i]['component_index']]/math.sqrt(class_counts[records[i]['class']]+class_counts[int(mirror[records[i]['class']])]) if records[i]['class']>0 else 1. for i in indices]);weights[positive]*=.5/weights[positive].sum();weights[~positive]*=.5/max(weights[~positive].sum(),1);gen=torch.Generator().manual_seed(SEED);sampler=WeightedRandomSampler(torch.from_numpy(weights),len(indices),replacement=True,generator=gen);loader=DataLoader(Crops(indices,records),batch_size=4,sampler=sampler,num_workers=0,pin_memory=True);validation=DataLoader(Crops(dev,records),batch_size=4,num_workers=0) if dev is not None else None
 history=[];best=float('inf');bestep=0;start=1;resume=RUN/f'model/{stage}_resume.pt'
 if resume.exists():
  state=torch.load(resume,map_location='cpu',weights_only=False);model.load_state_dict(state['state_dict']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler']);history=state['history'];best=state['best'];bestep=state['bestep'];start=state['epoch']+1;torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);gen.set_state(state['sampler_rng'])
 for epoch in range(start,epochs+1):
  if validation is not None and epoch>60 and epoch-1-bestep>=30:break
  begin=time.monotonic();model.train();losses=[];component_losses=[]
  for batch,y,_ in loader:
   batch={k:v.cuda(non_blocking=True) for k,v in batch.items()};y=y.cuda();local=batch['local'][:,None].float();context=batch['context'][:,None].float();shape=batch['shape'][:,None].float();silver=batch['silver'].long()
   if torch.rand((),device='cuda')<.5:
    local=local.flip(2);context=context.flip(2);shape=shape.flip(2);silver=vmap[silver.flip(1)];y=torch.where(y>0,locmap[y.clamp_min(0)],y)
   gamma=torch.empty((len(y),1,1,1,1),device='cuda').uniform_(.8,1.25);local=(local.clamp(0,1).pow(gamma)+.01*torch.randn_like(local)).clamp(0,1);context=(context.clamp(0,1).pow(gamma)+.01*torch.randn_like(context)).clamp(0,1)
   optimizer.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16):out=model(local,context,shape);loss,loss_terms=training_loss(out,y,silver)
   assert torch.isfinite(loss),'Nonfinite E28 training loss';scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(optimizer);scaler.update();values=torch.stack([loss.detach(),loss_terms['classification'],loss_terms['silver_anatomy']]).float().cpu().tolist();losses.append(values[0]);component_losses.append(values[1:])
  scheduler.step();record={'epoch':epoch,'loss':float(np.mean(losses)),'classification_loss':float(np.mean(component_losses,axis=0)[0]),'silver_anatomy_loss':float(np.mean(component_losses,axis=0)[1]),'seconds':time.monotonic()-begin,'updates':len(loader),'GPU_peak_bytes':torch.cuda.max_memory_allocated()}
  if validation is not None:
   model.eval();predictions=[]
   with torch.inference_mode():
    for batch,y,ix in validation:
     with torch.autocast('cuda',dtype=torch.float16):logits=model(batch['local'][:,None].cuda().float(),batch['context'][:,None].cuda().float(),batch['shape'][:,None].cuda().float(),False)['location_logits']
     assert torch.isfinite(logits).all(),'Nonfinite source validation logits'
     losses_val=torch.nn.functional.cross_entropy(logits.float(),y.cuda()-1,reduction='none').cpu().numpy();prob=logits.float().softmax(1).cpu().numpy()
     for i,loss_val,pp in zip(ix.tolist(),losses_val,prob):
      before=records[i]['source_baseline_class'];family=range(22,36) if 22<=before<=35 else range(45,53) if 45<=before<=52 else [];allowed=[c for c in family if c%2==before%2];target=records[i]['class'];eligible=target in allowed;conditional_loss=float(-np.log(max(float(pp[target-1])/max(float(pp[np.array(allowed)-1].sum()),1e-12),1e-12))) if eligible else None
      predictions.append({'row':i,'case_id':records[i]['case_id'],'class':target,'loss':float(loss_val),'selection_eligible':eligible,'conditional_fine_CE':conditional_loss,'probabilities':pp.tolist(),'prediction':int(pp.argmax()+1)})
   eligible=[r for r in predictions if r['selection_eligible']];assert eligible,'No source validation components for deployed fine-refinement task';cases=sorted({r['case_id'] for r in eligible});score=float(np.mean([np.mean([r['conditional_fine_CE'] for r in eligible if r['case_id']==c]) for c in cases]));record.update(source_case_CE=score,selection_metric='case-averaged conditional fine-class CE for baseline family/side eligible components',selection_components=len(eligible),correct=sum(r['class']==r['prediction'] for r in predictions),n_validation=len(predictions))
   if score<best:
    best=score;bestep=epoch;save(RUN/'model/development_best.pt',{'state_dict':model.state_dict(),'epoch':epoch,'source_case_CE':score});write_json(RUN/'model/development_predictions.json',predictions)
  history.append(record);write_json(RUN/f'model/{stage}_history.json',history);save(resume,{'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'best':best,'bestep':bestep,'history':history,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'sampler_rng':gen.get_state()});print(stage,json.dumps(record),flush=True)
 save(RUN/f'model/{stage}_last.pt',{'state_dict':model.state_dict(),'epoch':history[-1]['epoch'],'architecture':'E28_joint_anatomy_location'});return bestep if dev is not None else history[-1]['epoch']

def main():
 assert (RUN/'features/READY.json').exists()
 ready=json.loads((RUN/'features/READY.json').read_text());assert ready['config_sha256']==sha256_file(RUN/'config.json') and ready['source_split_sha256']==sha256_file(RUN/'source_split.json') and ready['records_sha256']==sha256_file(RUN/'features/records.jsonl')
 for key,digest in ready['arrays_sha256'].items():assert sha256_file(RUN/f'features/{key}.npy')==digest
 if (RUN/'model/LOCKED.json').exists():return
 records=[json.loads(s) for s in (RUN/'features/records.jsonl').read_text().splitlines()];split=json.loads((RUN/'source_split.json').read_text());assert not {records[i]['case_id'] for i in split['train_rows']}&{records[i]['case_id'] for i in split['validation_rows']}
 # One fixed-architecture source comparator; not the full-source baseline that
 # has seen these validation cases. No tree parameter search.
 preflight=RUN/'model/GPU_PREFLIGHT.json'
 if not preflight.exists():
  torch.set_num_threads(2);free,total=torch.cuda.mem_get_info();assert free>=6*2**30,'Need released assigned GPU before full E28 training';torch.cuda.reset_peak_memory_stats();batch,y,_=next(iter(DataLoader(Crops(split['train_rows'],records),batch_size=4)));model=JointAnatomyLocation().cuda()
  with torch.autocast('cuda',dtype=torch.float16):out=model(batch['local'][:,None].cuda().float(),batch['context'][:,None].cuda().float(),batch['shape'][:,None].cuda().float());loss,_=training_loss(out,y.cuda(),batch['silver'].cuda().long())
  assert torch.isfinite(loss);loss.backward();peak=torch.cuda.max_memory_allocated();assert peak<8*2**30;write_json(preflight,{'finite_forward_backward':True,'batch_size':4,'peak_allocated_bytes':peak,'available_bytes':free,'no_optimizer_step':True,'continue_directly_to_full_formal_training':True});del batch,y,model,out,loss;torch.cuda.empty_cache()
 baseline_path=RUN/'model/source_C02_comparator.joblib'
 source_rows=[json.loads(s) for s in (BASE/'features/train_records.jsonl').read_text().splitlines()];fitcases=set(split['training_C_cases']);ix=[i for i,r in enumerate(source_rows) if r['case_id'] in fitcases];actual_comparator_cases=sorted({source_rows[i]['case_id'] for i in ix})
 if not baseline_path.exists():
  data=np.load(BASE/'features/train.npz');baseline=clone(joblib.load(BASE/'model/classifier.joblib')).set_params(n_jobs=1);baseline.fit(data['X'][ix],data['y'][ix],sample_weight=data['sample_weight'][ix]);joblib.dump(baseline,baseline_path)
 baseline=joblib.load(baseline_path);sourcebaseline=[{'row':i,'prediction':int(baseline.predict(np.array(records[i]['baseline_geometry_features'],np.float32)[None])[0]),'class':records[i]['class']} for i in split['validation_rows']];write_json(RUN/'evaluation/SOURCE_BASELINE.json',{'rows':sourcebaseline,'correct':sum(r['prediction']==r['class'] for r in sourcebaseline),'classifier_sha256':sha256_file(baseline_path),'n_eligible_fit_cases':len(split['training_C_cases']),'n_actual_fit_cases':len(actual_comparator_cases),'actual_fit_cases':actual_comparator_cases,'n_actual_training_rows':len(ix),'source_records_sha256':sha256_file(BASE/'features/train_records.jsonl')})
 for row in sourcebaseline:records[row['row']]['source_baseline_class']=row['prediction']
 epoch=fit('development',split['train_rows'],records,split['validation_rows']);predictions=json.loads((RUN/'model/development_predictions.json').read_text());byrow={r['row']:r for r in sourcebaseline};fine_rows=[]
 for row in predictions:
  before=byrow[row['row']]['prediction'];after=refine_prediction(before,row['probabilities']);fine_rows.append({**row,'baseline_prediction':before,'fine_refined_prediction':after})
 baseline_correct=sum(r['baseline_prediction']==r['class'] for r in fine_rows);new_correct=sum(r['fine_refined_prediction']==r['class'] for r in fine_rows);write_json(RUN/'evaluation/SOURCE_RESULT.json',{'baseline_correct':baseline_correct,'new_correct':new_correct,'n_candidates':len(fine_rows),'source_gate_passed':new_correct>baseline_correct,'rows':fine_rows,'policy':'Refine only within baseline ICA/MCA coarse family and laterality; one source-selected epoch, no probability threshold search'})
 from scripts.astra6_e28.anatomy_diagnostics import run as anatomy_diagnostics
 anatomy_diagnostics(records,split)
 fit('final',split['final_rows'],records,epochs=epoch);write_json(RUN/'model/LOCKED.json',{'model_sha256':sha256_file(RUN/'model/final_last.pt'),'epochs':epoch,'full_formal_training_complete':True,'config_sha256':sha256_file(RUN/'config.json'),'source_split_sha256':sha256_file(RUN/'source_split.json'),'train_code_sha256':sha256_file(Path(__file__)),'architecture_sha256':sha256_file(Path(__file__).with_name('representation.py')),'no_MR40_CT5_fit':True})
if __name__=='__main__':main()
