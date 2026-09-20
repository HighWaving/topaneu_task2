import json,time,random,os
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from scripts.astra6_e25.common import *
SEED=20260909
from scripts.astra6_e01.e01_common import write_json,sha256_file
from scripts.astra6_e25.prepare import group

def seed():
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
class Data(Dataset):
 def __init__(self,indices):
  self.x=np.load(RUN/'features/images.npy',mmap_mode='r');self.rec=[json.loads(s) for s in (RUN/'features/records.jsonl').read_text().splitlines()];self.indices=indices
 def __len__(self):return len(self.indices)
 def __getitem__(self,i):
  idx=self.indices[i];return torch.from_numpy(self.x[idx].astype(np.float32)),float(self.rec[idx]['y'])
def save(path,state):
 tmp=path.with_suffix('.tmp');torch.save(state,tmp);os.replace(tmp,path)
def fit(stage,indices,dev_indices=None,epochs=100):
 seed();model=Filter().cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01);data=Data(indices);ys=np.array([data.rec[i]['y'] for i in indices]);counts=np.bincount(ys,minlength=2);assert counts.min()>0
 gen=torch.Generator().manual_seed(SEED);sampler=WeightedRandomSampler(torch.tensor(1/counts[ys],dtype=torch.double),num_samples=len(indices),replacement=True,generator=gen);dl=DataLoader(data,batch_size=32,sampler=sampler,num_workers=0,pin_memory=True)
 dv=DataLoader(Data(dev_indices),batch_size=32,num_workers=0) if dev_indices is not None else None
 history=[];best=float('inf');bestep=0;start=1;path=RUN/f'model/{stage}_resume.pt'
 if path.exists():
  st=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(st['state_dict']);opt.load_state_dict(st['optimizer']);history=st['history'];best=st['best'];bestep=st['bestep'];start=st['epoch']+1;torch.set_rng_state(st['rng']);torch.cuda.set_rng_state_all(st['cuda_rng']);gen.set_state(st['sampler_rng']);np.random.set_state(st['numpy_rng']);random.setstate(st['python_rng'])
 for epoch in range(start,epochs+1):
  if dv is not None and epoch>30 and epoch-1-bestep>=20:break
  model.train();losses=[];t=time.time()
  for x,y in dl:
   x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True).float();gamma=torch.empty((len(x),1,1,1,1),device='cuda').uniform_(.8,1.25);image=x[:,:2].clamp(0,1).pow(gamma);image=(image+.01*torch.randn_like(image)).clamp(0,1);x=torch.cat([image,x[:,2:3]],dim=1)
   for axis in [2,3,4]:
    if torch.rand((),device='cuda')<.5:x=x.flip(axis)
   opt.zero_grad(set_to_none=True);logits=model(x);loss=nn.functional.binary_cross_entropy_with_logits(logits,y);assert torch.isfinite(loss), 'nonfinite filter loss';loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss))
  row={'epoch':epoch,'train_balanced_sample_BCE':float(np.mean(losses)),'seconds':time.time()-t}
  if dv is not None:
   model.eval();ps=[];yy=[]
   with torch.inference_mode():
    for x,y in dv:ps.extend(model(x.cuda()).sigmoid().cpu().tolist());yy.extend(y.tolist())
   pp=np.clip(ps,1e-7,1-1e-7);yy=np.array(yy);bce=-(yy*np.log(pp)+(1-yy)*np.log(1-pp));val=float(.5*(bce[yy==0].mean()+bce[yy==1].mean()));row.update(dev_balanced_BCE=val,dev_AUROC=float(roc_auc_score(yy,pp)),dev_positive=int(yy.sum()),dev_negative=int((yy==0).sum()))
   if val<best:
    best=val;bestep=epoch;save(RUN/f'model/{stage}_best.pt',{'state_dict':model.state_dict(),'epoch':epoch,'validation_BCE':val});write_json(RUN/'model/development_best_predictions.json',{'rows':dev_indices,'y':yy.tolist(),'p':ps,'epoch':epoch})
  history.append(row);write_json(RUN/f'model/{stage}_history.json',history);print(json.dumps({'stage':stage,**row}),flush=True)
  save(path,{'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'history':history,'best':best,'bestep':bestep,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'sampler_rng':gen.get_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate()})
 save(RUN/f'model/{stage}_last.pt',{'state_dict':model.state_dict(),'epoch':history[-1]['epoch'],'architecture':'E25_OOF_image_shape_CNN'})
 return bestep if dv is not None else model

def main():
 config=json.loads((RUN/'config.json').read_text());assert (RUN/'features/READY.json').exists()
 if (RUN/'model/LOCKED.json').exists():return
 (RUN/'model').mkdir(parents=True,exist_ok=True);split=json.loads((RUN/'source_split.json').read_text());tr=split['training_rows'];dv=split['development_rows'];records=[json.loads(s) for s in (RUN/'features/records.jsonl').read_text().splitlines()];assert all(records[i]['fold']==0 for i in tr+dv);assert not {group(records[i]['case_id']) for i in tr}&{group(records[i]['case_id']) for i in dv};assert all(records[i]['development'] for i in dv)
 ep=fit('development',tr,dv);pred=json.loads((RUN/'model/development_best_predictions.json').read_text());pp=np.asarray(pred['p']);yy=np.asarray(pred['y']);deployed=np.array([records[i]['deployed'] for i in pred['rows']]);positive=deployed&(yy==1);assert positive.any();threshold=float(np.nextafter(pp[positive].min(),0.))
 # Baseline development members never fitted any existing F12 development case.
 from scripts.astra6_e14.train import combine
 parent=P/'artifacts/astra6_e12_MR_expanded_fp_filter_20260909';e14=P/'artifacts/astra6_e14_MR_filter_seed_stability_20260909';paths=[parent/'model/development_best.pt']+[e14/f'seed{s}/model/development_last.pt' for s in [20260910,20260911]];baseline=combine(paths).cuda().eval();images=np.load(RUN/'features/images.npy',mmap_mode='r');bp=[]
 with torch.inference_mode():
  for start in range(0,len(dv),32):bp.extend(baseline(torch.from_numpy(np.asarray(images[dv[start:start+32],:2],np.float32)).cuda()).sigmoid().cpu().tolist())
 bp=np.asarray(bp);bt=float(np.nextafter(bp[positive].min(),0.));negative=deployed&(yy==0);base_fp=int((negative&(bp>=bt)).sum());new_fp=int((negative&(pp>=threshold)).sum());components={(records[i]['case_id'],k) for i in pred['rows'] if records[i]['deployed'] for k in records[i]['matched_components']};passed=len(components)>=10 and int(negative.sum())>=10 and base_fp-new_fp>=max(2,int(np.ceil(.2*base_fp)))
 write_json(RUN/'model/THRESHOLD.json',{'threshold':threshold,'selection':'minimum source-development positive probability in original D.3/top5 pool, preserving every matched source candidate; no MR40 calibration','selected_epoch':ep,'source_only':True});write_json(RUN/'evaluation/SOURCE_RESULT.json',{'positive_candidates':int(positive.sum()),'matched_components':len(components),'negative_candidates':int(negative.sum()),'baseline_equal_coverage_threshold':bt,'new_threshold':threshold,'baseline_FP':base_fp,'new_FP':new_fp,'source_gate_passed':passed,'baseline_member_sha256':[sha256_file(f) for f in paths],'baseline_probabilities':bp.tolist(),'new_probabilities':pp.tolist(),'rows':dv,'limitations':'Both filters held out these F calibration cases; baseline trained on more original in-sample proposals. New F development uses only fold0 OOF cases to prevent crossfold upstream target leakage. Final F uses both folds.'});del baseline;torch.cuda.empty_cache()
 fit('final',split['final_rows'],epochs=ep);write_json(RUN/'model/LOCKED.json',{'model_sha256':sha256_file(RUN/'model/final_last.pt'),'threshold_sha256':sha256_file(RUN/'model/THRESHOLD.json'),'config_sha256':sha256_file(RUN/'config.json'),'code_sha256':sha256_file(Path(__file__)),'architecture_sha256':sha256_file(Path(__file__).with_name('common.py')),'source_sha256':sha256_file(RUN/'features/READY.json'),'epochs':ep,'full_formal_training_complete':True,'source_gate_passed':passed,'MR40_CT5_fit':False});print('E25_FULL_FORMAL_TRAINING_COMPLETE',ep,base_fp,new_fp,'source_gate',passed,flush=True)
if __name__=='__main__':main()
