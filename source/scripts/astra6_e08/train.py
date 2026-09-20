import json,time,random,os
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from scripts.astra6_e08.common import *
from scripts.astra6_e01.e01_common import write_json,sha256_file

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
   x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True).float();gamma=torch.empty((len(x),1,1,1,1),device='cuda').uniform_(.8,1.25);x=x.clamp(0,1).pow(gamma);x=(x+.01*torch.randn_like(x)).clamp(0,1)
   for axis in [2,3,4]:
    if torch.rand((),device='cuda')<.5:x=x.flip(axis)
   opt.zero_grad(set_to_none=True);logits=model(x);loss=nn.functional.binary_cross_entropy_with_logits(logits,y);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss))
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
 save(RUN/f'model/{stage}_last.pt',{'state_dict':model.state_dict(),'epoch':history[-1]['epoch'],'architecture':'E06_shared_multiscale_CNN'})
 return bestep if dv is not None else model

def main():
 assert (RUN/'features/READY.json').exists();assert (RUN/'SOURCE_GEOMETRY_VALIDATED.json').exists();assert not (RUN/'model/LOCKED.json').exists();seed();records=[json.loads(s) for s in (RUN/'features/records.jsonl').read_text().splitlines()];all_ix=[i for i,r in enumerate(records) if '_mr_' in r['case_id']];tr=[i for i in all_ix if not records[i]['development']];dv=[i for i in all_ix if records[i]['development']];assert not {records[i]['group'] for i in tr}&{records[i]['group'] for i in dv}
 # One forward/backward geometry/VRAM check; immediately continue full planned fit.
 net=Filter().cuda();x,y=next(iter(DataLoader(Data(tr),batch_size=32)));assert x.shape[1:]==(2,32,32,32) and torch.isfinite(x).all();z=net(x.cuda());assert z.shape==(len(x),);nn.functional.binary_cross_entropy_with_logits(z,y.cuda().float()).backward();write_json(RUN/'model/PREFLIGHT.json',{'batch_shape':list(x.shape),'finite':True,'forward_backward':True,'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated()});del net,x,y,z;torch.cuda.empty_cache()
 ep=fit('development',tr,dv);pred=json.loads((RUN/'model/development_best_predictions.json').read_text());pp=np.array(pred['p']);yy=np.array(pred['y']);threshold=float(np.nextafter(np.min(pp[yy==1]),0.));write_json(RUN/'model/THRESHOLD.json',{'threshold':threshold,'selection':'minimum original source-dev positive probability, nextafter toward zero','source_recall':float(np.mean(pp[yy==1]>=threshold)),'source_negative_rejection':float(np.mean(pp[yy==0]<threshold)),'selected_epoch':ep,'center2_used':False});write_json(RUN/'model/SELECTED_DURATION.json',{'epochs':ep,'max_epochs':100,'min_stop_epoch':30,'patience':20,'selection':'lowest balanced source-dev BCE'})
 model=fit('final',all_ix,epochs=ep);write_json(RUN/'model/LOCKED.json',{'model_sha256':sha256_file(RUN/'model/final_last.pt'),'threshold_sha256':sha256_file(RUN/'model/THRESHOLD.json'),'code_sha256':sha256_file(Path(__file__)),'architecture_sha256':sha256_file(Path(__file__).with_name('common.py')),'config_sha256':sha256_file(RUN/'config.json'),'source':{'shared_cache':json.loads((RUN/'features/READY.json').read_text()),'used_row_indices':all_ix,'used_case_ids':sorted({records[i]['case_id'] for i in all_ix}),'CT_rows_used':False},'frozen_E04_segmentation_sha256':sha256_file(P/'artifacts/astra6_e04_crop_segmentation_20260909T031146Z/model/final_last.pt'),'frozen_E02_candidate_assignment_sha256':sha256_file(P/'artifacts/astra6_e02_multiscale_vessel_signature_20260908T173916Z/candidate_predictions.jsonl'),'epochs':ep,'center2_used_for_fit_or_calibration':False});print('FULL_FORMAL_TRAINING_COMPLETE',flush=True)
if __name__=='__main__':main()
