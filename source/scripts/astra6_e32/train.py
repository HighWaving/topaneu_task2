"""Full fixed-budget matched segmentation fits with exact epoch-boundary resumption."""
import argparse,json,os,random,time,math
import numpy as np,torch
from scipy.ndimage import map_coordinates
from torch import nn
from scripts.astra6_e32.common import *
from scripts.astra6_e04.run_e04 import Segmenter,block,largest
from scripts.astra6_e01.e01_common import write_json,sha256_file
SEED=20260910
class Model(Segmenter):
 def __init__(self):
  super().__init__();self.a=block(3,8)
 def forward(self,x):
  # Identical parameterization/initialization; vessel channel is never consumed.
  x=torch.cat([x[:,:1],torch.zeros_like(x[:,1:2]),x[:,2:]],dim=1)
  return super().forward(x)
def atomic(path,data):
 temp=path.with_suffix('.tmp');torch.save(data,temp);os.replace(temp,path)
def source_score(prob,r,origin,step,target):
 lo,fg=native_foreground(prob,origin,step,r['image_shape']);fallback=not fg.any()
 if fallback:
  low,high=np.array(r['low']),np.array(r['high']);lo=np.maximum(0,np.floor(low).astype(int));hi=np.minimum(r['image_shape'],np.ceil(high).astype(int));g=np.stack(np.meshgrid(*[np.arange(a,b) for a,b in zip(lo,hi)],indexing='ij'));c=(g-((low+high)/2)[:,None,None,None])/(2*np.maximum(high-low,1))[:,None,None,None]*32+15.5
  fg=largest(map_coordinates(target['fallback_probability'],c.reshape(3,-1),order=1,mode='constant',cval=0,prefilter=False).reshape(tuple(hi-lo))>=.5)
 coords=target['coords']-lo;valid=np.all((coords>=0)&(coords<np.array(fg.shape)),axis=1);inside=coords[valid];intersection=int(fg[tuple(inside.T)].sum()) if len(inside) else 0
 return 2*intersection/max(1,int(fg.sum())+r['GT_voxels']),fallback,int(fg.sum())
def seed():
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
def pools(records,indices):
 result={}
 for i in indices:
  r=records[i];result.setdefault(r['kind'],{}).setdefault((r['case_id'],r['component_index']),[]).append(i)
 return {k:list(v.values()) for k,v in result.items()}
def main(arm):
 torch.set_num_threads(2);out=RUN/arm;(out/'model').mkdir(exist_ok=True)
 if (out/'model/LOCKED.json').exists():return
 assert (RUN/'features/READY.json').exists() and (RUN/'native_validation/READY.json').exists()
 assert (RUN/'SOURCE_SHAPE_QUALITY_EXCLUSION.json').exists(),'Source quality finalization must precede any fit'
 cfg=json.loads((RUN/'config.json').read_text());split=json.loads((RUN/'source_split.json').read_text());records=[json.loads(s) for s in (RUN/'features/records.jsonl').read_text().splitlines()]
 ready=json.loads((RUN/'features/READY.json').read_text());quality=json.loads((RUN/'SOURCE_SHAPE_QUALITY_EXCLUSION.json').read_text())
 hashes={k:sha256_file(RUN/path) for k,path in [('config_sha256','config.json'),('source_split_sha256','source_split.json'),('records_sha256','features/records.jsonl')]}
 assert all(ready[k]==v for k,v in hashes.items())
 assert len(split['train_rows'])==quality['after']['train_rows'] and len(split['final_rows'])==quality['after']['final_rows']
 assert all(records[i]['GT_voxels']>1 for i in split['train_rows']+split['final_rows'])
 manifest=out/'model/FIT_INPUTS.json'
 if manifest.exists():assert json.loads(manifest.read_text())['hashes']==hashes
 else:write_json(manifest,{'hashes':hashes,'train_rows':split['train_rows'],'final_rows':split['final_rows'],'validation_rows':split['validation_rows'],'quality_finalized_before_any_optimizer_update':True})
 assert all(records[i][arm+'_target_voxels']>0 for i in split['train_rows']+split['final_rows']),'Representation erased a positive training target; investigate before fitting'
 x=np.load(out/'features/images.npy');y=np.load(out/'features/targets.npy');geo=np.load(out/'features/geometry.npz');targets={i:dict(np.load(RUN/f'native_validation/{i}.npz')) for i in split['validation_rows']}
 # One forward/backward geometry/memory preflight, no optimizer update.
 seed();m=Model().cuda();xx=torch.from_numpy(x[:8].astype(np.float32)).cuda();yy=torch.from_numpy(y[:8,None].astype(np.float32)).cuda()
 with torch.autocast('cuda',dtype=torch.float16):zz=m(xx);loss=nn.functional.binary_cross_entropy_with_logits(zz,yy)
 assert torch.isfinite(loss);loss.backward();peak=torch.cuda.max_memory_allocated();assert peak<10*2**30
 write_json(out/'model/PREFLIGHT.json',{'batch':8,'finite_forward_backward':True,'peak_gpu_bytes':peak,'parameters':sum(p.numel() for p in m.parameters()),'no_optimizer_update':True});del m,xx,yy,zz,loss;torch.cuda.empty_cache()
 chosen=None
 for stage in ['development','final']:
  if stage=='final':chosen=json.loads((out/'model/SOURCE_SELECTION.json').read_text())['selected_epoch']
  indices=split['train_rows'] if stage=='development' else split['final_rows'];pool=pools(records,indices);kinds=sorted(pool)
  source_components=len({(records[i]['case_id'],records[i]['component_index']) for i in split['train_rows']});ncomp=len({(records[i]['case_id'],records[i]['component_index']) for i in indices});steps=128 if stage=='development' else math.ceil(128*ncomp/source_components);epochs=120 if stage=='development' else chosen
  seed();rng=np.random.default_rng(SEED);model=Model().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=120,eta_min=.00001);scaler=torch.amp.GradScaler('cuda');history=[];best=-1;bestep=None;resume=out/f'model/{stage}_resume.pt';start=1
  if resume.exists():
   s=torch.load(resume,map_location='cpu',weights_only=False);assert s['fit_input_hashes']==hashes;model.load_state_dict(s['state_dict']);opt.load_state_dict(s['optimizer']);scheduler.load_state_dict(s['scheduler']);scaler.load_state_dict(s['scaler']);rng.bit_generator.state=s['sampler_rng'];torch.set_rng_state(s['torch_rng']);torch.cuda.set_rng_state_all(s['cuda_rng']);random.setstate(s['python_rng']);np.random.set_state(s['numpy_rng']);history=s['history'];best=s['best'];bestep=s['best_epoch'];start=s['epoch']+1
  for ep in range(start,epochs+1):
   begin=time.monotonic();model.train();total=0
   for _ in range(steps):
    batch=[]
    for j in range(8):
     kind=kinds[int(rng.integers(len(kinds)))];groups=pool[kind];component=groups[int(rng.integers(len(groups)))];batch.append(component[int(rng.integers(len(component)))])
    xx=torch.from_numpy(x[batch].astype(np.float32)).cuda();yy=torch.from_numpy(y[batch,None].astype(np.float32)).cuda()
    for axis in [2,3,4]:
     if torch.rand((),device='cuda')<.5:xx=xx.flip(axis);yy=yy.flip(axis)
    gamma=torch.empty((8,1,1,1,1),device='cuda').uniform_(.8,1.25);image=(xx[:,:1].clamp(0,1).pow(gamma)+.01*torch.randn_like(xx[:,:1])).clamp(0,1);xx=torch.cat([image,xx[:,1:]],1);opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16):
     zz=model(xx);pp=zz.float().sigmoid();dice=1-(2*(pp*yy).sum((1,2,3,4))+1)/(pp.sum((1,2,3,4))+yy.sum((1,2,3,4))+1);loss=.5*nn.functional.binary_cross_entropy_with_logits(zz,yy)+.5*dice.mean()
    assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(opt);scaler.update();total+=float(loss)
   scheduler.step();record={'epoch':ep,'train_loss':total/steps,'updates':steps*ep,'seconds':time.monotonic()-begin,'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
   if stage=='development' and ep%5==0:
    model.eval();sc=[]
    with torch.inference_mode():
     for i in split['validation_rows']:
      with torch.autocast('cuda',dtype=torch.float16):pr=model(torch.from_numpy(x[i:i+1].astype(np.float32)).cuda()).float().sigmoid().cpu().numpy()[0,0]
      score,fallback,vol=source_score(pr,records[i],geo['origin'][i],geo['step'][i],targets[i]);sc.append({'row':i,'case_id':records[i]['case_id'],'Dice':score,'fallback':fallback,'predicted_voxels':vol,'GT_voxels':records[i]['GT_voxels']})
    means=[np.mean([r['Dice'] for r in sc if r['case_id']==c]) for c in sorted({r['case_id'] for r in sc})];score=float(np.mean(means));record.update(source_case_mean_Dice=score,source_candidate_mean_Dice=float(np.mean([r['Dice'] for r in sc])),fallback_count=sum(r['fallback'] for r in sc))
    if ep>=20 and score>best:
     best=score;bestep=ep;atomic(out/'model/development_best.pt',{'state_dict':model.state_dict(),'epoch':ep,'source_case_mean_Dice':score});write_json(out/'model/SOURCE_SELECTION.json',{'selected_epoch':ep,'source_case_mean_Dice':score,'rows':sc,'selection_uses_MR40':False})
   record['seconds']=time.monotonic()-begin;history.append(record);write_json(out/f'model/{stage}_history.json',history)
   state={'fit_input_hashes':hashes,'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'sampler_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'epoch':ep,'history':history,'best':best,'best_epoch':bestep};atomic(resume,state);print(arm,stage,json.dumps(record),flush=True)
  atomic(out/f'model/{stage}_last.pt',{'state_dict':model.state_dict(),'epoch':epochs,'arm':arm});del model,opt,scheduler,scaler;torch.cuda.empty_cache()
 write_json(out/'model/LOCKED.json',{'full_formal_training_complete':True,'fit_input_hashes':hashes,'development_epochs':120,'selected_epoch':chosen,'final_epochs':chosen,'model_sha256':sha256_file(out/'model/final_last.pt'),'config_sha256':sha256_file(RUN/'config.json'),'training_code_sha256':sha256_file(Path(__file__)),'no_MR40_CT5_gradient_fit':True})
 print('E32 ARM COMPLETE',arm,chosen,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['normalized'],required=True);main(p.parse_args().arm)
