from __future__ import annotations
import argparse,json,os,time,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scipy.ndimage import map_coordinates,label
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from scripts.astra6_e01.e01_common import *
from scripts.astra6_e03.run_e03 import BASE,read,rows,log,seed,load_image,component_records_fast
E03=P/'artifacts/astra6_e03_image_anatomy_20260909T025044Z'

def crop_coordinates(low,high):
 low,high=np.asarray(low),np.asarray(high);extent=np.maximum(high-low,1.);center=(low+high)/2
 return np.stack(np.meshgrid(*[center[i]+(np.arange(32)-15.5)/32*2*extent[i] for i in range(3)],indexing='ij'),axis=0)
def crop_volume(arr,low,high,order):return map_coordinates(arr,crop_coordinates(low,high).reshape(3,-1),order=order,mode='constant',cval=0,prefilter=False).reshape(32,32,32)
def prior():
 g=np.stack(np.meshgrid(*[(np.arange(32)-15.5)/8]*3,indexing='ij'));return (np.sum(g*g,axis=0)<=1).astype(np.float32)
def box_support():
 g=np.stack(np.meshgrid(*[(np.arange(32)-15.5)/8]*3,indexing='ij'));return (np.abs(g).max(0)<=1).astype(np.float32)
PRIOR=prior();SUPPORT=box_support()
def largest(mask):
 cc,n=label(mask,np.ones((3,3,3),np.uint8))
 if not n:return mask
 counts=np.bincount(cc.ravel());counts[0]=0;return cc==np.argmax(counts)
def tests():
 a=np.broadcast_to(np.arange(80)[:,None,None],(80,80,80)).astype(np.float32);c=crop_volume(a,[30,30,30],[50,50,50],1)
 assert np.allclose(c[:,16,16],40+(np.arange(32)-15.5)/32*40)
 assert np.all(PRIOR<=SUPPORT)
 a=np.zeros((9,9,9),bool);a[1:3,1:3,1:3]=1;a[7,7,7]=1;assert largest(a).sum()==8
 return {'box_normalized_sampling':True,'prior_support':True,'largest_component':True}

def prepare(run):
 rec=rows(BASE/'features/train_records.jsonl');groups={};mapping=[];mir=[]
 for r in rec:
  key=(r['case_id'],r['source_class_id'],r['component_id'],r['sample_index'])
  if key not in groups:groups[key]=(len(groups),r)
  mapping.append(groups[key][0]);mir.append(r['view']=='mirror')
 assert len(groups)==2493 and len(rec)==4986 and not any('center2' in r['case_id'] for r in rec)
 np.savez_compressed(run/'features/rows.npz',crop_index=mapping,mirrored=mir)
 shutil.copy2(BASE/'features/train_records.jsonl',run/'features/train_records.jsonl')
 progress=read(run/'features/progress.json') if (run/'features/progress.json').exists() else {'completed_cases':0,'normalization':{}}
 x=np.lib.format.open_memmap(run/'features/images.npy',mode='r+' if progress['completed_cases'] or progress.get('image_cached_cases',0) else 'w+',dtype=np.float16,shape=(2493,32,32,32));y=np.lib.format.open_memmap(run/'features/targets.npy',mode='r+' if progress['completed_cases'] or progress.get('image_cached_cases',0) else 'w+',dtype=np.uint8,shape=x.shape)
 cases=sorted({r['case_id'] for r in rec});bycase={c:[] for c in cases}
 for _,(i,r) in groups.items():bycase[r['case_id']].append((i,r))
 norms=progress['normalization']
 for n,cid in enumerate(cases,1):
  if n<=progress['completed_cases']:
   assert all(np.isfinite(x[i]).all() and y[i].sum()>0 for i,_ in bycase[cid]);continue
  if n<=progress.get('image_cached_cases',0):
   arr=None;aff,shape=load_nifti_geometry(DATA/f'images/{cid}_0000.nii.gz')
  else:arr,aff,norms[cid]=load_image(cid);shape=arr.shape
  gt,ga,gs=load_nifti(DATA/f'location_masks/{cid}.nii.gz');assert gs==shape and np.allclose(aff,ga,atol=1e-4)
  # Match current source components by class and exact base bounds, independent of component numbering.
  comps=component_records_fast(gt);component_cache={}
  for idx,r in bycase[cid]:
   key=(r['source_class_id'],r['component_id'])
   if key not in component_cache:
    base=next(z for _,z in bycase[cid] if z['source_class_id']==key[0] and z['component_id']==key[1] and z['sample_index']==0)
    comp=next(c for c in comps if c['class_id']==key[0] and np.array_equal(c['coords'].min(0)-.5,base['low']) and np.array_equal(c['coords'].max(0)+.5,base['high']))
    lo=comp['coords'].min(0);hi=comp['coords'].max(0)+1;small=np.zeros(tuple(hi-lo),np.uint8);small[tuple((comp['coords']-lo).T)]=1;component_cache[key]=(lo-1,np.pad(small,1))
   lo,small=component_cache[key];coords=crop_coordinates(r['low'],r['high'])
   if arr is not None:x[idx]=map_coordinates(arr,coords.reshape(3,-1),order=1,mode='constant',cval=0,prefilter=False).reshape(32,32,32)
   target=map_coordinates(small,(coords-lo[:,None,None,None]).reshape(3,-1),order=0,mode='constant',cval=0,prefilter=False).reshape(32,32,32)
   valid_fov=np.all((coords>=0)&(coords<=(np.asarray(shape)-1)[:,None,None,None]),axis=0)
   y[idx]=target*SUPPORT*valid_fov
   assert y[idx].sum()>0
  del arr,gt,comps
  x.flush();y.flush();write_json(run/'features/progress.json',{'completed_cases':n,'normalization':norms,'image_cached_cases':progress.get('image_cached_cases',0)});log(f'source segmentation crops {n}/218')
 assert np.isfinite(x).all()
 write_json(run/'features/normalization.json',norms);write_json(run/'features/SOURCE_READY.json',{'n_crops':2493,'n_rows':4986,'images_sha256':sha256_file(run/'features/images.npy'),'targets_sha256':sha256_file(run/'features/targets.npy')})

class SegData(Dataset):
 def __init__(self,run,indices):
  self.x=np.load(run/'features/images.npy',mmap_mode='r');self.y=np.load(run/'features/targets.npy',mmap_mode='r');self.f=dict(np.load(run/'features/rows.npz'));self.indices=indices
 def __len__(self):return len(self.indices)
 def __getitem__(self,i):
  row=self.indices[i];j=self.f['crop_index'][row];x=self.x[j];y=self.y[j]
  # Binary shape augmentation flips native first axis consistently; no location class transformation is used by this model.
  if self.f['mirrored'][row]:x=x[::-1];y=y[::-1]
  return torch.from_numpy(np.stack([x,PRIOR]).astype(np.float32)),torch.from_numpy(np.array(y[None],dtype=np.float32,copy=True))
def block(a,b):return nn.Sequential(nn.Conv3d(a,b,3,padding=1,bias=False),nn.GroupNorm(4,b),nn.SiLU(),nn.Conv3d(b,b,3,padding=1,bias=False),nn.GroupNorm(4,b),nn.SiLU())
class Segmenter(nn.Module):
 def __init__(self):
  super().__init__();self.a=block(2,8);self.b=block(8,16);self.c=block(16,32);self.d=block(48,16);self.e=block(24,8);self.out=nn.Conv3d(8,1,1)
 def forward(self,x):
  a=self.a(x);b=self.b(nn.functional.avg_pool3d(a,2));c=self.c(nn.functional.avg_pool3d(b,2));d=self.d(torch.cat([nn.functional.interpolate(c,size=b.shape[2:],mode='trilinear',align_corners=False),b],1));e=self.e(torch.cat([nn.functional.interpolate(d,size=a.shape[2:],mode='trilinear',align_corners=False),a],1));return self.out(e)
def fit(run,stage,trainix,devix=None,epochs=80):
 seed();model=Segmenter().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
 dl=DataLoader(SegData(run,trainix),batch_size=32,shuffle=True,num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(20260909));dv=DataLoader(SegData(run,devix),batch_size=32,num_workers=0) if devix is not None else None
 history=[];best=-1.;bestep=0;support=torch.from_numpy(SUPPORT[None,None]).cuda()
 start_epoch=1;resume=run/f'model/{stage}_resume.pt'
 if resume.exists():
  state=torch.load(resume,weights_only=False,map_location='cpu');model.load_state_dict(state['state_dict']);opt.load_state_dict(state['optimizer']);history=state['history'];best=state['best'];bestep=state['bestep'];start_epoch=state['epoch']+1
  torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);dl.generator.set_state(state['loader_rng'])
 elif stage=='development' and (run/'recovery_20260909/development_best.pt').exists():
  state=torch.load(run/'recovery_20260909/development_best.pt',weights_only=False,map_location='cpu');model.load_state_dict(state['state_dict']);best=state['dev_dice'];bestep=state['epoch'];start_epoch=bestep+1
  history=[h for h in read(run/'recovery_20260909/development_history.json') if h['epoch']<=bestep]
  write_json(run/'model/RECOVERY.json',{'resume_epoch':bestep,'optimizer_reset':True,'reason':'interrupted legacy run saved weights only; archived original history and weights','not_exact_resume':True})
 if start_epoch>epochs or (dv is not None and start_epoch>20 and start_epoch-1-bestep>=12):
  last=run/f'model/{stage}_last.pt';tmp=last.with_suffix('.tmp');torch.save({'state_dict':model.state_dict(),'epoch':start_epoch-1},tmp);os.replace(tmp,last)
  return bestep if dv is not None else model
 for ep in range(start_epoch,epochs+1):
  epoch_start=time.time()
  model.train();total=0.;n=0
  for x,y in dl:
   x=x.cuda();y=y.cuda();opt.zero_grad(set_to_none=True);z=model(x);p=z.sigmoid();dice=1-(2*(p*y).sum((1,2,3,4))+1)/(p.sum((1,2,3,4))+y.sum((1,2,3,4))+1);loss=.5*nn.functional.binary_cross_entropy_with_logits(z,y)+.5*dice.mean();assert torch.isfinite(loss), 'nonfinite loss';loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=loss.item()*len(x);n+=len(x)
  record={'epoch':ep,'train_loss':total/n,'train_seconds':time.time()-epoch_start,'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
  if dv is not None:
   model.eval();scores=[];bases=[]
   with torch.no_grad():
    for x,y in dv:
     prob=(model(x.cuda()).sigmoid()*support).cpu().numpy()[:,0];t=y.numpy()[:,0]>0
     for pp,tt in zip(prob,t):
      pred=largest(pp>=.5)
      if not pred.any():pred=PRIOR>0
      scores.append(float((2*(pred&tt).sum()+1e-6)/(pred.sum()+tt.sum()+1e-6)));bases.append(float((2*((PRIOR>0)&tt).sum()+1e-6)/(PRIOR.sum()+tt.sum()+1e-6)))
   score=float(np.mean(scores));record.update(dev_dice=score,dev_ellipsoid_dice=float(np.mean(bases)),dev_components=len(scores))
   if score>best:best=score;bestep=ep;torch.save({'state_dict':model.state_dict(),'epoch':ep,'dev_dice':score},run/f'model/{stage}_best.pt')
  history.append(record);write_json(run/f'model/{stage}_history.json',history);log(f'{stage} {record}')
  state={'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'epoch':ep,'history':history,'best':best,'bestep':bestep,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'loader_rng':dl.generator.get_state()}
  tmp=resume.with_suffix('.tmp');torch.save(state,tmp);os.replace(tmp,resume)
  if dv is not None and ep>=20 and ep-bestep>=12:break
 torch.save({'state_dict':model.state_dict(),'epoch':ep},run/f'model/{stage}_last.pt')
 return bestep if dv is not None else model

def train(run):
 rec=rows(run/'features/train_records.jsonl');split=read(E03/'source_split.json');dev=set(split['development_cases']);tr=[i for i,r in enumerate(rec) if r['case_id'] not in dev];dv=[i for i,r in enumerate(rec) if r['case_id'] in dev and r['view']=='original' and r['sample_index']==0]
 assert not {rec[i]['case_id'] for i in tr}&dev and len(dv)==52
 split['selection']='maximum E04 source development binary crop Dice; E03 case IDs reused only';write_json(run/'source_split.json',split);ep=fit(run,'development',tr,dv)
 write_json(run/'model/SELECTED_DURATION.json',{'epochs':ep,'selection':'maximum source development component crop Dice, threshold .5; fixed cleanup/fallback','max_epochs':80,'patience':12,'minimum_stop_epoch':20})
 model=fit(run,'final',list(range(len(rec))),epochs=ep)
 write_json(run/'model/LOCKED.json',{'time':datetime.now(timezone.utc).isoformat(),'model_sha256':sha256_file(run/'model/final_last.pt'),'code_sha256':sha256_file(Path(__file__)),'config_sha256':sha256_file(run/'config.json'),'source':read(run/'features/SOURCE_READY.json'),'center2_used_for_fit':False,'epochs':ep})
 del model;torch.cuda.empty_cache()

def infer(run):
 assert (run/'model/LOCKED.json').exists();assert sha256_file(run/'model/final_last.pt')==read(run/'model/LOCKED.json')['model_sha256']
 model=Segmenter().cuda();model.load_state_dict(torch.load(run/'model/final_last.pt',weights_only=False)['state_dict']);model.eval();ids=read(BASE/'eval_case_ids.json');base=rows(BASE/'candidate_predictions.jsonl');out=[];checks={};fallbacks=0
 for n,cid in enumerate(ids,1):
  arr,aff,_=load_image(cid);mask=np.zeros(arr.shape,np.uint8);bx,sc,_=load_boxes(BOX_DIR/f'{cid}_boxes.pkl');sel=select_candidates(bx,sc)
  for rank,(idx,score,low,high) in reversed(list(enumerate(sel))):
   r=next(r for r in base if r['case_id']==cid and r['original_index']==idx);assert np.array_equal(low,r['low']) and np.array_equal(high,r['high']) and score==r['score'];cl=r['predicted_class_id']
   crop=crop_volume(arr,low,high,1);x=np.stack([crop,PRIOR])[None].astype(np.float32)
   with torch.no_grad():prob=model(torch.from_numpy(x).cuda()).sigmoid().cpu().numpy()[0,0]
   lo=np.maximum(np.floor(low).astype(int),0);hi=np.minimum(np.ceil(high).astype(int),arr.shape);region=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi));native=np.stack(np.meshgrid(*[np.arange(a,b) for a,b in zip(lo,hi)],indexing='ij'),axis=0)
   center=(low+high)/2;extent=np.maximum(high-low,1);coords=(native-center[:,None,None,None])/(2*extent[:,None,None,None])*32+15.5
   nativeprob=map_coordinates(prob,coords.reshape(3,-1),order=1,mode='constant',cval=0,prefilter=False).reshape(tuple(hi-lo));fg=largest(nativeprob>=.5);fallback=not fg.any()
   if fallback:fill_ellipsoid(mask,low,high,cl);fallbacks+=1
   else:mask[region][fg]=cl
   out.append({'case_id':cid,'original_index':idx,'rank':rank,'score':score,'low':low.tolist(),'high':high.tolist(),'before_class':cl,'predicted_class_id':cl,'empty_fallback':fallback,'n_refined_voxels':int(fg.sum())})
  oa,osh=load_nifti_geometry(BASE/f'predictions/mr_center2_k05/{cid}.nii.gz');assert osh==arr.shape and np.allclose(aff,oa,atol=1e-4) and mask.max()<=52
  nifti_output(mask,DATA/f'images/{cid}_0000.nii.gz',run/f'predictions/mr_center2_k05/{cid}.nii.gz');checks[cid]={'native_geometry':True,'frozen_selection_and_classes':True,'draw_order':'low score first','n_candidates':len(sel)};del arr,mask;log(f'segmentation inference {n}/40')
 (run/'candidate_predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in out)+'\n');write_json(run/'prediction_validity.json',checks)
 write_json(run/'PREDICTIONS_HASHED_BEFORE_EVAL_GT.json',{'time':datetime.now(timezone.utc).isoformat(),'model_sha256':sha256_file(run/'model/final_last.pt'),'prediction_tree_sha256':sha256_tree(run/'predictions'),'n_cases':40,'n_candidates':len(out),'empty_fallbacks':fallbacks})
 del model;torch.cuda.empty_cache()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--stage',choices=['prepare','train','infer','evaluate'],required=True);a=ap.parse_args();r=a.run
 if a.stage in ['train','infer']:assert os.environ.get('CUDA_VISIBLE_DEVICES')=='GPU-a643ded3-193b-58e8-b362-be93dc8eac14' and torch.cuda.device_count()==1
 else:assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 assert tests();assert all(sha256_file(Path(p))==h for p,h in read(r/'config.json')['scorer_hashes'].items())
 start=time.time()
 if a.stage=='prepare':assert not (r/'features/SOURCE_READY.json').exists();prepare(r)
 elif a.stage=='train':assert not (r/'model/LOCKED.json').exists();train(r)
 elif a.stage=='infer':assert not (r/'PREDICTIONS_HASHED_BEFORE_EVAL_GT.json').exists();infer(r)
 else:
  assert not (r/'DONE.json').exists()
  from scripts.astra6_e04.evaluate import evaluate
  evaluate(r)
 write_json(r/f'logs/{a.stage}_runtime.json',{'seconds':time.time()-start,'time':datetime.now(timezone.utc).isoformat(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'output_bytes':sum(p.stat().st_size for p in r.rglob('*') if p.is_file())})
if __name__=='__main__':main()
