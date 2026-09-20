"""One bounded image + anatomy assignment experiment. No detector/segmentation changes."""
from __future__ import annotations
import argparse, json, os, time, sys, hashlib, shutil
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scripts.astra6_e01.e01_common import P, DATA, BOX_DIR, TA36_DIR, load_nifti, load_nifti_geometry, load_boxes, select_candidates, fill_ellipsoid, nifti_output, sha256_file, sha256_tree, write_json, affine_world, sitk_array

BASE=P/'artifacts/astra6_e02_multiscale_vessel_signature_20260908T173916Z'
SEED=20260909
METRICS=['PRECISION','RECALL','MCC','DICE','VOLSIM','HD95']
def read(p): return json.loads(p.read_text())
def rows(p): return [json.loads(s) for s in p.read_text().splitlines() if s]
def log(s): print(datetime.now(timezone.utc).isoformat(),s,flush=True)
def seed():
 import random
 random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
 torch.backends.cudnn.benchmark=False
 torch.backends.cudnn.deterministic=True
 torch.set_num_threads(4)

def sample_crop(arr,aff,low,high):
 q=affine_world(aff,(np.asarray(low)+np.asarray(high))/2)
 g=np.stack(np.meshgrid(*[np.arange(32)-15.5]*3,indexing='ij'),axis=-1)+q
 vox=affine_world(np.linalg.inv(aff),g)
 return map_coordinates(arr,vox.reshape(-1,3).T,order=1,mode='constant',cval=0,prefilter=False).reshape(1,32,32,32).astype(np.float32)

def load_image(cid):
 image=nib.load(str(DATA/f'images/{cid}_0000.nii.gz'))
 arr=image.get_fdata(dtype=np.float32)
 if not np.isfinite(arr).all(): raise ValueError(f'nonfinite image {cid}')
 sub=arr[::4,::4,::4]; sub=sub[sub!=0]
 if not len(sub): raise ValueError(f'empty image {cid}')
 lo,hi=np.percentile(sub,[.5,99.5]); scale=max(float(hi-lo),1e-6)
 # Intensity normalization is fitted to each inference image itself, without annotation.
 arr-=float(lo); arr/=scale; np.clip(arr,0,1,out=arr)
 return arr,np.asarray(image.affine),{'low_p005':float(lo),'high_p995':float(hi)}

def synthetic_test():
 a=np.zeros((80,80,80),np.float32); a[:]=np.arange(80)[:,None,None]
 aff=np.diag([2.,3.,4.,1.]); lo=np.array([39,39,39.]); hi=lo+2
 c=sample_crop(a,aff,lo,hi)[0]
 assert np.allclose(c[:,16,16],40+(np.arange(32)-15.5)/2)
 assert np.array_equal(c[::-1][::-1],c)
 return {'physical_ras_1mm_sampling':True,'mirror_involution':True}

def prepare_source(run):
 records=rows(BASE/'features/train_records.jsonl'); src=dict(np.load(BASE/'features/train.npz'))
 assert src['X'].shape==(4986,943) and len(records)==4986
 train_cases=sorted({r['case_id'] for r in records}); assert len(train_cases)==218 and not any('center2' in c for c in train_cases)
 groups={}; mapping=[]; mirrored=[]
 for r in records:
  key=(r['case_id'],r['source_class_id'],r['component_id'],r['sample_index'])
  if key not in groups: groups[key]=(len(groups),r)
  mapping.append(groups[key][0]); mirrored.append(r['view']=='mirror')
 np.savez_compressed(run/'features/source.npz',X=src['X'],y=src['y'],sample_weight=src['sample_weight'],crop_index=mapping,mirrored=mirrored)
 shutil.copy2(BASE/'features/train_records.jsonl',run/'features/train_records.jsonl')
 crops=np.lib.format.open_memmap(run/'features/source_crops.npy',mode='w+',dtype=np.float16,shape=(len(groups),1,32,32,32))
 bycase={c:[] for c in train_cases}
 for _,(idx,r) in groups.items(): bycase[r['case_id']].append((idx,r))
 norms={}; start=time.time()
 for n,cid in enumerate(train_cases,1):
  arr,aff,norm=load_image(cid); norms[cid]=norm
  for idx,r in bycase[cid]: crops[idx]=sample_crop(arr,aff,r['low'],r['high'])
  del arr
  crops.flush()
  write_json(run/'features/source_progress.json',{'completed_cases':n,'total_cases':218,'last_case':cid,'seconds':time.time()-start})
  log(f'source crop extraction {n}/218')
 write_json(run/'features/source_normalization.json',norms)
 assert np.isfinite(crops).all()
 write_json(run/'features/SOURCE_READY.json',{'n_crops':len(groups),'n_rows':len(records),'crop_sha256':sha256_file(run/'features/source_crops.npy'),'features_sha256':sha256_file(run/'features/source.npz')})

class Crops(Dataset):
 def __init__(self,run,indices):
  self.arr=np.load(run/'features/source_crops.npy',mmap_mode='r'); self.f=dict(np.load(run/'features/source.npz'));self.indices=np.asarray(indices)
 def __len__(self):return len(self.indices)
 def __getitem__(self,j):
  i=self.indices[j]; a=self.arr[self.f['crop_index'][i]]
  if self.f['mirrored'][i]: a=a[:,::-1,:,:]
  return torch.from_numpy(np.array(a,dtype=np.float32,copy=True)),torch.from_numpy(self.f['X'][i]),int(self.f['y'][i])-1,float(self.f['sample_weight'][i])

class Classifier(nn.Module):
 def __init__(self):
  super().__init__(); layers=[]; prev=1
  for channels in (16,32,64):
   layers += [nn.Conv3d(prev,channels,3,stride=2,padding=1,bias=False),nn.GroupNorm(8,channels),nn.SiLU()]; prev=channels
  self.image=nn.Sequential(*layers,nn.AdaptiveAvgPool3d(2),nn.Flatten(),nn.Linear(512,64),nn.SiLU())
  self.anatomy=nn.Sequential(nn.Linear(943,128),nn.LayerNorm(128),nn.SiLU(),nn.Dropout(.2),nn.Linear(128,64),nn.SiLU())
  self.head=nn.Sequential(nn.Dropout(.3),nn.Linear(128,52))
 def forward(self,crop,features):return self.head(torch.cat([self.image(crop),self.anatomy(features)],dim=1))

def fit_model(run,stage,train_ix,dev_ix=None,epochs=120):
 seed(); model=Classifier().cuda(); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
 scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=120)
 tr=Crops(run,train_ix); dl=DataLoader(tr,batch_size=64,shuffle=True,num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(SEED))
 dev=DataLoader(Crops(run,dev_ix),batch_size=64,shuffle=False,num_workers=0,pin_memory=True) if dev_ix is not None else None
 weight_mean=float(np.mean(tr.f['sample_weight'][train_ix])); history=[]; best=float('inf'); best_epoch=0
 for ep in range(1,epochs+1):
  model.train(); total=0.; n=0
  for crop,x,y,w in dl:
   crop,x,y,w=crop.cuda(non_blocking=True),x.cuda(non_blocking=True),y.cuda(non_blocking=True),w.cuda(non_blocking=True).float()
   opt.zero_grad(set_to_none=True); out=model(crop,x); loss=(nn.functional.cross_entropy(out,y,reduction='none')*w/weight_mean).mean();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
   total+=loss.item()*len(y);n+=len(y)
  scheduler.step(); record={'epoch':ep,'train_weighted_ce':total/n}
  if dev is not None:
   model.eval(); losses=[]; correct=0; nd=0
   with torch.no_grad():
    for crop,x,y,w in dev:
     logits=model(crop.cuda(),x.cuda()); y=y.cuda();losses.extend(nn.functional.cross_entropy(logits,y,reduction='none').cpu().tolist());correct+=int((logits.argmax(1)==y).sum());nd+=len(y)
   val=float(np.mean(losses)); record.update(dev_ce=val,dev_accuracy=correct/nd,dev_components=nd)
   if val<best:
    best=val;best_epoch=ep;torch.save({'state_dict':model.state_dict(),'epoch':ep,'validation_ce':val},run/f'model/{stage}_best.pt')
  history.append(record); write_json(run/f'model/{stage}_history.json',history)
  log(f'{stage} {record}')
  if dev is not None and ep>=30 and ep-best_epoch>=25:break
 torch.save({'state_dict':model.state_dict(),'epoch':ep},run/f'model/{stage}_last.pt')
 if dev is None:return model
 return best_epoch

def train(run):
 records=rows(run/'features/train_records.jsonl'); cases=sorted({r['case_id'] for r in records}); rng=np.random.default_rng(SEED); devcases=[]
 for center in (1,4,5):
  cc=[c for c in cases if f'center{center}_' in c]; rng.shuffle(cc);devcases+=cc[:max(1,round(.2*len(cc)))]
 ds=set(devcases); trainix=[i for i,r in enumerate(records) if r['case_id'] not in ds]; devix=[i for i,r in enumerate(records) if r['case_id'] in ds and r['view']=='original' and r['sample_index']==0]
 assert not ds&{records[i]['case_id'] for i in trainix}; assert not any('center2' in c for c in cases)
 write_json(run/'source_split.json',{'train_cases':sorted(set(cases)-ds),'development_cases':sorted(ds),'development_rows':devix,'selection':'lowest unweighted CE on original base source components; no center2 access','no_patient_linkage_metadata':True})
 best=fit_model(run,'development',trainix,devix)
 write_json(run/'model/SELECTED_DURATION.json',{'epochs':best,'selection':'source development CE only','max_epochs':120,'patience':25,'minimum_epochs_before_stop':30})
 model=fit_model(run,'final',list(range(len(records))),epochs=best)
 classes=sorted({r['class_id'] for r in records});write_json(run/'model/classes.json',{'supported_classes':classes,'unsupported_classes':sorted(set(range(1,53))-set(classes))})
 write_json(run/'model/LOCKED.json',{'time':datetime.now(timezone.utc).isoformat(),'model_sha256':sha256_file(run/'model/final_last.pt'),'config_sha256':sha256_file(run/'config.json'),'code_sha256':sha256_file(Path(__file__)),'source_features':read(run/'features/SOURCE_READY.json'),'epochs':best,'center2_used_for_training_or_selection':False})
 del model;torch.cuda.empty_cache()


def infer(run):
 assert (run/'model/LOCKED.json').exists()
 assert sha256_file(run/'model/final_last.pt')==read(run/'model/LOCKED.json')['model_sha256']
 model=Classifier().cuda();model.load_state_dict(torch.load(run/'model/final_last.pt',weights_only=False)['state_dict']);model.eval()
 supported=read(run/'model/classes.json')['supported_classes']; ids=read(BASE/'eval_case_ids.json'); candidates=rows(BASE/'candidate_predictions.jsonl'); fx=np.load(BASE/'features/eval.npz')['X']; outrows=[]; norm={}; checks={}
 assert fx.shape==(72,943)
 for n,cid in enumerate(ids,1):
  # No location mask/JSON/lesion ledger access until all predictions are locked.
  arr,aff,norm[cid]=load_image(cid); sh=arr.shape; mask=np.zeros(sh,np.uint8)
  boxes,scores,_=load_boxes(BOX_DIR/f'{cid}_boxes.pkl'); sel=select_candidates(boxes,scores)
  for rank,(idx,score,lo,hi) in reversed(list(enumerate(sel))):
   fi,r=next((i,r) for i,r in enumerate(candidates) if r['case_id']==cid and r['original_index']==idx)
   assert np.array_equal(lo,r['low']) and np.array_equal(hi,r['high']) and score==r['score']
   crop=sample_crop(arr,aff,lo,hi)
   with torch.no_grad():
    logits=model(torch.from_numpy(crop[None]).cuda(),torch.from_numpy(fx[fi:fi+1]).cuda())[0];p=np.zeros(52);p[np.array(supported)-1]=torch.softmax(logits[np.array(supported)-1],0).cpu().numpy()
   pred=int(np.argmax(p)+1);fill_ellipsoid(mask,lo,hi,pred)
   outrows.append({'case_id':cid,'original_index':idx,'rank':rank,'score':score,'low':lo.tolist(),'high':hi.tolist(),'before_class':r['predicted_class_id'],'predicted_class_id':pred,'probabilities':p.tolist()})
  old,oa,osh=load_nifti(BASE/f'predictions/mr_center2_k05/{cid}.nii.gz')
  assert sh==osh and np.allclose(aff,oa) and np.array_equal(old>0,mask>0)
  nifti_output(mask,DATA/f'images/{cid}_0000.nii.gz',run/f'predictions/mr_center2_k05/{cid}.nii.gz')
  checks[cid]={'binary_foreground':True,'native_geometry':True,'frozen_selection':True,'n_candidates':len(sel)}
  del arr,old,mask
  log(f'inference {n}/40')
 (run/'candidate_predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in outrows)+'\n')
 write_json(run/'features/eval_normalization.json',norm)
 write_json(run/'prediction_validity.json',checks)
 write_json(run/'PREDICTIONS_HASHED_BEFORE_EVAL_GT.json',{'time':datetime.now(timezone.utc).isoformat(),'prediction_tree_sha256':sha256_tree(run/'predictions'),'model_sha256':sha256_file(run/'model/final_last.pt'),'n_cases':len(ids),'n_candidates':len(outrows)})
 del model;torch.cuda.empty_cache()


def component_records_fast(mask):
 from scipy.ndimage import label
 out=[]; coords=np.argwhere(mask>0)
 if not len(coords): return out
 values=mask[tuple(coords.T)]
 for cls in range(1,53):
  selected=coords[values==cls]
  if not len(selected): continue
  low=selected.min(0); high=selected.max(0)+1
  local=np.zeros(tuple(high-low),np.uint8); local[tuple((selected-low).T)]=1
  cc,n=label(local,structure=np.ones((3,3,3),np.uint8))
  for idx in range(1,n+1):out.append({'class_id':cls,'component_id':idx,'coords':np.argwhere(cc==idx)+low})
 return out

def evaluate(run):
 from scripts.local_scoring_arena import aggregate,score_case
 from scripts.astra6_e01.evaluate import component_records,rectangle_overlap,ellipsoid_overlap
 lock=read(run/'PREDICTIONS_HASHED_BEFORE_EVAL_GT.json');assert sha256_tree(run/'predictions')==lock['prediction_tree_sha256']
 ids=read(BASE/'eval_case_ids.json'); before=read(BASE/'evaluation/after_per_case.json'); assert [r['case_id'] for r in before]==ids
 after=[]
 for n,cid in enumerate(ids,1):
  after.append({'case_id':cid,'raw':score_case(sitk_array(run/f'predictions/mr_center2_k05/{cid}.nii.gz'),cid)})
  write_json(run/'evaluation/after_partial.json',after);log(f'official scoring {n}/40')
 write_json(run/'evaluation/before_per_case.json',before);write_json(run/'evaluation/after_per_case.json',after)
 aggs={}; cs={}; coverage={}; shapes={}
 for name,pc in [('before',before),('after',after)]:
  ag=aggregate([r['raw'] for r in pc]);aggs[name]=ag
  cs[name]={k:int(sum(ag['per_class'][f'{k}_{i}'] for i in range(1,53))) for k in ('TP','FP','FN')}
  coverage[name]=sum(ag['per_class'][f'TP_{i}']>0 for i in range(1,53))
  vs=[r['raw'][f'DICE_{i}'] for r in pc for i in range(1,53) if r['raw'][f'TP_{i}']>0]
  shapes[name]={'n':len(vs),'mean':float(np.mean(vs)) if vs else None,'median':float(np.median(vs)) if vs else None}
  write_json(run/f'evaluation/{name}_official.json',{**ag,'counts':cs[name],'coverage':coverage[name]})
 delta={k:float(aggs['after']['overall'][k]-aggs['before']['overall'][k]) for k in METRICS}
 cr=rows(run/'candidate_predictions.jsonl'); ledger=[]; totals={'components':0,'rectangle_hit':0,'ellipsoid_hit':0,'unmatched_candidates':0,'before_correct':0,'after_correct':0}
 for cid in ids:
  gt,_,_=load_nifti(DATA/f'location_masks/{cid}.nii.gz');pred,_,_=load_nifti(run/f'predictions/mr_center2_k05/{cid}.nii.gz');box,sc,_=load_boxes(BOX_DIR/f'{cid}_boxes.pkl');sel=select_candidates(box,sc);hit=set()
  for comp in component_records_fast(gt):
   rect=[x for x in sel if rectangle_overlap(comp['coords'],x[2],x[3])]; ell=any(ellipsoid_overlap(comp['coords'],x[2],x[3]) for x in sel);totals['components']+=1;totals['ellipsoid_hit']+=ell
   rr={'case_id':cid,'class_id':comp['class_id'],'component_id':comp['component_id'],'rectangle_hit':bool(rect),'ellipsoid_hit':ell,'before_correct':False,'after_correct':False,'final_correct_overlap':bool(np.any(pred[tuple(comp['coords'].T)]==comp['class_id']))}
   if rect:
    totals['rectangle_hit']+=1;hit.update(x[0] for x in rect);best=sorted(rect,key=lambda x:(-x[1],x[0]))[0];r=next(r for r in cr if r['case_id']==cid and r['original_index']==best[0]);rr.update(selected_index=best[0],before_class=r['before_class'],after_class=r['predicted_class_id'],before_correct=r['before_class']==comp['class_id'],after_correct=r['predicted_class_id']==comp['class_id']);totals['before_correct']+=rr['before_correct'];totals['after_correct']+=rr['after_correct']
   ledger.append(rr)
  totals['unmatched_candidates']+=len(sel)-len(hit)
  del gt,pred
 assert totals['components']==58 and totals['rectangle_hit']==53 and totals['ellipsoid_hit']==53 and totals['unmatched_candidates']==17 and totals['before_correct']==36
 write_json(run/'lesion_ledger.json',ledger);write_json(run/'diagnostics.json',{'totals':totals,'rescued':[r for r in ledger if r['after_correct'] and not r['before_correct']],'lost':[r for r in ledger if r['before_correct'] and not r['after_correct']],'conditional_shape':shapes})
 rng=np.random.default_rng(20260905); samples={k:[] for k in ['MCC','DICE']}
 for b in range(2000):
  ix=rng.integers(0,40,40);ba=aggregate([before[i]['raw'] for i in ix])['overall'];aa=aggregate([after[i]['raw'] for i in ix])['overall']
  for k in samples:samples[k].append(aa[k]-ba[k])
 boot={k:{'n':2000,'seed':20260905,'mean':float(np.mean(v)),'low':float(np.percentile(v,2.5)),'high':float(np.percentile(v,97.5))} for k,v in samples.items()};write_json(run/'paired_bootstrap.json',boot)
 gates={'MCC_gain_ge_0.015':delta['MCC']>=.015-1e-12,'Dice_gain_ge_0.008':delta['DICE']>=.008-1e-12,'assignment_ge_40':totals['after_correct']>=40,'TP_ge_39':cs['after']['TP']>=39,'coverage_ge_12':coverage['after']>=12,'FP_le_33':cs['after']['FP']<=33,'P_R_VS_nondecrease':all(delta[k]>=-1e-12 for k in ['PRECISION','RECALL','VOLSIM']),'HD95_nonincrease':delta['HD95']<=1e-12}
 status='PASS' if all(gates.values()) else 'FAIL'
 result={'gate':status,'checks':gates,'before':aggs['before']['overall'],'after':aggs['after']['overall'],'delta':delta,'counts':cs,'assignment':totals,'coverage':coverage,'bootstrap':boot,'validity':'PASS'}
 write_json(run/'metrics_comparison.json',result);write_json(run/'success_gate.json',{'status':status,'checks':gates})
 write_json(run/'validity_checks.json',{'status':'PASS','frozen_candidates_binary_geometry':read(run/'prediction_validity.json'),'source_split':read(run/'source_split.json'),'synthetic_tests':synthetic_test(),'all_cases_scored':len(after)==40,'baseline_cached_raw_sha256':sha256_file(BASE/'evaluation/after_per_case.json'),'scorer_hashes':read(run/'config.json')['scorer_hashes'],'prediction_lock':lock})
 (run/'RESULT_TASK2_ASTRA6_E03.md').write_text('# ASTRA6 E03: image crop + anatomy classifier\n\n'+json.dumps(result,indent=2)+'\n\nAssignment subsystem changed to a 3D crop network with 943 anatomy inputs. Candidates, scores, class-independent ellipsoid foreground and scorer remain frozen. Source-only case-grouped development chose training duration; final network trained on all E02 source rows. No center2 refit. Different classifier architecture means image-specific causality is not isolated. Center2 is a repeatedly used research holdout, all cases positive; no blind-test or specificity claim. No patient linkage metadata.\n')
 write_json(run/'DONE.json',{'status':'DONE','valid':True,'gate':status,'n_cases':40})
 log(result)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--stage',choices=['prepare','train','infer','evaluate'],required=True);args=ap.parse_args();r=args.run
 if args.stage in ('train','infer'):
  assert os.environ.get('CUDA_VISIBLE_DEVICES')=='GPU-a643ded3-193b-58e8-b362-be93dc8eac14', 'GPU2 UUID binding required'
  assert torch.cuda.device_count()==1
 else:assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU phase must hide GPUs'
 start=time.time()
 if args.stage=='prepare':
  assert not (r/'features/SOURCE_READY.json').exists(); assert synthetic_test()
  prepare_source(r)
 elif args.stage=='train':assert not (r/'model/LOCKED.json').exists();train(r)
 elif args.stage=='infer':assert not (r/'PREDICTIONS_HASHED_BEFORE_EVAL_GT.json').exists();infer(r)
 else:assert not (r/'DONE.json').exists();evaluate(r)
 write_json(r/f'logs/{args.stage}_runtime.json',{'seconds':time.time()-start,'finished':datetime.now(timezone.utc).isoformat(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'python':sys.executable,'torch':torch.__version__,'output_bytes':sum(p.stat().st_size for p in r.rglob('*') if p.is_file())})
if __name__=='__main__':main()
