"""Fixed controlled S continuation; every epoch resumes full optimizer/RNG."""
from scripts.astra6_e04.run_e04 import *
from scripts.astra6_e26.prepare import RUN,BASE
import math

def native_scores(probs,targets):
 scores=[]
 for prob,t in zip(probs,targets):
  fg=largest(map_coordinates(prob,t['coords'].reshape(3,-1),order=1,mode='constant',cval=0,prefilter=False).reshape(t['truth'].shape)>=.5)
  if not fg.any():fg=t['ellipse']
  scores.append(float(2*(fg&t['truth']).sum()/max(1,fg.sum()+int(t['GT_voxels']))))
 return scores

def fit(run,stage,trainix,devix=None,epochs=100,augmented=False):
 seed();model=Segmenter().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
 dl=DataLoader(SegData(run,trainix),batch_size=32,shuffle=True,num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(20260909));dv=DataLoader(SegData(run,devix),batch_size=32,num_workers=0) if devix is not None else None
 targets=[dict(np.load(run/f'native_validation/{j:02d}.npz')) for j in range(31)] if dv is not None else None
 history=[];best=-1.;bestep=0;support=torch.from_numpy(SUPPORT[None,None]).cuda()
 start_epoch=1;resume=run/f'model/{stage}_resume.pt'
 if resume.exists():
  state=torch.load(resume,weights_only=False,map_location='cpu');model.load_state_dict(state['state_dict']);opt.load_state_dict(state['optimizer']);history=state['history'];best=state['best'];bestep=state['bestep'];start_epoch=state['epoch']+1
  torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);dl.generator.set_state(state['loader_rng'])
 elif stage=='development' and (run/'recovery_20260909/development_best.pt').exists():
  state=torch.load(run/'recovery_20260909/development_best.pt',weights_only=False,map_location='cpu');model.load_state_dict(state['state_dict']);best=state['dev_dice'];bestep=state['epoch'];start_epoch=bestep+1
  history=[h for h in read(run/'recovery_20260909/development_history.json') if h['epoch']<=bestep]
  write_json(run/'model/RECOVERY.json',{'resume_epoch':bestep,'optimizer_reset':True,'reason':'interrupted legacy run saved weights only; archived original history and weights','not_exact_resume':True})
 if start_epoch>epochs or (dv is not None and start_epoch>30 and start_epoch-1-bestep>=20):
  last=run/f'model/{stage}_last.pt';tmp=last.with_suffix('.tmp');torch.save({'state_dict':model.state_dict(),'epoch':start_epoch-1},tmp);os.replace(tmp,last)
  return bestep if dv is not None else model
 for ep in range(start_epoch,epochs+1):
  epoch_start=time.time()
  model.train();total=0.;n=0
  for x,y in dl:
   x=x.cuda();y=y.cuda()
   if augmented and ep>13:
    gamma=torch.empty((len(x),1,1,1,1),device=x.device).uniform_(.8,1.25);image=x[:,0:1].clamp(0,1).pow(gamma);image=(image+.01*torch.randn_like(image)).clamp(0,1);x=torch.cat([image,x[:,1:2]],dim=1)
    for axis in [2,3,4]:
     if torch.rand((),device=x.device)<.5:x=x.flip(axis);y=y.flip(axis)
   opt.zero_grad(set_to_none=True);z=model(x);p=z.sigmoid();dice=1-(2*(p*y).sum((1,2,3,4))+1)/(p.sum((1,2,3,4))+y.sum((1,2,3,4))+1);loss=.5*nn.functional.binary_cross_entropy_with_logits(z,y)+.5*dice.mean();assert torch.isfinite(loss), 'nonfinite loss';loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=loss.item()*len(x);n+=len(x)
  record={'epoch':ep,'train_loss':total/n,'train_seconds':time.time()-epoch_start,'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
  if dv is not None:
   model.eval();probs=[]
   with torch.no_grad():
    for x,y in dv:probs.extend(model(x.cuda()).sigmoid().cpu().numpy()[:,0])
   scores=native_scores(probs,targets);score=float(np.mean(scores));record.update(dev_native_Dice=score,dev_components=len(scores))
   if score>best:
    best=score;bestep=ep;torch.save({'state_dict':model.state_dict(),'epoch':ep,'dev_native_Dice':score},run/'model/development_best.pt')
  history.append(record);write_json(run/f'model/{stage}_history.json',history);log(f'{stage} {record}')
  state={'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'epoch':ep,'history':history,'best':best,'bestep':bestep,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'loader_rng':dl.generator.get_state()}
  tmp=resume.with_suffix('.tmp');torch.save(state,tmp);os.replace(tmp,resume)
  if dv is not None and ep>=30 and ep-bestep>=20:break
 torch.save({'state_dict':model.state_dict(),'epoch':ep},run/f'model/{stage}_last.pt')
 return bestep if dv is not None else model

def main():
 torch.set_num_threads(2);assert (RUN/'native_validation/READY.json').exists()
 if (RUN/'model/LOCKED.json').exists():return
 free,total=torch.cuda.mem_get_info()
 while free<2.5*2**30:
  print('E26_WAITING_FOR_RESERVED_HEADROOM',free,flush=True);time.sleep(30);free,total=torch.cuda.mem_get_info()
 split=json.loads((BASE/'source_split.json').read_text());records=[json.loads(s) for s in (BASE/'features/train_records.jsonl').read_text().splitlines()];assert not {records[i]['case_id'] for i in split['train_rows']}&set(split['development_cases']);targets=[dict(np.load(RUN/f'native_validation/{j:02d}.npz')) for j in range(31)];net=Segmenter().cuda();state=torch.load(BASE/'model/development_last.pt',map_location='cpu',weights_only=False);assert state['epoch']==13;net.load_state_dict(state['state_dict']);x,y=next(iter(DataLoader(SegData(RUN,split['train_rows']),batch_size=32)));z=net(x.cuda());loss=nn.functional.binary_cross_entropy_with_logits(z,y.cuda());assert torch.isfinite(loss);loss.backward();peak=torch.cuda.max_memory_allocated();assert peak<1.3*2**30,('S exceeded shared-GPU reservation',peak);write_json(RUN/'model/PREFLIGHT.json',{'batch':32,'finite_forward_backward':True,'allocated_peak_bytes':peak,'global_free_bytes_before_S':free,'global_free_bytes_after_S':torch.cuda.mem_get_info()[0],'GPU_total_bytes':total,'no_optimizer_update':True});del x,y,z,loss
 net.zero_grad(set_to_none=True);net.eval();data=SegData(RUN,split['development_detector_rows']);xx=torch.stack([data[i][0] for i in range(31)])
 with torch.no_grad():probs=net(xx.cuda()).sigmoid().cpu().numpy()[:,0]
 baseline_scores=native_scores(probs,targets);baseline=float(np.mean(baseline_scores));write_json(RUN/'evaluation/BASELINE_NATIVE.json',{'E17_development13_native_Dice':baseline,'scores':baseline_scores,'checkpoint_sha256':sha256_file(BASE/'model/development_last.pt'),'decoder':'identical native interpolation/largestcomponent/empty-ellipse behavior to deployedE17; fullGTcomponent denominator'});del net,xx,probs;torch.cuda.empty_cache();results={}
 for arm in ['control','augmented']:
  out=RUN/arm;(out/'model').mkdir(parents=True,exist_ok=True)
  for name,src in [('features',BASE/'features'),('native_validation',RUN/'native_validation')]:
   if not (out/name).exists():(out/name).symlink_to(src,target_is_directory=True)
  marker=out/'model/INITIALIZED.json'
  if not marker.exists():
   dev=torch.load(BASE/'model/development_resume.pt',map_location='cpu',weights_only=False);assert dev['epoch']==13;dev['best']=baseline;dev['bestep']=13;torch.save(dev,out/'model/development_resume.pt');torch.save({'state_dict':dev['state_dict'],'epoch':13,'dev_native_Dice':baseline},out/'model/development_best.pt');shutil.copy2(BASE/'model/final_resume.pt',out/'model/final_resume.pt');write_json(marker,{'development_parent_sha256':sha256_file(BASE/'model/development_resume.pt'),'full_parent_sha256':sha256_file(BASE/'model/final_resume.pt'),'selection_candidates_start_at_epoch13':True,'baseline_native_Dice':baseline});del dev
  complete=out/'model/COMPLETE.json'
  if complete.exists():results[arm]=json.loads(complete.read_text());continue
  epoch=fit(out,'development',split['train_rows'],split['development_detector_rows'],epochs=100,augmented=arm=='augmented');best=torch.load(out/'model/development_best.pt',map_location='cpu',weights_only=False);assert epoch>=13 and best['epoch']==epoch;score=float(best['dev_native_Dice']);del best;torch.cuda.empty_cache();fit(out,'final',list(range(len(records))),epochs=epoch,augmented=arm=='augmented');torch.cuda.empty_cache();history=json.loads((out/'model/development_history.json').read_text());results[arm]={'selected_epoch':epoch,'development_completed_epoch':history[-1]['epoch'],'native_Dice':score,'model_sha256':sha256_file(out/'model/final_last.pt'),'full_source_total_updates':epoch*math.ceil(len(records)/32),'both_parent13_and_continuation_complete':True};write_json(complete,results[arm]);print('E26_ARM_FULL_TRAINING_COMPLETE',arm,json.dumps(results[arm]),flush=True)
 selected='augmented' if results['augmented']['native_Dice']>=results['control']['native_Dice']+.005 else 'control';passed=results[selected]['native_Dice']>=baseline+.005;shutil.copy2(RUN/selected/'model/final_last.pt',RUN/'model/final_last.pt');write_json(RUN/'evaluation/source_validation.json',{'baseline_E17_native_Dice':baseline,'arms':results,'selected_arm':selected,'source_gate_passed':passed,'comparison_case_fit':False});frozen=P/'artifacts/astra6_e14_MR_filter_seed_stability_20260909/candidate_predictions.jsonl';write_json(RUN/'model/LOCKED.json',{'model_sha256':sha256_file(RUN/'model/final_last.pt'),'frozen_filter_assignments_sha256':sha256_file(frozen),'source_gate_passed':passed,'selected_arm':selected,'epochs':results[selected]['selected_epoch'],'both_arms_full_formal_training_complete':True,'source':json.loads((BASE/'features/SOURCE_READY.json').read_text()),'config_sha256':sha256_file(RUN/'config.json'),'code_sha256':sha256_file(Path(__file__)),'no_MR40_CT5_fit':True});print('E26_FULL_FORMAL_TRAINING_COMPLETE',selected,'source_gate',passed,flush=True)
if __name__=='__main__':main()
