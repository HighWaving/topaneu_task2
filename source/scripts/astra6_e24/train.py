"""One fixed CPU boosted-tree location experiment; source-only selection."""
from pathlib import Path
import json,hashlib,time,re,os
import numpy as np
import xgboost as xgb
P=Path(__file__).resolve().parents[2]
BASE=P/'artifacts/astra6_e02_multiscale_vessel_signature_20260908T173916Z'
OLD=P/'artifacts/astra6_e05_learned_location_splits_20260909'
AUDIT=P/'artifacts/source_location_detector_box_audit_20260909'
RUN=P/'artifacts/astra6_e24_boosted_location_20260909'
PARAMS=dict(objective='multi:softprob',num_class=52,eval_metric='mlogloss',tree_method='hist',device='cpu',nthread=4,max_depth=4,eta=.05,min_child_weight=1,subsample=.9,colsample_bytree=.8,reg_lambda=5,seed=20260909)
def write(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,path)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def group(c):return re.sub(r'(_(?:mr|ct)_\d+)_\d+$',r'\1',c)
class Save(xgb.callback.TrainingCallback):
 def __init__(self,stage):self.stage=stage;self.start=time.time()
 def after_iteration(self,model,epoch,evals_log):
  if (epoch+1)%10==0:
   path=RUN/f'model/{self.stage}_last.ubj';tmp=path.with_name(path.stem+'_tmp.ubj');model.save_model(tmp);os.replace(tmp,path)
   write(RUN/f'model/{self.stage}_progress.json',{'rounds':model.num_boosted_rounds(),'seconds':time.time()-self.start,'evaluation':evals_log})
  return False

def main():
 (RUN/'model').mkdir(parents=True,exist_ok=True)
 config={'experiment':'E24','hypothesis':'Stagewise regularized boosting may resolve vessel feature interactions missed by randomized independent trees; change classifier only, same943features and source cases.','params':PARAMS,'max_development_rounds':400,'early_stopping_rounds':50,'selection':'minimum source56-case-disjoint original-lesion logloss; final full-source model uses selected round count','checkpoint':'every10rounds UBJ; deterministic replay if interrupted development, full selected model saved','budget_hours':2,'source_gate':'GT56 correct>=43 and actualD31 correct>=25, OR actualD31 correct>=27 and GT56 correct>=40; no MR40 access on failure','baseline':{'source_GT_correct':40,'source_GT_n':56,'source_D_correct':25,'source_D_n':31},'no_MR40_CT5_fit':True,'not_new_blind_validation':'source split and MR40 have prior experiment exposure; no claim of untouched validation','xgboost_version':xgb.__version__}
 if (RUN/'config.json').exists():assert json.loads((RUN/'config.json').read_text())==config
 else:write(RUN/'config.json',config)
 if (RUN/'model/LOCKED.json').exists():print('E24_ALREADY_COMPLETE');return
 z=np.load(BASE/'features/train.npz');X=z['X'];y=z['y'].astype(int)-1;w=z['sample_weight'];rec=[json.loads(s) for s in (BASE/'features/train_records.jsonl').read_text().splitlines()];oldsplit=json.loads((OLD/'source_split.json').read_text());dev={group(c) for c in oldsplit['development_cases']};tr=np.array([i for i,r in enumerate(rec) if group(r['case_id']) not in dev]);dv=np.array(oldsplit['development_rows']);assert X.shape==(4986,943) and np.isfinite(X).all();assert all(group(rec[i]['case_id']) in dev for i in dv);assert not any('center2' in r['case_id'] for r in rec);assert not {group(rec[i]['case_id']) for i in tr}&dev
 write(RUN/'source_split.json',{'train_cases':sorted({rec[i]['case_id'] for i in tr}),'development_cases':oldsplit['development_cases'],'development_rows':dv.tolist(),'train_rows':tr.tolist(),'no_comparison_case_fit':True,'features_sha256':sha(BASE/'features/train.npz'),'records_sha256':sha(BASE/'features/train_records.jsonl')})
 vals,cnt=np.unique(y[tr],return_counts=True);cw=dict(zip(vals,1/np.sqrt(cnt)));tw=np.array([cw[t] for t in y[tr]]);tw/=tw.mean()
 dt=xgb.DMatrix(X[tr],label=y[tr],weight=tw,nthread=4);dd=xgb.DMatrix(X[dv],label=y[dv],nthread=4)
 chosen=RUN/'model/development_selected.ubj'
 if chosen.exists():
  model=xgb.Booster();model.load_model(chosen);rounds=model.num_boosted_rounds()
 else:
  model=xgb.train(PARAMS,dt,num_boost_round=400,evals=[(dt,'train'),(dd,'source')],early_stopping_rounds=50,callbacks=[Save('development')],verbose_eval=10)
  rounds=int(model.best_iteration)+1;model=model[:rounds];model.save_model(chosen)
 gp=model.predict(dd).argmax(1)+1;gc=int((gp==y[dv]+1).sum())
 rows=json.loads((AUDIT/'RESULT.json').read_text())['rows'];features=[]
 for cid in sorted({r['case_id'] for r in rows}):
  rr=[r for r in rows if r['case_id']==cid];xx=np.load(AUDIT/f'{cid}.npz')['X'];assert len(rr)==len(xx);features.extend(xx)
 assert rows==sorted(rows,key=lambda r:r['case_id']);dp=model.predict(xgb.DMatrix(np.asarray(features),nthread=4)).argmax(1)+1;dc=sum(int(p)==r['label'] for p,r in zip(dp,rows));passed=(gc>=43 and dc>=25) or (dc>=27 and gc>=40)
 write(RUN/'evaluation/SOURCE_RESULT.json',{'selected_rounds':rounds,'GT_correct':gc,'GT_n':len(dv),'GT_predictions':gp.tolist(),'GT_labels':(y[dv]+1).tolist(),'D_correct':dc,'D_n':len(rows),'D_rows':[{**r,'E24_prediction':int(v)} for r,v in zip(rows,dp)],'source_gate_passed':passed,'no_MR40_opened':True})
 final=xgb.train(PARAMS,xgb.DMatrix(X,label=y,weight=w,nthread=4),num_boost_round=rounds,callbacks=[Save('final')],verbose_eval=False);final.save_model(RUN/'model/classifier.ubj')
 write(RUN/'model/LOCKED.json',{'full_formal_training_complete':True,'selected_and_final_rounds':rounds,'model_sha256':sha(RUN/'model/classifier.ubj'),'code_sha256':sha(Path(__file__)),'config_sha256':sha(RUN/'config.json'),'source_split_sha256':sha(RUN/'source_split.json'),'source_gate_passed':passed,'next_action':'MR40 C-only paired evaluation with frozen E17S/E14F' if passed else 'Reject at source gate; retain baseline; continue E23 OOF data work'})
 print('E24_FULL_FORMAL_TRAINING_COMPLETE',rounds,gc,dc,'source_gate',passed,flush=True)
if __name__=='__main__':main()
