"""E05: learned tree thresholds vs E02 random thresholds; frozen features and E04 shape."""
import json,time,re,os,shutil
from pathlib import Path
import numpy as np,joblib
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,log_loss
from scripts.astra6_e01.e01_common import P,sha256_file,write_json
BASE=P/'artifacts/astra6_e02_multiscale_vessel_signature_20260908T173916Z'
SHAPE=P/'artifacts/astra6_e04_crop_segmentation_20260909T031146Z'
RUN=P/'artifacts/astra6_e05_learned_location_splits_20260909'
PARAMS=dict(max_depth=16,min_samples_leaf=2,max_features=.5,bootstrap=False,class_weight=None,n_jobs=4,random_state=20260909)
def group(c):return re.sub(r'(_ct_\d+)_\d+$',r'\1',c)
def fit(name,kind,X,y,w):
 path=RUN/f'model/{name}.joblib';model=joblib.load(path) if path.exists() else kind(n_estimators=64,warm_start=True,**PARAMS)
 start=len(model.estimators_)+64 if hasattr(model,'estimators_') else 64
 for n in range(start,513,64):
  t=time.time();model.set_params(n_estimators=n);model.fit(X,y,sample_weight=w);tmp=path.with_suffix('.tmp');joblib.dump(model,tmp,compress=3);os.replace(tmp,path)
  print(json.dumps({'stage':name,'trees':n,'seconds':time.time()-t}),flush=True)
 return model

def train():
 for sub in ['model','logs','frozen_assignment','predictions/mr_center2_k05','evaluation']:(RUN/sub).mkdir(parents=True,exist_ok=True)
 config={'experiment':'E05','hypothesis':'learned impurity-minimizing thresholds separate nearby vascular classes better than randomized thresholds','change':'ExtraTrees to RandomForest; bootstrap remains false; same512 trees and all other parameters','parameters':PARAMS,'source':str(BASE),'shape':str(SHAPE),'max_trees':512,'checkpoint_every_trees':64,'budget_hours':2,'selection':'source group-disjoint diagnostics; final official MR40 paired comparison; no holdout fitting','container':'user owns; out of scope'}
 write_json(RUN/'config.json',config)
 data=np.load(BASE/'features/train.npz');X,y,w=data['X'],data['y'],data['sample_weight'];rec=[json.loads(s) for s in (BASE/'features/train_records.jsonl').read_text().splitlines()];assert X.shape==(4986,943) and np.isfinite(X).all() and not any('center2' in r['case_id'] for r in rec)
 old=json.loads((SHAPE/'source_split.json').read_text());dg={group(c) for c in old['development_cases']};tr=np.array([i for i,r in enumerate(rec) if group(r['case_id']) not in dg]);dv=np.array([i for i,r in enumerate(rec) if group(r['case_id']) in dg and r['view']=='original' and r['sample_index']==0]);assert not {group(rec[i]['case_id']) for i in tr}&{group(rec[i]['case_id']) for i in dv}
 write_json(RUN/'source_split.json',{'train_cases':sorted({rec[i]['case_id'] for i in tr}),'development_cases':sorted({rec[i]['case_id'] for i in dv}),'development_rows':dv.tolist(),'group_rule':'CT scan suffixes grouped; no available patient linkage beyond IDs','features_sha256':sha256_file(BASE/'features/train.npz'),'records_sha256':sha256_file(BASE/'features/train_records.jsonl')})
 # Recompute class balancing from training partition only for source diagnostics.
 vals,cnt=np.unique(y[tr],return_counts=True);cw=dict(zip(vals,1/np.sqrt(cnt)));tw=np.array([cw[z] for z in y[tr]]);tw/=tw.mean()
 results={}
 for name,kind in [('development_baseline',ExtraTreesClassifier),('development_new',RandomForestClassifier)]:
  model=fit(name,kind,X[tr],y[tr],tw);proba=model.predict_proba(X[dv]);pred=model.classes_[proba.argmax(1)];results[name]={'accuracy':accuracy_score(y[dv],pred),'balanced_accuracy':balanced_accuracy_score(y[dv],pred),'correct':int((y[dv]==pred).sum()),'n':len(dv),'predictions':pred.tolist(),'labels':y[dv].tolist()};write_json(RUN/'evaluation/source_validation.json',results)
  print(json.dumps({'validation':name,**results[name]}),flush=True)
 model=fit('classifier',RandomForestClassifier,X,y,w)
 write_json(RUN/'model/CLASSIFIER_LOCKED.json',{'sha256':sha256_file(RUN/'model/classifier.joblib'),'code_sha256':sha256_file(Path(__file__)),'config_sha256':sha256_file(RUN/'config.json'),'source_features_sha256':sha256_file(BASE/'features/train.npz'),'n_trees':len(model.estimators_),'n_source_cases':218,'center2_used_for_fit':False,'source_diagnostics':results})
 # Lock assignments without opening evaluation GT. Shape is the unchanged E04 network.
 fx=np.load(BASE/'features/eval.npz')['X'];rr=[json.loads(s) for s in (BASE/'candidate_predictions.jsonl').read_text().splitlines()];pred=model.predict(fx);assert len(rr)==len(pred)==72
 for r,cl in zip(rr,pred):r['E02_class']=r['predicted_class_id'];r['predicted_class_id']=int(cl)
 (RUN/'frozen_assignment/candidate_predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in rr)+'\n');shutil.copy2(BASE/'eval_case_ids.json',RUN/'frozen_assignment/eval_case_ids.json')
 link=RUN/'frozen_assignment/predictions'
 if not link.exists():link.symlink_to(SHAPE/'predictions',target_is_directory=True)
 shutil.copy2(SHAPE/'model/final_last.pt',RUN/'model/final_last.pt');write_json(RUN/'model/LOCKED.json',{'model_sha256':sha256_file(RUN/'model/final_last.pt'),'classifier_sha256':sha256_file(RUN/'model/classifier.joblib'),'frozen_assignment_sha256':sha256_file(RUN/'frozen_assignment/candidate_predictions.jsonl'),'shape_unchanged':'E04','time':time.time()})
 print('FORMAL_TRAINING_COMPLETE',flush=True)
if __name__=='__main__':train()
