from pathlib import Path
import json,os
import numpy as np,joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from scripts.astra6_e09.prepare import RUN,BASE
from scripts.astra6_e01.e01_common import write_json,sha256_file
PARAMS=dict(max_depth=16,min_samples_leaf=2,max_features=.5,bootstrap=False,class_weight=None,n_jobs=4,random_state=20260905)
def fit(name,X,y):
 path=RUN/f'model/{name}.joblib';model=joblib.load(path) if path.exists() else ExtraTreesClassifier(n_estimators=64,warm_start=True,**PARAMS);classes,counts=np.unique(y,return_counts=True);cw=dict(zip(classes,1/np.sqrt(counts)));w=np.array([cw[c] for c in y]);w/=w.mean();start=len(model.estimators_)+64 if hasattr(model,'estimators_') else 64
 for n in range(start,513,64):
  model.set_params(n_estimators=n);model.fit(X,y,sample_weight=w);tmp=path.with_suffix('.tmp');joblib.dump(model,tmp,compress=3);os.replace(tmp,path);print(name,n,flush=True)
 return model

def main():
 assert (RUN/'features/READY.json').exists();data=np.load(RUN/'features/train.npz');X,y=data['X'],data['y'];rows=[json.loads(s) for s in (RUN/'features/train_records.jsonl').read_text().splitlines()];split=json.loads((RUN/'source_split.json').read_text());dev=set(split['source_development_cases']);tr=[i for i,r in enumerate(rows) if r['case_id'] not in dev];dv=[i for i,r in enumerate(rows) if r['case_id'] in dev and r['view']=='original' and r['sample_index']==0];assert not {rows[i]['case_id'] for i in tr}&dev and not {r['case_id'] for r in rows}&set(split['fixed_CT5']);model=fit('development',X[tr],y[tr]);base=joblib.load(BASE/'model/classifier.joblib');results={}
 for name,m in [('E02',base),('E09',model)]:
  pred=m.predict(X[dv]);results[name]={'correct':int((pred==y[dv]).sum()),'n':len(dv),'accuracy':accuracy_score(y[dv],pred),'balanced_accuracy':balanced_accuracy_score(y[dv],pred),'predictions':pred.tolist(),'labels':y[dv].tolist()}
 write_json(RUN/'evaluation/source_validation.json',results);print('source_validation',json.dumps(results),flush=True);model=fit('classifier',X,y);write_json(RUN/'model/LOCKED.json',{'model_sha256':sha256_file(RUN/'model/classifier.joblib'),'code_sha256':sha256_file(Path(__file__)),'config_sha256':sha256_file(RUN/'config.json'),'source':json.loads((RUN/'features/READY.json').read_text()),'CT5_used_for_fit':False,'MR40_used_for_fit':False,'use_for_CT_only':True,'n_trees':512});print('FULL_FORMAL_TRAINING_COMPLETE',flush=True)
if __name__=='__main__':main()
