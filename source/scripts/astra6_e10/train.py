from pathlib import Path
import json,os
import numpy as np,joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from scripts.astra6_e10.prepare import RUN,BASE,SHAPE
from scripts.astra6_e10.features import FEATURE_VERSION
from scripts.astra6_e01.e01_common import sha256_file,write_json
PARAMS=dict(max_depth=16,min_samples_leaf=2,max_features=.5,bootstrap=False,class_weight=None,n_jobs=4,random_state=20260905)
def fit(name,X,y,contact):
 path=RUN/f'model/{name}.joblib';model=joblib.load(path) if path.exists() else ExtraTreesClassifier(n_estimators=64,warm_start=True,**PARAMS);vals,counts=np.unique(y,return_counts=True);cw=dict(zip(vals,1/np.sqrt(counts)));w=np.array([cw[c] for c in y]);w/=w.mean();start=len(model.estimators_)+64 if hasattr(model,'estimators_') else 64
 for n in range(start,513,64):
  model.set_params(n_estimators=n);model.fit(X,y,sample_weight=w);tmp=path.with_suffix('.tmp');joblib.dump(model,tmp,compress=3);os.replace(tmp,path);print(name,n,flush=True)
 if contact:model.feature_version=FEATURE_VERSION;model.required_segmenter_sha256=sha256_file(SHAPE/'model/final_last.pt')
 joblib.dump(model,path,compress=3);return model

def main():
 assert (RUN/'features/READY.json').exists();rec=[json.loads(s) for s in (RUN/'features/train_records.jsonl').read_text().splitlines()];split=json.loads((RUN/'source_split.json').read_text());dev=set(split['development_cases']);tr=[i for i,r in enumerate(rec) if r['case_id'] not in dev];dv=split['development_rows'];helper=json.loads((RUN/'source_segmenter/model/LOCKED.json').read_text());assert not {rec[i]['case_id'] for i in helper['train_rows']}&dev
 data=np.load(RUN/'features/development.npz');X,y=data['X'],data['y'];base=fit('development_baseline',X[tr,:943],y[tr],False);new=fit('development_new',X[tr],y[tr],True);results={}
 for name,model,xx in [('E02',base,X[dv,:943]),('E10',new,X[dv])]:
  pred=model.predict(xx);results[name]={'correct':int((pred==y[dv]).sum()),'n':len(dv),'accuracy':accuracy_score(y[dv],pred),'balanced_accuracy':balanced_accuracy_score(y[dv],pred),'predictions':pred.tolist(),'labels':y[dv].tolist()}
 write_json(RUN/'evaluation/source_validation.json',results);print('source_validation',json.dumps(results),flush=True);data=np.load(RUN/'features/final.npz');model=fit('classifier',data['X'],data['y'],True);write_json(RUN/'model/LOCKED.json',{'model_sha256':sha256_file(RUN/'model/classifier.joblib'),'code_sha256':sha256_file(Path(__file__)),'features_code_sha256':sha256_file(Path(__file__).with_name('features.py')),'config_sha256':sha256_file(RUN/'config.json'),'feature_version':FEATURE_VERSION,'source':json.loads((RUN/'features/READY.json').read_text()),'no_center2_fit':True,'n_trees':512});print('FULL_FORMAL_TRAINING_COMPLETE',flush=True)
if __name__=='__main__':main()
