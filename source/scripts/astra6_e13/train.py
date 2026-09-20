from pathlib import Path
import os,json
import numpy as np,joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from scripts.astra6_e13.prepare import RUN,PARENT,P
from scripts.astra6_e01.e01_common import load_boxes,select_candidates,sha256_file,write_json

def fit(name,X,y):
 dest=RUN/f'model/{name}.joblib';model=joblib.load(dest) if dest.exists() else ExtraTreesClassifier(n_estimators=64,warm_start=True,max_depth=12,min_samples_leaf=5,max_features=.5,bootstrap=False,class_weight='balanced',random_state=20260909,n_jobs=4)
 start=len(model.estimators_)+64 if hasattr(model,'estimators_') else 64
 for n in range(start,513,64):
  model.set_params(n_estimators=n);model.fit(X,y);tmp=dest.with_suffix('.tmp');joblib.dump(model,tmp,compress=3);os.replace(tmp,dest);print(name,n,flush=True)
 model.feature_version='anatomical_fp_v1';model.required_image_filter_sha256=sha256_file(PARENT/'model/final_last.pt');joblib.dump(model,dest,compress=3);return model

def main():
 assert (RUN/'features/READY.json').exists();rec=[json.loads(s) for s in (RUN/'features/records.jsonl').read_text().splitlines()];tr=[i for i,r in enumerate(rec) if not r['development']];dv=[i for i,r in enumerate(rec) if r['development']];assert not {rec[i]['group'] for i in tr}&{rec[i]['group'] for i in dv};data=np.load(RUN/'features/development.npz');X,y=data['X'],data['y'];m=fit('development',X[tr],y[tr]);q=m.predict_proba(X[dv])[:,list(m.classes_).index(1)];p=X[dv,-1];image_threshold=json.loads((PARENT/'model/THRESHOLD.json').read_text())['threshold'];pools={}
 for cid in {rec[i]['case_id'] for i in dv}:
  bx,sc,_=load_boxes(P/f'artifacts/ct_fold2_all109_boxes/{cid}_boxes.pkl');pools[cid]={r[0] for r in select_candidates(bx,sc)}
 deployed=np.array([rec[i]['augmentation']==0 and rec[i]['original_index'] in pools[rec[i]['case_id']] for i in dv]);parent=deployed&(p>=image_threshold);pos=parent&(y[dv]==1);assert pos.sum()>0;threshold=float(np.nextafter(q[pos].min(),0.));keep=parent&(q>=threshold)
 source={'parent_deployed_positives':int(pos.sum()),'retained_positives':int((keep&(y[dv]==1)).sum()),'parent_deployed_negatives':int((parent&(y[dv]==0)).sum()),'retained_negatives':int((keep&(y[dv]==0)).sum()),'all_pool_AUROC':float(roc_auc_score(y[dv],q))};write_json(RUN/'evaluation/source_validation.json',source);write_json(RUN/'model/development_predictions.json',{'rows':dv,'anatomical_probability':q.tolist(),'image_probability':p.tolist(),'y':y[dv].tolist(),'deployed':deployed.tolist()});write_json(RUN/'model/THRESHOLD.json',{'threshold':threshold,'parent_image_threshold':image_threshold,'selection':'minimum anatomical probability among positive original source deployment candidates retained by parent E11','source_only':True,'CT5_used':False,**source});print('source_validation',json.dumps(source),flush=True)
 data=np.load(RUN/'features/final.npz');fit('final',data['X'],data['y']);write_json(RUN/'model/LOCKED.json',{'model_sha256':sha256_file(RUN/'model/final.joblib'),'threshold_sha256':sha256_file(RUN/'model/THRESHOLD.json'),'code_sha256':sha256_file(Path(__file__)),'config_sha256':sha256_file(RUN/'config.json'),'source':json.loads((RUN/'features/READY.json').read_text()),'n_trees':512,'no_CT5_fit':True});print('FULL_FORMAL_TRAINING_COMPLETE',flush=True)
if __name__=='__main__':main()
