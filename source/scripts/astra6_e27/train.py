"""Fixed512-tree source/full fits, gated by paired source performance."""
from pathlib import Path
import json,time,os,re
import numpy as np,joblib
from sklearn.ensemble import ExtraTreesClassifier
from scripts.astra6_e27.common import RUN,P
from scripts.astra6_e01.e01_common import write_json,sha256_file
OLD=P/'artifacts/astra6_e05_learned_location_splits_20260909'
def group(c):return re.sub(r'(_(?:mr|ct)_\d+)_\d+$',r'\1',c)
def fit(stage,X,y,w,seed):
 dest=RUN/f'model/{stage}.joblib';model=joblib.load(dest) if dest.exists() else ExtraTreesClassifier(n_estimators=64,warm_start=True,max_depth=16,min_samples_leaf=2,max_features=.5,bootstrap=False,class_weight=None,n_jobs=1,random_state=seed)
 model.feature_version='E27_junction_v1'
 start=len(model.estimators_)+64 if hasattr(model,'estimators_') else 64
 for n in range(start,513,64):
  t=time.monotonic();model.set_params(n_estimators=n);model.fit(X,y,sample_weight=w);tmp=dest.with_suffix('.tmp');joblib.dump(model,tmp,compress=3);os.replace(tmp,dest);print('E27_FOREST',stage,n,time.monotonic()-t,flush=True)
 assert len(model.estimators_)==512;return model

def main():
 assert (RUN/'features/READY.json').exists()
 if (RUN/'model/LOCKED.json').exists():return
 z=np.load(RUN/'features/train.npz');X,y,w=z['X'],z['y'],z['sample_weight'];records=[json.loads(s) for s in (RUN/'features/train_records.jsonl').read_text().splitlines()];old=json.loads((OLD/'source_split.json').read_text());dev={group(c) for c in old['development_cases']};tr=np.array([i for i,r in enumerate(records) if group(r['case_id']) not in dev]);dv=np.asarray(old['development_rows']);assert not {group(records[i]['case_id']) for i in tr}&dev;assert all(group(records[i]['case_id']) in dev for i in dv);assert X.shape==(4986,1235) and np.isfinite(X).all();write_json(RUN/'source_split.json',{'training_rows':tr.tolist(),'development_rows':dv.tolist(),'train_cases':sorted({records[i]['case_id'] for i in tr}),'development_cases':old['development_cases'],'same_baseline_case_partition':True,'no_MR40_CT5_fit':True})
 vals,cnt=np.unique(y[tr],return_counts=True);cw=dict(zip(vals,1/np.sqrt(cnt)));tw=np.array([cw[v] for v in y[tr]]);tw/=tw.mean();model=fit('development',X[tr],y[tr],tw,20260909);gp=model.predict(X[dv]);gc=int((gp==y[dv]).sum());ta=np.load(RUN/'features/source_TA.npz');tp=model.predict(ta['X']);tc=int((tp==ta['y']).sum());rows=json.loads((RUN/'features/source_TA_rows.json').read_text());assert all(r['case_id'] in old['development_cases'] for r in rows);baseline=joblib.load(OLD/'model/development_baseline.joblib');assert int((baseline.predict(X[dv,:943])==y[dv]).sum())==40;assert int((baseline.predict(ta['X'][:,:943])==ta['y']).sum())==25;passed=(gc>=40 and tc>=27) or (gc>=43 and tc>=25)
 result={'GT_correct':gc,'GT_n':len(dv),'GT_baseline_correct':40,'TA_detector_correct':tc,'TA_detector_n':len(tp),'TA_detector_baseline_correct':25,'GT_predictions':gp.tolist(),'GT_labels':y[dv].tolist(),'TA_rows':[{**r,'junction_prediction':int(v)} for r,v in zip(rows,tp)],'source_gate_passed':passed,'no_MR40_opened':True};write_json(RUN/'evaluation/source_validation.json',result);print('E27_SOURCE_RESULT',gc,tc,'gate',passed,flush=True)
 final=fit('classifier',X,y,w,20260905);write_json(RUN/'model/LOCKED.json',{'classifier_sha256':sha256_file(RUN/'model/classifier.joblib'),'development_sha256':sha256_file(RUN/'model/development.joblib'),'source_gate_passed':passed,'full512tree_development_and_final_complete':True,'feature_set':'E27_junction_v1','n_features':1235,'feature_code_sha256':sha256_file(Path(__file__).with_name('common.py')),'code_sha256':sha256_file(Path(__file__)),'config_sha256':sha256_file(RUN/'config.json'),'source_split_sha256':sha256_file(RUN/'source_split.json'),'source_manifest_sha256':sha256_file(RUN/'features/READY.json'),'source_input_disclosure':json.loads((RUN/'SOURCE_INPUT_DISCLOSURE.json').read_text()),'no_MR40_CT5_fit':True});print('E27_FULL_FORMAL_TRAINING_COMPLETE',flush=True)
if __name__=='__main__':main()
