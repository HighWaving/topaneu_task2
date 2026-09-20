"""Complete the two predeclared 512-tree MR anatomy models."""
import json,os
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from scripts.astra6_e19.prepare import RUN,F
from scripts.astra6_e01.e01_common import write_json,sha256_file

def fit(stage,X,y):
    path=RUN/f'model/{stage}.joblib';model=joblib.load(path) if path.exists() else ExtraTreesClassifier(n_estimators=64,warm_start=True,max_depth=12,min_samples_leaf=5,max_features=.5,bootstrap=False,class_weight='balanced',random_state=20260909,n_jobs=4)
    start=len(model.estimators_)+64 if hasattr(model,'estimators_') else 64
    for n in range(start,513,64):
        model.set_params(n_estimators=n);model.fit(X,y);temp=path.with_suffix('.tmp');joblib.dump(model,temp,compress=3);os.replace(temp,path);print('E19_TREES',stage,n,flush=True)
    model.feature_version='anatomical_fp_v1';model.required_image_filter_sha256=sha256_file(F/'model/final_last.pt');model.expected_modality='MR';joblib.dump(model,path,compress=3);return model

def main():
    assert (RUN/'features/SILVER_READY.json').exists();records=[json.loads(s) for s in (RUN/'features/records.jsonl').read_text().splitlines()];tr=np.array([i for i,r in enumerate(records) if not r['development']]);dv=np.array([i for i,r in enumerate(records) if r['development']]);assert not {records[i]['group'] for i in tr}&{records[i]['group'] for i in dv}
    for stage in ['development','final']:
        data=np.load(RUN/f'features/{stage}_silver.npz');ix=tr if stage=='development' else np.arange(len(records));fit(stage,data['X'][ix],data['y'][ix])
    write_json(RUN/'model/TRAINING_COMPLETE.json',{'development_trees':512,'final_trees':512,'development_sha256':sha256_file(RUN/'model/development.joblib'),'final_sha256':sha256_file(RUN/'model/final.joblib'),'source':json.loads((RUN/'features/SILVER_READY.json').read_text()),'configuration_sha256':sha256_file(RUN/'config.json'),'training_code_sha256':sha256_file(Path(__file__)),'no_MR40_or_CT5_fit':True});print('E19_FULL_FORMAL_TRAINING_COMPLETE',flush=True)

if __name__=='__main__':main()
