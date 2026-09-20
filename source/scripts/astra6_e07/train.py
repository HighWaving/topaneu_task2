import json,time,os
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from scripts.astra6_e07.model import *
from scripts.astra6_e01.e01_common import P,DATA,sha256_file,write_json
from scripts.astra6_e05.run_e05 import BASE,SHAPE,PARAMS
RUN=P/'artifacts/astra6_e07_hierarchical_location_20260909';PREV=P/'artifacts/astra6_e05_learned_location_splits_20260909'

def fit_node(path,X,y,w):
 model=joblib.load(path) if path.exists() else ExtraTreesClassifier(n_estimators=64,warm_start=True,**PARAMS);start=len(model.estimators_)+64 if hasattr(model,'estimators_') else 64
 for n in range(start,513,64):
  model.set_params(n_estimators=n);model.fit(X,y,sample_weight=w);tmp=path.with_suffix('.tmp');joblib.dump(model,tmp,compress=3);os.replace(tmp,path)
 print('node_done',str(path),len(X),flush=True);return model

def fit(stage,X,y,w):
 d=RUN/f'model/{stage}';d.mkdir(parents=True,exist_ok=True);g=np.array([CLASS_TO_TERRITORY[int(c)] for c in y]);parent=fit_node(d/'parent.joblib',X,g,w);children={}
 for group in np.unique(g):
  ix=g==group;children[int(group)]=fit_node(d/f'territory{group}.joblib',X[ix],y[ix],w[ix])
 model=HierarchicalExtraTrees(parent,children,np.unique(y));joblib.dump(model,RUN/f'model/{stage}.joblib',compress=3);return model

def main():
 labels=json.loads((DATA/'location_mapping.json').read_text())['labels'];assert all(CLASS_TO_TERRITORY[v]==4 for k,v in labels.items() if '-4.' in k or k.startswith('4.'));assert all(CLASS_TO_TERRITORY[v]==5 for k,v in labels.items() if '-5.' in k)
 for s in ['model','evaluation','logs','frozen_assignment','predictions/mr_center2_k05']:(RUN/s).mkdir(parents=True,exist_ok=True)
 write_json(RUN/'config.json',{'experiment':'E07','hypothesis':'factorized territory then branch classification reduces interference between anatomically distant classes while retaining ExtraTrees regularization','territories':TERRITORIES,'official_location_mapping_sha256':sha256_file(DATA/'location_mapping.json'),'parameters':PARAMS,'trees_per_node':512,'checkpoint_every_trees':64,'budget_hours':2,'features_and_weights':'sameE02; source-controlled split sameE05','seed_note':'source-controlled E05 baseline and E07 use20260909; historical finalE02 uses20260905, no seed search','adoption':'current official MR40 Pareto improvement required; no heldout fit; E04 remains shape baseline','not_combined_with_E06':True})
 data=np.load(BASE/'features/train.npz');X,y,w=data['X'],data['y'],data['sample_weight'];rr=[json.loads(s) for s in (BASE/'features/train_records.jsonl').read_text().splitlines()];split=json.loads((PREV/'source_split.json').read_text());dev=set(split['development_cases']);tr=np.array([i for i,r in enumerate(rr) if r['case_id'] not in dev]);dv=np.array(split['development_rows']);assert not any('center2' in r['case_id'] for r in rr);write_json(RUN/'source_split.json',split)
 vals,cnt=np.unique(y[tr],return_counts=True);cw=dict(zip(vals,1/np.sqrt(cnt)));tw=np.array([cw[z] for z in y[tr]]);tw/=tw.mean();model=fit('development',X[tr],y[tr],tw);pred=model.predict(X[dv]);baseline=json.loads((PREV/'evaluation/source_validation.json').read_text())['development_baseline'];result={'baseline':baseline,'new':{'accuracy':accuracy_score(y[dv],pred),'balanced_accuracy':balanced_accuracy_score(y[dv],pred),'correct':int((pred==y[dv]).sum()),'n':len(dv),'predictions':pred.tolist(),'labels':y[dv].tolist()}};write_json(RUN/'evaluation/source_validation.json',result);print('source_validation',json.dumps(result['new']),flush=True)
 model=fit('classifier',X,y,w);write_json(RUN/'model/CLASSIFIER_LOCKED.json',{'sha256':sha256_file(RUN/'model/classifier.joblib'),'training_features_sha256':sha256_file(BASE/'features/train.npz'),'code_sha256':sha256_file(Path(__file__)),'architecture_sha256':sha256_file(Path(__file__).with_name('model.py')),'config_sha256':sha256_file(RUN/'config.json'),'source_cases':218,'center2_used':False})
 candidates=[json.loads(s) for s in (BASE/'candidate_predictions.jsonl').read_text().splitlines()];pred=model.predict(np.load(BASE/'features/eval.npz')['X'])
 for r,c in zip(candidates,pred):r['E02_class']=r['predicted_class_id'];r['predicted_class_id']=int(c)
 (RUN/'frozen_assignment/candidate_predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in candidates)+'\n');import shutil
 shutil.copy2(BASE/'eval_case_ids.json',RUN/'frozen_assignment/eval_case_ids.json');link=RUN/'frozen_assignment/predictions'
 if not link.exists():link.symlink_to(SHAPE/'predictions',target_is_directory=True)
 shutil.copy2(SHAPE/'model/final_last.pt',RUN/'model/final_last.pt');write_json(RUN/'model/LOCKED.json',{'model_sha256':sha256_file(RUN/'model/final_last.pt'),'classifier_sha256':sha256_file(RUN/'model/classifier.joblib'),'frozen_assignment_sha256':sha256_file(RUN/'frozen_assignment/candidate_predictions.jsonl'),'shape_unchanged':'E04'});print('FULL_FORMAL_TRAINING_COMPLETE',flush=True)
if __name__=='__main__':main()
