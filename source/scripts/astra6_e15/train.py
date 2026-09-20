from pathlib import Path
import json
import numpy as np,joblib
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from scripts.astra6_e15.prepare import RUN,PARENT
from scripts.astra6_e09 import train as training
from scripts.astra6_e01.e01_common import write_json,sha256_file

def main():
 assert (RUN/'features/READY.json').exists();training.RUN=RUN;data=np.load(RUN/'features/train.npz');X,y=data['X'],data['y'];rec=[json.loads(s) for s in (RUN/'features/train_records.jsonl').read_text().splitlines()];split=json.loads((RUN/'source_split.json').read_text());dev=set(split['source_development_cases']);tr=[i for i,r in enumerate(rec) if r['case_id'] not in dev];dv=[i for i,r in enumerate(rec) if r['case_id'] in dev and r['view']=='original' and r['sample_index']==0];assert len(dv)==11 and not {rec[i]['case_id'] for i in tr}&dev and not {r['case_id'] for r in rec}&set(split['fixed_CT5']);new=training.fit('development',X[tr],y[tr]);base=joblib.load(PARENT/'model/development.joblib');results={}
 for name,m in [('E09_same_source_split_on_TA36',base),('E15',new)]:
  pred=m.predict(X[dv]);results[name]={'correct':int((pred==y[dv]).sum()),'n':len(dv),'accuracy':accuracy_score(y[dv],pred),'balanced_accuracy':balanced_accuracy_score(y[dv],pred),'predictions':pred.tolist(),'labels':y[dv].tolist()}
 write_json(RUN/'evaluation/source_validation.json',results);print('source_validation',json.dumps(results),flush=True);training.fit('classifier',X,y);write_json(RUN/'model/LOCKED.json',{'model_sha256':sha256_file(RUN/'model/classifier.joblib'),'source':json.loads((RUN/'features/READY.json').read_text()),'config_sha256':sha256_file(RUN/'config.json'),'code_sha256':sha256_file(Path(__file__)),'shared_training_code_sha256':sha256_file(Path(training.__file__)),'CT5_used_for_fit':False,'use_for_CT_only':True,'n_trees':512});print('FULL_FORMAL_TRAINING_COMPLETE',flush=True)
if __name__=='__main__':main()
