"""Single data-addition contrast; no parameter search or validation-case fitting."""
import json, os, re, time
from pathlib import Path
import joblib, numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from scripts.astra6_e01.e01_common import write_json, sha256_file
P=Path(__file__).resolve().parents[2]
RUN=P/'artifacts/astra6_e30_expanded_CT_supervision_for_MR_20260910'
OLD=P/'artifacts/astra6_e02_multiscale_vessel_signature_20260908T173916Z'
EXP=P/'artifacts/astra6_e09_CT_expanded_location_20260909'
SRC=P/'artifacts/astra6_e29_staged_anatomy_location_20260910'
def group(c): return re.sub(r'(_(?:mr|ct)_\d+)_\d+$',r'\1',c)
def main():
 if (RUN/'model/LOCKED.json').exists():return
 cfg=json.loads((RUN/'config.json').read_text());dev=set(cfg['development_cases']);devg={group(c) for c in dev}
 rows=[json.loads(s) for s in (EXP/'features/train_records.jsonl').read_text().splitlines()]
 extra=set(json.loads((EXP/'source_split.json').read_text())['additional_eligible_CT_cases'])
 assert len(extra)==42
 z=np.load(EXP/'features/train.npz');X,y=z['X'],z['y'];old=np.load(OLD/'features/train.npz')
 before=[i for i,r in enumerate(rows) if r['case_id'] not in extra]
 assert np.array_equal(X[before],old['X']) and np.array_equal(y[before],old['y'])
 counts=dict(zip(*np.unique(old['y'],return_counts=True)));w=np.array([1/np.sqrt(counts[c]) for c in old['y']]);w/=w.mean();assert np.allclose(w,old['sample_weight'],atol=1e-12)
 src=[json.loads(s) for s in (SRC/'features/records.jsonl').read_text().splitlines()]
 val=[src[i] for i in cfg['source_validation_rows']];assert len(val)==26 and all(r['case_id'] in dev for r in val)
 vx=np.array([r['baseline_geometry_features'] for r in val],np.float32);vy=np.array([r['class'] for r in val])
 forbidden=set(json.loads((OLD/'eval_case_ids.json').read_text()))|set(json.loads((EXP/'source_split.json').read_text())['fixed_CT5'])
 assert not {group(r['case_id']) for r in rows}&{group(c) for c in forbidden}
 results={};lineage={};(RUN/'model').mkdir(exist_ok=True)
 for arm in ['baseline','expanded']:
  ix=[i for i,r in enumerate(rows) if group(r['case_id']) not in devg and (arm=='expanded' or r['case_id'] not in extra)]
  fitcases=sorted({rows[i]['case_id'] for i in ix});assert not {group(c) for c in fitcases}&devg
  path=RUN/f'model/{arm}_source.joblib';start=time.monotonic()
  m=joblib.load(path) if path.exists() else ExtraTreesClassifier(n_estimators=64,warm_start=True,max_depth=16,min_samples_leaf=2,max_features=.5,bootstrap=False,class_weight=None,n_jobs=2,random_state=20260905)
  counts=dict(zip(*np.unique(y[ix],return_counts=True)));w=np.array([1/np.sqrt(counts[c]) for c in y[ix]]);w/=w.mean()
  for n in range(len(m.estimators_)+64 if hasattr(m,'estimators_') else 64,513,64):
   m.set_params(n_estimators=n);m.fit(X[ix],y[ix],sample_weight=w);tmp=path.with_suffix('.tmp');joblib.dump(m,tmp,compress=3);os.replace(tmp,path);print(arm,n,round(time.monotonic()-start,1),flush=True)
  pred=m.predict(vx);results[arm]={'correct':int((pred==vy).sum()),'n':len(val),'predictions':pred.tolist(),'labels':vy.tolist(),'cases':[r['case_id'] for r in val],'model_sha256':sha256_file(path),'fit_cases':fitcases,'n_training_rows':len(ix)}
  lineage[arm]={'actual_fit_cases':fitcases,'selection_cases':sorted(dev),'selection_excluded_from_gradient_fit':True}
 sourcegate=results['expanded']['correct']>=results['baseline']['correct']+2
 write_json(RUN/'SOURCE_RESULT.json',{'arms':results,'source_gate':sourcegate,'expanded_rescued':sum(a!=y and b==y for a,b,y in zip(results['baseline']['predictions'],results['expanded']['predictions'],vy)),'expanded_lost':sum(a==y and b!=y for a,b,y in zip(results['baseline']['predictions'],results['expanded']['predictions'],vy))})
 final=EXP/'model/classifier.joblib';lock=json.loads((EXP/'model/LOCKED.json').read_text());assert sha256_file(final)==lock['model_sha256'];m=joblib.load(final);assert len(m.estimators_)==512
 target=RUN/'model/classifier.joblib'
 if not target.exists():target.symlink_to(final)
 write_json(RUN/'CASE_LINEAGE.json',{'source':lineage,'final_fit_cases':sorted({r['case_id'] for r in rows}),'additional_CT_cases':sorted(extra),'source_upstream_audit':str(SRC/'CASE_LINEAGE.json'),'known_limits':cfg['known_limits'],'source_training_geometry':'GT lesion boxes plus organizer predicted vessel; no detector/segmenter dependency','source_evaluation_geometry':'E29 fixed OOF detector operating boxes; D/S own-case exclusion in referenced audit','MR40_inference_vessels':'existing TA36 predicted vessel features'})
 write_json(RUN/'model/LOCKED.json',{'full_formal_training_complete':True,'trees_each':512,'model_sha256':lock['model_sha256'],'reused_final_model':str(final),'config_sha256':sha256_file(RUN/'config.json'),'source_gate':sourcegate,'MR40_CT5_gradient_fit':False,'old_feature_identity_and_weight_formula_verified':True})
 print('E30 SOURCE COMPLETE',results['baseline']['correct'],results['expanded']['correct'],flush=True)
if __name__=='__main__':main()
