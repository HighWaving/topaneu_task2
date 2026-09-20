import json, sys, platform
from pathlib import Path
import numpy as np, joblib, sklearn
from scipy.spatial import cKDTree
from scripts.astra6_e01.e01_common import *
from scripts.astra6_e02.run_e02 import extended_schema, multiscale, E01
from scripts.local_scoring_arena import aggregate
R=P/'artifacts/astra6_e02_multiscale_vessel_signature_20260908T173916Z'
O=R/'review_20260909'; O.mkdir(exist_ok=True)
def read(p): return json.loads(p.read_text())
def rows(p): return [json.loads(s) for s in p.read_text().splitlines()]
checks={}
def check(k,v):
 checks[k]=bool(v)
 if not v: raise AssertionError(k)
h=read(R/'hashes.json')
check('model_hash',sha256_file(R/'model/classifier.joblib')==h['model_sha256'])
check('prediction_hash',sha256_tree(R/'predictions')==h['predictions_tree_sha256']==(R/'PREDICTIONS_HASHED_BEFORE_EVAL_GT').read_text().strip())
check('config_hash',sha256_file(R/'config.json')==h['config_sha256'])
evidence=read(P/'planning/astra6_e02_20260908/evidence.json')
check('scorer_hashes',all(sha256_file(Path(p))==v for p,v in evidence['planner_source_hashes'].items()))
ids=read(E01/'eval_case_ids.json'); check('eval_ids',ids==read(R/'eval_case_ids.json') and len(set(ids))==40)
a=dict(np.load(E01/'features/train.npz')); b=dict(np.load(R/'features/train.npz'))
check('train_shapes',a['X'].shape==(4986,295) and b['X'].shape==(4986,943))
check('frozen_train_columns',np.array_equal(a['X'],b['X'][:,:295]))
check('frozen_labels_weights',all(np.array_equal(a[k],b[k]) for k in ('y','sample_weight')))
check('finite_float32',b['X'].dtype==np.float32 and np.isfinite(b['X']).all())
check('records_exact',sha256_file(E01/'features/train_records.jsonl')==sha256_file(R/'features/train_records.jsonl'))
records=rows(R/'features/train_records.jsonl'); schema=read(E01/'feature_schema.json'); names,pair,perm=extended_schema(schema)
check('schema',names==read(R/'feature_schema.json')['features'])
manifest=read(R/'source_manifest.json')['cases']; train=set(read(R/'train_case_ids.json'))
check('source_split',len(train)==218 and len(manifest)==329 and all('center2' not in c['case_id'] for c in manifest) and not (train&set(ids)) and all('center2' not in c for c in train) and all(r['case_id'] in train for r in records))
check('corrected_case_ids',not any('center1_mr_150' in c for c in train) and any('center1_mr_702' in c for c in train))
check('source_counts',sum(c['positive'] for c in manifest)==218 and len({(r['case_id'],r['source_class_id'],r['component_id']) for r in records})==277)
lpair={int(k):int(v) for k,v in schema['location_lr_pair'].items()}
check('record_labels',all(int(b['y'][i])==r['class_id']==(r['source_class_id'] if r['view']=='original' else lpair[r['source_class_id']]) for i,r in enumerate(records)))
groups={}
for i,r in enumerate(records): groups.setdefault((r['case_id'],r['source_class_id'],r['component_id'],r['sample_index']),{})[r['view']]=i
check('cached_mirror',all(np.array_equal(b['X'][g['original'],295:][perm],b['X'][g['mirror'],295:]) for g in groups.values()))
# Independent synthetic physical geometry: nontrivial affine, off-FOV queries, missing labels, LR direct reflection.
aff=np.array([[0,-2,0,12],[3,0,0,-7],[0,0,4,5],[0,0,0,1]],float)
pts={v:np.array([[v-18,2,3],[v-17,8,-4]],float) for v in range(1,37) if v%5}
ge={'trees':{v:cKDTree(pts[v]) if v in pts else None for v in range(1,37)}}
lo=np.array([1.,-8,2]); hi=lo+2; q=affine_world(aff,(lo+hi)/2)
f=multiscale(ge,aff,lo,hi); expected=[]
for v in range(1,37):
 for axis in range(3):
  for radius in (2,5,10):
   for sign in (-1,1):
    z=q+np.eye(3)[axis]*radius*sign
    expected.append(min(np.linalg.norm(pts[v]-z,axis=1).min(),50)/50 if v in pts else 1)
check('physical_queries_sentinel',np.allclose(f,expected,atol=1e-7))
s=np.array([-1,1,1]); mg={'trees':{v:cKDTree(pts[pair[v]]*s) if pair[v] in pts else None for v in range(1,37)}}
ma=np.eye(4); ma[:3,3]=q*s
check('direct_reflection',np.allclose(multiscale(mg,ma,np.zeros(3),np.zeros(3)),f[perm],atol=1e-7))
check('mirror_involution',np.array_equal(perm[perm],np.arange(648)))
model=joblib.load(R/'model/classifier.joblib'); old=joblib.load(E01/'model/classifier.joblib')
check('model_hyperparameters',model.get_params()==old.get_params())
er=rows(R/'candidate_predictions.jsonl'); before={(r['case_id'],r['original_index']):r for r in rows(E01/'candidate_predictions.jsonl')}
check('candidate_identity',len(er)==72 and set(before)=={(r['case_id'],r['original_index']) for r in er})
ex=np.load(R/'features/eval.npz')['X']; probs=np.zeros((72,52)); probs[:,model.classes_.astype(int)-1]=model.predict_proba(ex)
check('cached_feature_model_predictions',all(np.allclose(probs[i],r['probabilities'],atol=1e-12) and np.argmax(probs[i])+1==r['predicted_class_id'] for i,r in enumerate(er)))
for cid in ids:
 box,score,_=load_boxes(BOX_DIR/f'{cid}_boxes.pkl'); sel=select_candidates(box,score)
 x,aff,shape=load_nifti(R/f'predictions/mr_center2_k05/{cid}.nii.gz'); y,ya,ys=load_nifti(E01/f'predictions/mr_center2_k05/{cid}.nii.gz')
 check(f'{cid}:geometry',shape==ys and np.allclose(aff,ya) and x.dtype==np.uint8 and x.min()>=0 and x.max()<=52 and np.array_equal(x>0,y>0))
 painted=np.zeros(shape,np.uint8)
 for rank,(idx,sc,l,hi) in reversed(list(enumerate(sel))):
  r=next(r for r in er if r['case_id']==cid and r['original_index']==idx); e=before[(cid,idx)]
  check(f'{cid}:{idx}:selection',sc==r['score']==e['score'] and np.array_equal(l,r['low']) and np.array_equal(hi,r['high']) and np.array_equal(l,e['low']) and np.array_equal(hi,e['high']) and r['rank']==rank)
  fill_ellipsoid(painted,l,hi,r['predicted_class_id'])
 check(f'{cid}:paint',np.array_equal(x,painted))
 del x,y,painted
 print('verified',cid,flush=True)
pcb=read(R/'evaluation/before_per_case.json'); pca=read(R/'evaluation/after_per_case.json')
check('raw_before_exact',pcb==read(E01/'evaluation/experiment_per_case.json'))
check('paired_raw_order',[r['case_id'] for r in pcb]==ids==[r['case_id'] for r in pca])
shape_stats={}; aggs={}
for arm,pc in [('before',pcb),('after',pca)]:
 ag=aggregate([r['raw'] for r in pc]); aggs[arm]=ag
 check(arm+'_aggregate',ag['overall']==read(R/f'evaluation/{arm}_official.json')['overall'])
 vals=[r['raw'][f'DICE_{c}'] for r in pc for c in range(1,53) if r['raw'][f'TP_{c}']>0]
 shape_stats[arm]={'definition':'official TP case/class Dice','n':len(vals),'mean':float(np.mean(vals)),'median':float(np.median(vals))}
check('diagnostic_invariants',read(R/'diagnostics.json')['fixed_candidates']==read(R/'metrics_comparison.json')['assignment'])
support={str(c):{view:len({(r['case_id'],r['source_class_id'],r['component_id']) for r in records if r['class_id']==c and r['view']==view}) for view in ('original','mirror')} for c in range(1,53)}
write_json(O/'class_support.json',support)
write_json(O/'conditional_shape.json',shape_stats)
write_json(O/'validity_checks.json',{'status':'PASS','checks':checks,'limitations':['Retrospective verification cannot independently prove historical one-fit count or unrecorded code execution.','No patient linkage metadata: case/center isolation only.','Original code has incomplete diagnostic reporting; corrected conditional shape is in this review.','center2 is a repeatedly used research holdout, all 40 cases positive.']})
write_json(O/'provenance.json',{'python':sys.executable,'sklearn':sklearn.__version__,'numpy':np.__version__,'review_code_sha256':sha256_file(Path(__file__)),'baseline_raw_sha256':sha256_file(E01/'evaluation/experiment_per_case.json'),'source_files':{f:sha256_file(E01/f) for f in ['source_manifest.json','train_case_ids.json','features/train.npz','features/train_records.jsonl','feature_schema.json']}})
print(json.dumps({'validity':'PASS','shape':shape_stats},indent=2))
