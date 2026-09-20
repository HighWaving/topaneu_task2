"""Fixed E17 D/S/F, with expanded CT-supervised classifier; no GT read."""
import json,time
import joblib,nibabel as nib,numpy as np
from scripts.astra6_e30.train import P,RUN,OLD
from scripts.astra6_e28.infer import draw,F,S
from scripts.astra6_e01.e01_common import DATA,sha256_file,sha256_tree,write_json,nifti_output

def main():
 if (RUN/'PREDICTIONS_HASHED_BEFORE_EVAL_GT.json').exists():return
 lock=json.loads((RUN/'model/LOCKED.json').read_text());assert sha256_file(RUN/'model/classifier.joblib')==lock['model_sha256']
 model=joblib.load(RUN/'model/classifier.joblib');model.n_jobs=2
 control=joblib.load(OLD/'model/classifier.joblib');control.n_jobs=2
 records=[json.loads(s) for s in (S/'candidate_predictions.jsonl').read_text().splitlines()]
 er=json.loads((OLD/'features/eval_records.json').read_text());X=np.load(OLD/'features/eval.npz')['X'];assert len(er)==len(X)
 mapping={(r['case_id'],r['original_index']):i for i,r in enumerate(er)};assert len(mapping)==len(er)
 oldpred=control.predict(X);newprob=model.predict_proba(X);newpred=model.classes_[newprob.argmax(1)]
 proof=json.loads((F/'FIXED_UPSTREAM_PARITY.json').read_text());assert proof['complete']
 ids=json.loads((OLD/'eval_case_ids.json').read_text());out=[];checks={};start=time.monotonic()
 for n,cid in enumerate(ids,1):
  template=DATA/f'images/{cid}_0000.nii.gz';img=nib.load(str(template));mask=np.zeros(img.shape,np.uint8);oldmask=np.zeros(img.shape,np.uint8)
  cache=F/f'fixed_upstream/{cid}.npz';assert sha256_file(cache)==proof['cases'][cid]['probabilities_sha256'];z=np.load(cache);probs={int(i):q for i,q in zip(z['original_index'],z['probability'])}
  for old in [r for r in records if r['case_id']==cid]:
   i=mapping[(cid,old['original_index'])];erow=er[i]
   assert np.allclose(old['low'],erow['low'],atol=1e-6) and np.allclose(old['high'],erow['high'],atol=1e-6)
   assert int(oldpred[i])==old['predicted_class_id'],(cid,i,'baseline classifier mismatch')
   row=dict(old);cl=int(newpred[i]);pp=np.zeros(52);pp[model.classes_.astype(int)-1]=newprob[i];row.update(baseline_class_id=old['predicted_class_id'],predicted_class_id=cl,classifier='E30_reused_C09',probabilities=pp.tolist())
   if old['filter_keep']:
    lo,hi=np.array(old['low']),np.array(old['high']);pr=probs[old['original_index']]
    draw(oldmask,pr,lo,hi,old['predicted_class_id']);draw(mask,pr,lo,hi,cl)
   out.append(row)
  ref=nib.load(str(S/f'predictions/mr_center2_k05/{cid}.nii.gz'));arr=np.asanyarray(ref.dataobj)
  assert np.allclose(img.affine,ref.affine,atol=1e-4) and np.array_equal(oldmask,arr),cid
  assert np.array_equal(mask>0,arr>0),cid
  nifti_output(mask,template,RUN/f'predictions/mr_center2_k05/{cid}.nii.gz');checks[cid]={'baseline_native_exact':True,'foreground_union_exact':True,'geometry_features_and_C02_class_exact':True};print('E30 inference',n,40,cid,flush=True)
 (RUN/'candidate_predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in out)+'\n')
 write_json(RUN/'RESEARCH_START.json',{'baseline':'E17','baseline_candidates_sha256':sha256_file(S/'candidate_predictions.jsonl')})
 write_json(RUN/'FIXED_UPSTREAM_PARITY.json',{'baseline':'E17','cases':checks,'complete':len(checks)==40})
 write_json(RUN/'PREDICTIONS_HASHED_BEFORE_EVAL_GT.json',{'prediction_tree_sha256':sha256_tree(RUN/'predictions'),'model_sha256':lock['model_sha256'],'baseline':'E17','n_cases':40,'seconds':time.monotonic()-start,'GT_not_read':True})
if __name__=='__main__':main()
