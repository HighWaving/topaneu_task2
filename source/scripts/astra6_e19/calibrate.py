"""Use only actual TA36 predictions for the fixed source deployment calibration."""
import json
from pathlib import Path
import numpy as np,joblib
from scripts.astra6_e19.prepare import RUN,F,CACHE,P,DATA
from scripts.astra6_e01.e01_common import compute_feature,load_nifti_geometry,sha256_file,write_json
from scripts.astra6_e02.run_e02 import multiscale
from scripts.delivery.geometry import vessel_geometry_fast

def main():
    assert (RUN/'model/TRAINING_COMPLETE.json').exists();split=json.loads((RUN/'source_split.json').read_text());records=[json.loads(s) for s in (RUN/'features/records.jsonl').read_text().splitlines()];indices=split['anatomical_calibration_rows'];pp=np.load(RUN/'features/development_image_probability.npy');groups={}
    for i in indices:assert records[i]['development'] and records[i]['augmentation']==0;groups.setdefault(records[i]['case_id'],[]).append(i)
    features={};provenance={};models=P/'submission_models_20260909/models/ta36';expected_models={str(f.relative_to(models)):sha256_file(f) for f in sorted(models.rglob('*')) if f.is_file()}
    for cid,ix in sorted(groups.items()):
        case=CACHE/cid;vessel=case/'predicted_vessel.nii.gz';lock=json.loads((case/'PROVENANCE.json').read_text());image=DATA/f'images/{cid}_0000.nii.gz';assert lock['image_sha256']==sha256_file(image) and lock['GT_not_input'] and lock['model_hashes']==expected_models;provenance[cid]={'vessel_sha256':sha256_file(vessel),'input_and_models':lock};aff,shape=load_nifti_geometry(image);geom=vessel_geometry_fast(vessel,shape,aff)
        for i in ix:
            r=records[i];lo,hi=np.array(r['low']),np.array(r['high']);features[i]=np.concatenate([compute_feature(geom,aff,lo,hi,'MR'),multiscale(geom,aff,lo,hi),[pp[i]]]).astype(np.float32)
        print('E19_ACTUAL_TA36_CALIBRATION',cid,flush=True)
    X=np.asarray([features[i] for i in indices],np.float32);y=np.array([records[i]['y'] for i in indices]);assert X.shape==(52,944) and y.sum()==43 and np.isfinite(X).all();model=joblib.load(RUN/'model/development.joblib');q=model.predict_proba(X)[:,list(model.classes_).index(1)];threshold=float(np.nextafter(q[y==1].min(),0.));keep=q>=threshold;neg_before=int((y==0).sum());neg_after=int((keep&(y==0)).sum());gate=int((keep&(y==1)).sum())==43 and neg_before-neg_after>=2
    np.savez_compressed(RUN/'features/actual_TA36_calibration.npz',X=X,y=y,rows=indices)
    source={'source_cases':len(groups),'source_positives':43,'retained_positives':int((keep&(y==1)).sum()),'source_negatives':neg_before,'retained_negatives':neg_after,'source_gate_passed':gate,'actual_TA36_used':True,'no_MR40_or_CT5_access':True};write_json(RUN/'evaluation/source_validation.json',source);write_json(RUN/'model/THRESHOLD.json',{'threshold':threshold,'parent_image_threshold':split['parent_image_threshold'],'selection':'minimum source-development anatomical probability among43 parent-retained positives, evaluated with actualTA36; one fixed threshold','source_only':True,**source});write_json(RUN/'model/development_predictions.json',{'rows':indices,'anatomical_probability':q.tolist(),'image_probability':pp[indices].tolist(),'y':y.tolist(),'keep':keep.tolist()});write_json(RUN/'features/TA36_CALIBRATION_PROVENANCE.json',provenance)
    write_json(RUN/'model/LOCKED.json',{'model_sha256':sha256_file(RUN/'model/final.joblib'),'threshold_sha256':sha256_file(RUN/'model/THRESHOLD.json'),'config_sha256':sha256_file(RUN/'config.json'),'calibration_features_sha256':sha256_file(RUN/'features/actual_TA36_calibration.npz'),'training_complete':json.loads((RUN/'model/TRAINING_COMPLETE.json').read_text()),'source_gate_passed':gate,'no_MR40_or_CT5_fit':True});print('E19_SOURCE_CALIBRATION_RESULT',source,flush=True)

if __name__=='__main__':main()
