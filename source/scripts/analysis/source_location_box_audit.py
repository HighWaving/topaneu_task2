"""Source-only paired location audit; never accesses MR40 or CT5.

Test whether a source-held-out classifier degrades when the same 31 lesions
use actual detector boxes rather than GT boxes. No new model is fitted here.
"""
import json
from pathlib import Path
import joblib
import numpy as np
from scripts.astra6_e01.e01_common import P,DATA,load_nifti_geometry,compute_feature,write_json
from scripts.astra6_e02.run_e02 import multiscale,extended_schema,E01
from scripts.delivery.geometry import vessel_geometry_fast

def main():
    base=P/'artifacts/astra6_e02_multiscale_vessel_signature_20260908T173916Z'
    seg=P/'artifacts/astra6_e17_MR_detector_crop_segmentation_20260909'
    out=P/'artifacts/source_location_detector_box_audit_20260909';out.mkdir(exist_ok=True)
    config={'hypothesis':'Location features may degrade on displaced detector boxes compared with GT boxes for the same source lesions','predeclared_training_trigger':'At least2 fewer correct predictions among31 same source lesions on detector boxes','no_MR40_or_CT5_access':True,'limitations':'Source detector has source training exposure; source vessels are organizer TopBrain predicted silver, not inference TA36; this is a feature-input diagnostic only'}
    write_json(out/'config.json',config)
    paired=json.loads((seg/'evaluation/SOURCE_PRECONDITION.json').read_text())
    records=[json.loads(s) for s in (seg/'features/train_records.jsonl').read_text().splitlines()]
    old=np.load(base/'features/train.npz');oldrecords=[json.loads(s) for s in (base/'features/train_records.jsonl').read_text().splitlines()]
    model=joblib.load(P/'artifacts/astra6_e05_learned_location_splits_20260909/model/development_baseline.joblib')
    dev=set(json.loads((P/'artifacts/astra6_e05_learned_location_splits_20260909/source_split.json').read_text())['development_cases'])
    schema=json.loads((E01/'feature_schema.json').read_text());_,vpair,_=extended_schema(schema)
    groups={}
    for gi,di in zip(paired['paired_GT_rows'],paired['paired_detector_rows']):
        r=records[di];g=oldrecords[gi];assert r['case_id']==g['case_id'] and r['source_class_id']==g['class_id'] and r['case_id'] in dev and 'center2' not in r['case_id'];groups.setdefault(r['case_id'],[]).append((gi,di))
    rows=[]
    for n,(cid,items) in enumerate(sorted(groups.items()),1):
        cache=out/f'{cid}.npz'
        if cache.exists():features=np.load(cache)['X']
        else:
            aff,shape=load_nifti_geometry(DATA/f'images/{cid}_0000.nii.gz');geom=vessel_geometry_fast(DATA/f'vessel_masks/{cid}.nii.gz',shape,aff);features=[]
            for gi,di in items:
                r=records[di];lo,hi=np.array(r['low']),np.array(r['high']);features.append(np.concatenate([compute_feature(geom,aff,lo,hi,'MR',False,vpair),multiscale(geom,aff,lo,hi)]))
            features=np.asarray(features,np.float32);np.savez_compressed(cache,X=features)
        assert features.shape==(len(items),943) and np.isfinite(features).all()
        dp=model.predict(features);gp=model.predict(old['X'][[gi for gi,di in items]])
        for (gi,di),d,g in zip(items,dp,gp):rows.append({'case_id':cid,'GT_row':gi,'detector_row':di,'label':oldrecords[gi]['class_id'],'GT_box_prediction':int(g),'detector_box_prediction':int(d)})
        print('SOURCE_LOCATION_AUDIT',n,len(groups),cid,flush=True)
    gc=sum(r['label']==r['GT_box_prediction'] for r in rows);dc=sum(r['label']==r['detector_box_prediction'] for r in rows)
    write_json(out/'RESULT.json',{'n':len(rows),'GT_box_correct':gc,'detector_box_correct':dc,'correct_count_gap':gc-dc,'training_trigger_passed':gc-dc>=2,'rows':rows,**config})
    print('SOURCE_LOCATION_RESULT',len(rows),gc,dc,'trigger',gc-dc>=2,flush=True)

if __name__=='__main__':main()
