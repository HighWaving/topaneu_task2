"""Generate source-only TA36 predictions and audit location feature shift.

Per-case output/provenance permits restart without rerunning completed cases.
No comparison-cohort images or labels are read by this experiment.
"""
import argparse,json,os,subprocess,sys,time,hashlib
from pathlib import Path
import joblib,nibabel as nib,numpy as np
from nibabel.processing import resample_from_to
from scripts.astra6_e01.e01_common import P,DATA,load_nifti_geometry,compute_feature,write_json
from scripts.astra6_e02.run_e02 import multiscale,extended_schema,E01
from scripts.delivery.geometry import vessel_geometry_fast

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',required=True);a=parser.parse_args()
    source=P/'artifacts/source_location_detector_box_audit_20260909';audit=json.loads((source/'RESULT.json').read_text());groups={}
    for r in audit['rows']:groups.setdefault(r['case_id'],[]).append(r)
    out=P/'artifacts/source_MR_TA36_distribution_audit_20260909';out.mkdir(exist_ok=True)
    config={'hypothesis':'Organizer-provided source silver vessels may differ from deployment TA36 and shift MR location features','cases':sorted(groups),'source_lesions':31,'training_trigger':'TA36 actual-detector-box correct count at least2 lower than silver-vessel correct count on identical source lesions','budget_hours':4,'source_development_only':True,'GT_vessel_used':False,'no_MR40_or_CT5_access':True,'checkpoint_strategy':'Persist native prediction, features, input/model hashes and cumulative report per case; resume by matching provenance','next_if_trigger':'Prepare actual TA36 training features for remaining eligible source MR positive cases and fit fixed512-tree C with same source split; evaluate only after locking full model'}
    write_json(out/'config.json',config)
    app=P/'submission_models_20260909/source/dependencies/ta36_app';models=P/'submission_models_20260909/models/ta36'
    model_hashes={str(f.relative_to(models)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(models.rglob('*')) if f.is_file()}
    sys.path.insert(0,str(app/'ta36'));from reorient_nii import reorient_nii
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOPANEU_MODEL_ROOT=str(models),PYTHONPATH=os.pathsep.join([str(P/'vendor/delivery_runtime_deps'),str(app/'vendor')]))
    rec=[json.loads(s) for s in (P/'artifacts/astra6_e17_MR_detector_crop_segmentation_20260909/features/train_records.jsonl').read_text().splitlines()]
    classifier=joblib.load(P/'artifacts/astra6_e05_learned_location_splits_20260909/model/development_baseline.joblib');schema=json.loads((E01/'feature_schema.json').read_text());_,vpair,_=extended_schema(schema)
    rows=[];start=time.monotonic()
    for cid,lesions in sorted(groups.items()):
        assert 'center2' not in cid and '_mr_' in cid
        image=DATA/f'images/{cid}_0000.nii.gz';case=out/cid;case.mkdir(exist_ok=True);native=case/'predicted_vessel.nii.gz';lock=case/'PROVENANCE.json';expected={'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),'model_hashes':model_hashes,'GT_not_input':True}
        if native.exists() and lock.exists():assert json.loads(lock.read_text())==expected
        else:
            inp=case/'input';dest=case/'output';inp.mkdir(exist_ok=True);dest.mkdir(exist_ok=True);im=nib.load(image);reorient_nii(im,targ_aff='LPS').to_filename(inp/f'{cid}_0000.nii.gz')
            cmd=[sys.executable,str(P/'scripts/delivery/ta36_cached_preprocessing.py'),str(app/'ta36/run_inference.py'),'--input',str(inp),'--output',str(dest),'--suffix','_0000.nii.gz','--sequential','--n_infer_workers','1','--n_pre_post_workers','1']
            with (case/'inference.log').open('w') as log:subprocess.run(cmd,cwd=app,env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
            v=resample_from_to(nib.load(dest/f'{cid}.nii.gz'),im,order=0);nib.Nifti1Image(np.asarray(v.dataobj,dtype=np.uint8),im.affine).to_filename(native);write_json(lock,expected)
        aff,shape=load_nifti_geometry(image);geom=vessel_geometry_fast(native,shape,aff);features=[]
        for row in lesions:
            r=rec[row['detector_row']];lo,hi=np.array(r['low']),np.array(r['high']);features.append(np.concatenate([compute_feature(geom,aff,lo,hi,'MR',False,vpair),multiscale(geom,aff,lo,hi)]))
        x=np.asarray(features,np.float32);assert x.shape==(len(lesions),943) and np.isfinite(x).all();np.savez_compressed(case/'features.npz',X=x)
        for row,pred in zip(lesions,classifier.predict(x)):rows.append({**row,'TA36_detector_prediction':int(pred)})
        silver=sum(r['label']==r['detector_box_prediction'] for r in rows);ta=sum(r['label']==r['TA36_detector_prediction'] for r in rows)
        write_json(out/'PROGRESS.json',{'cases_complete':len({r['case_id'] for r in rows}),'complete':len(rows)==31,'n':len(rows),'silver_correct':silver,'TA36_correct':ta,'training_trigger_passed':len(rows)==31 and silver-ta>=2,'elapsed_seconds':time.monotonic()-start,'rows':rows})
        print('SOURCE_MR_TA36',len({r['case_id'] for r in rows}),len(groups),cid,'accuracy',ta,len(rows),flush=True)
    write_json(out/'RESULT.json',json.loads((out/'PROGRESS.json').read_text()))

if __name__=='__main__':main()
