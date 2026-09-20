"""Prefetch the same source audit in reverse on the second allocated GPU.

Work in separate scratch directories. Atomically rename a completed directory
only if the primary audit has not started that case. Never overwrite its work.
"""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import nibabel as nib,numpy as np
from nibabel.processing import resample_from_to
from scripts.astra6_e01.e01_common import P,DATA,write_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',required=True);parser.add_argument('--case-list',type=Path);parser.add_argument('--part',type=int,default=0);parser.add_argument('--parts',type=int,default=1);parser.add_argument('--progress-name',default='PREFETCH_PROGRESS.json');a=parser.parse_args();assert 0<=a.part<a.parts
    out=P/'artifacts/source_MR_TA36_distribution_audit_20260909';config=json.loads((a.case_list or out/'config.json').read_text());scratch=out/'prefetch_scratch';scratch.mkdir(exist_ok=True);case_ids=config['cases'][a.part::a.parts]
    app=P/'submission_models_20260909/source/dependencies/ta36_app';models=P/'submission_models_20260909/models/ta36'
    hashes={str(f.relative_to(models)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(models.rglob('*')) if f.is_file()}
    sys.path.insert(0,str(app/'ta36'));from reorient_nii import reorient_nii
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOPANEU_MODEL_ROOT=str(models),PYTHONPATH=os.pathsep.join([str(P/'vendor/delivery_runtime_deps'),str(app/'vendor')]))
    completed=[];skipped=[];start=time.monotonic()
    for cid in reversed(case_ids):
        assert '_mr_' in cid and 'center2' not in cid
        target=out/cid
        if target.exists():skipped.append(cid);continue
        case=scratch/cid;case.mkdir(exist_ok=True);image=DATA/f'images/{cid}_0000.nii.gz';expected={'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),'model_hashes':hashes,'GT_not_input':True};lock=case/'PROVENANCE.json';native=case/'predicted_vessel.nii.gz'
        if lock.exists() and native.exists():assert json.loads(lock.read_text())==expected
        else:
            inp=case/'input';dest=case/'output';inp.mkdir(exist_ok=True);dest.mkdir(exist_ok=True);im=nib.load(image);reorient_nii(im,targ_aff='LPS').to_filename(inp/f'{cid}_0000.nii.gz')
            cmd=[sys.executable,str(P/'scripts/delivery/ta36_cached_preprocessing.py'),str(app/'ta36/run_inference.py'),'--input',str(inp),'--output',str(dest),'--suffix','_0000.nii.gz','--sequential','--n_infer_workers','1','--n_pre_post_workers','1']
            with (case/'inference.log').open('w') as log:subprocess.run(cmd,cwd=app,env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
            v=resample_from_to(nib.load(dest/f'{cid}.nii.gz'),im,order=0);nib.Nifti1Image(np.asarray(v.dataobj,dtype=np.uint8),im.affine).to_filename(native);write_json(lock,expected)
        if not target.exists():
            try:case.rename(target);completed.append(cid)
            except OSError:
                if not target.exists():raise
                skipped.append(cid)
        else:skipped.append(cid)
        write_json(out/a.progress_name,{'completed_native_cases':completed,'skipped_primary_cases':skipped,'gpu':a.gpu,'elapsed_seconds':time.monotonic()-start,'complete':False})
        print('SOURCE_TA36_PREFETCH',len(completed),cid,flush=True)
    write_json(out/a.progress_name,{'completed_native_cases':completed,'skipped_primary_cases':skipped,'gpu':a.gpu,'elapsed_seconds':time.monotonic()-start,'complete':True})

if __name__=='__main__':main()
