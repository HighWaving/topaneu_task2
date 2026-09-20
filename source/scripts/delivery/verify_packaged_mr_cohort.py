"""Check packaged native refinement against all 40 locked E17 predictions.

Inference receives raw image, predicted TA36 vessels and frozen detector boxes.
Reference predictions are read only after the inference subprocess exits.
"""
from pathlib import Path
import hashlib,json,os,subprocess,sys
import SimpleITK as sitk
import numpy as np

P=Path(__file__).resolve().parents[2]
B=P/'submission_models_20260909'
R=P/'artifacts/astra6_e17_MR_detector_crop_segmentation_20260909'
OUT=P/'artifacts/packaged_MR_E17_cohort_20260909'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ids=json.loads((P/'artifacts/astra6_e02_multiscale_vessel_signature_20260908T173916Z/eval_case_ids.json').read_text())
    cfg=json.loads((B/'configs/models.json').read_text())
    sha=hashlib.sha256((B/'models/segmentation_MR.pt').read_bytes()).hexdigest()
    assert sha==json.loads((R/'model/LOCKED.json').read_text())['model_sha256']
    result={}
    for cid in ids:
        dest=OUT/f'{cid}.mha'
        cmd=[sys.executable,'-m','scripts.delivery.refine_with_filter','--image',str(P.parent/f'data_topaneu26/images/{cid}_0000.nii.gz'),'--predicted-vessel',str(P/f'artifacts/ta36_mr_center2_output/{cid}.nii.gz'),'--boxes',str(P/f'artifacts/fold1_eval_center2_epoch60/{cid}_boxes.pkl'),'--classifier',str(B/'models/location_MR.joblib'),'--segmentation',str(B/'models/segmentation_MR.pt'),'--fp-filter',str(B/'models/filter_MR.pt'),'--fp-threshold',str(cfg['MR_fp_threshold']),'--modality','MR','--device','cuda','--output',str(dest)]
        with (OUT/f'{cid}.log').open('w') as log:
            subprocess.run(cmd,cwd=B/'source',check=True,stdout=log,stderr=subprocess.STDOUT)
        actual=sitk.ReadImage(str(dest));expected=sitk.ReadImage(str(R/f'predictions/mr_center2_k05/{cid}.nii.gz'))
        diff=int(np.count_nonzero(sitk.GetArrayViewFromImage(actual)!=sitk.GetArrayViewFromImage(expected)))
        geometry=actual.GetSize()==expected.GetSize() and all(np.allclose(getattr(actual,m)(),getattr(expected,m)(),atol=1e-5,rtol=0) for m in ['GetSpacing','GetOrigin','GetDirection'])
        assert diff==0 and geometry,(cid,diff,geometry)
        result[cid]={'different_voxels':diff,'geometry_matches':bool(geometry)}
        (OUT/'PROGRESS.json').write_text(json.dumps({'segmentation_sha256':sha,'cases':result,'complete':len(result)==len(ids)},indent=2)+'\n')
        print('VERIFIED',len(result),len(ids),cid,flush=True)
    assert len(result)==40

if __name__=='__main__':main()
