"""Verify file-path E26 refinement against locked MR40 prediction, no GT input."""
from pathlib import Path
import json,os,subprocess,time
import numpy as np,SimpleITK as sitk
from scripts.astra6_e26.prepare import RUN,P
from scripts.astra6_e01.e01_common import sha256_file,write_json

def main():
 case='topaneu_center2_mr_002';raw=P/'artifacts/r2_raw_cohort_parity_20260909/MR_work';dest=RUN/'runtime';dest.mkdir(exist_ok=True);env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='GPU-f7491bbb-3972-1264-755a-63dec96b4a7d');output=dest/'MR002.mha';py=P.parent/'conda_envs/nnunet_v100/bin/python';b=P/'submission_models_20260909';threshold=json.loads((b/'configs/models.json').read_text())['MR_fp_threshold'];cmd=[str(py),'-m','scripts.delivery.refine_with_filter','--image',str(raw/'case_0000.nii.gz'),'--predicted-vessel',str(raw/'predicted_vessel.nii.gz'),'--boxes',str(raw/'boxes/case_boxes.pkl'),'--classifier',str(b/'models/location_MR.joblib'),'--segmentation',str(RUN/'model/final_last.pt'),'--fp-filter',str(b/'models/filter_MR.pt'),'--fp-threshold',str(threshold),'--modality','MR','--device','cuda','--output',str(output)]
 t=time.monotonic()
 with (dest/'refinement.log').open('w') as f:subprocess.run(cmd,cwd=P,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 a=sitk.ReadImage(str(output));ref=sitk.ReadImage(str(RUN/f'predictions/mr_center2_k05/{case}.nii.gz'));exact=np.array_equal(sitk.GetArrayViewFromImage(a),sitk.GetArrayViewFromImage(ref));geometry=all(np.allclose(getattr(a,m)(),getattr(ref,m)(),atol=1e-5,rtol=0) for m in ['GetSize','GetSpacing','GetOrigin','GetDirection']);result={'case':case,'exact_native_mask':bool(exact),'geometry_verified':bool(geometry),'output_uint8':a.GetPixelID()==sitk.sitkUInt8,'seconds':time.monotonic()-t,'runtime':json.loads(output.with_suffix('.json').read_text()),'segmentation_sha256':sha256_file(RUN/'model/final_last.pt'),'inputs':'Original raw image, raw-inference D and TA36 outputs from completed r2 pipeline; noGT','limitation':'Refinement-only timing. Upstream D/TA36 reused; not a fresh full runtime or T4/container validation.'};write_json(dest/'REFINEMENT_VERIFIED.json',result);assert exact and geometry and result['output_uint8'];print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
