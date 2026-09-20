from pathlib import Path
import json,time
import numpy as np
from scripts.astra6_e01.e01_common import P,DATA,TA36_DIR,compute_feature,load_nifti_geometry,write_json
from scripts.astra6_e02.run_e02 import multiscale
from scripts.delivery.geometry import vessel_geometry_fast
B=P/'artifacts/astra6_e02_multiscale_vessel_signature_20260908T173916Z';records=json.loads((B/'features/eval_records.json').read_text());expected=np.load(B/'features/eval.npz')['X'];assert len(records)==len(expected)==72;current=None;actual=[];t=time.monotonic()
for r in records:
 cid=r['case_id']
 if cid!=current:
  aff,shape=load_nifti_geometry(DATA/f'images/{cid}_0000.nii.gz');geom=vessel_geometry_fast(TA36_DIR/f'{cid}.nii.gz',shape,aff);current=cid
 feat=np.concatenate([compute_feature(geom,aff,np.asarray(r['low']),np.asarray(r['high']),'MR'),multiscale(geom,aff,r['low'],r['high'])]).astype(np.float32);actual.append(feat)
actual=np.array(actual);result={'n_candidates':72,'n_features':943,'bit_exact_float32':np.array_equal(actual,expected),'max_abs_error':float(abs(actual-expected).max()),'seconds':time.monotonic()-t};write_json(P/'reports/geometry_equivalence_20260909.json',result);print(result);assert result['bit_exact_float32']
