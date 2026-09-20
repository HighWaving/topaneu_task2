"""Cache native target coordinates and already-held-out learned fallback, source only."""
import json
import numpy as np,nibabel as nib
from scripts.astra6_e31.common import P,RUN
from scripts.astra6_e01.e01_common import DATA,write_json,sha256_file
from scripts.astra6_e04.run_e04 import component_records_fast
F=P/'artifacts/astra6_e25_oof_shape_fp_filter_20260909'
def main():
 from scripts.astra6_e31.finalize_training_rows import main as finalize
 finalize()
 from scripts.astra6_e31.lineage import main as lineage
 lineage()
 if (RUN/'native_validation/READY.json').exists():return
 (RUN/'native_validation').mkdir(exist_ok=True);s=json.loads((RUN/'source_split.json').read_text());records=[json.loads(x) for x in (RUN/'features/records.jsonl').read_text().splitlines()];checks=[]
 for i in s['validation_rows']:
  r=records[i];cid=r['case_id'];gt=nib.load(str(DATA/f'location_masks/{cid}.nii.gz'));comps=component_records_fast(np.asanyarray(gt.dataobj));coords=comps[r['component_index']]['coords'];assert len(coords)==r['GT_voxels']
  fr=json.loads((F/f'features/cases/{cid}.json').read_text());old=next(x for x in fr if x['original_index']==r['candidate_index']);assert old['fold']==1 and old['matched_components']==[r['component_index']]
  provenance=json.loads((F/f'features/cases/{cid}.provenance.json').read_text());assert provenance['own_case_excluded_from_D_and_S_by_group']
  probability=np.load(F/f'features/cases/{cid}.npz')['x'][old['case_row'],2].astype(np.float32)
  dest=RUN/f'native_validation/{i}.npz';np.savez_compressed(dest,coords=coords.astype(np.int32),fallback_probability=probability,low=r['low'],high=r['high']);checks.append({'row':i,'case_id':cid,'source_D_S_gradient_exclusion':True,'target_sha256':sha256_file(dest)})
 write_json(RUN/'native_validation/READY.json',{'n_candidates':len(checks),'checks':checks,'source_only':True,'known_supervised_planning_exposure':True})
if __name__=='__main__':main()
