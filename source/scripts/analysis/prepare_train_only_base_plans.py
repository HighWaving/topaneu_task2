"""CPU-only fresh base planning; architecture/anchors remain explicitly pending."""
import json,pickle,hashlib
from pathlib import Path
import numpy as np
from nndet.planning.experiment.v001 import D3V001
P=Path(__file__).resolve().parents[2];R=P/'artifacts/train_only_planning_repair_20260909'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ready=json.loads((R/'PROPERTIES_READY.json').read_text());report={'status':'fresh_base_plans_only_architecture_anchors_pending','scopes':{}}
 for scope,expected in ready['scopes'].items():
  pre=R/scope/'preprocessed';props=pre/'properties/dataset_properties.pkl';assert sha(props)==expected['dataset_properties_sha256'];planner=D3V001(pre);plan=planner.plan_base('3d');assert set(plan['dataset_properties']['instance_props_per_patient'])==set(expected['cases']);assert plan['dataset_properties']['intensity_properties'] is None;assert plan['normalization_schemes'][0]=='nonCT';assert 'anchors' not in plan and 'architecture' not in plan
  out=pre/'BASE_ONLY_NOT_TRAINABLE.pkl';out.write_bytes(pickle.dumps(plan));report['scopes'][scope]={'base_plan_sha256':sha(out),'properties_sha256':sha(props),'target_spacing':np.asarray(plan['target_spacing']).tolist(),'transpose_forward':[int(x) for x in plan['transpose_forward']],'use_mask_for_norm':dict(plan['use_mask_for_norm']),'case_count':len(expected['cases'])}
 (R/'BASE_PLANS_READY.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
