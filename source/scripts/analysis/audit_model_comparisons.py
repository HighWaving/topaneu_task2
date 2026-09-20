"""Verify each comparison refers to its own official prediction version."""
import json,math,hashlib
from pathlib import Path
from scripts.analysis.current_official_evaluation import RUNS,P
PAIRS={'CT_E20':('CT_E16','astra6_e20_CT_detector_crop_segmentation_20260909'),'E18':('E17','astra6_e18_MR_hierarchical_C_with_current_F_20260909'),'E17':('E14','astra6_e17_MR_detector_crop_segmentation_20260909'),'CT_E16':('CT_E13','astra6_e16_CT_expanded_segmentation_20260909'),'CT_E15':('CT_E13','astra6_e15_CT_predicted_vessel_location_20260909'),'CT_E13':('CT_E11','astra6_e13_CT_anatomical_fp_filter_20260909'),'E14':('E06','astra6_e14_MR_filter_seed_stability_20260909'),'E05':('E04','astra6_e05_learned_location_splits_20260909'),'E06':('E04','astra6_e06_image_fp_filter_20260909'),'E07':('E04','astra6_e07_hierarchical_location_20260909'),'E08':('E04','astra6_e08_mr_only_fp_filter_20260909'),'E08D':('E04','astra6_e08_deployed_pool_calibration_20260909'),'CT_E09':('CT_E04','astra6_e09_CT_expanded_location_20260909'),'E10':('E04','astra6_e10_mask_contact_location_20260909'),'CT_E11':('CT_E09','astra6_e11_CT_expanded_fp_filter_20260909'),'E12':('E06','astra6_e12_MR_expanded_fp_filter_20260909')}
METRICS=['PRECISION','RECALL','MCC','DICE','VOLSIM','HD95']
def main():
 root=P/'artifacts/current_official_20260909';out={};pending=[]
 for version,(baseline,run) in PAIRS.items():
  run=P/'artifacts'/run;report=run/'evaluation/paired_official_comparison.json'
  if not report.exists():pending.append(version);continue
  assert RUNS[version].is_relative_to(run),(version,run)
  r=json.loads(report.read_text());a=json.loads((root/version/'official.json').read_text())['overall'];b=json.loads((root/baseline/'official.json').read_text())['overall']
  for field,expected in [('before',b),('after',a)]:
   assert r[field].keys()==expected.keys(),(version,field,'metric keys')
   for k,v in expected.items():assert r[field][k]==v or (isinstance(v,float) and math.isnan(v) and math.isnan(r[field][k])),(version,field,k,r[field][k],v)
  for k in METRICS:assert abs(r['delta'][k]-(a[k]-b[k]))<1e-12,(version,k)
  pc=[json.loads((root/v/'per_case.json').read_text()) for v in [baseline,version]];assert [x['case_id'] for x in pc[0]]==[x['case_id'] for x in pc[1]]
  out[version]={'baseline':baseline,'metric_values_match_official':True,'paired_cases':len(pc[0]),'report_sha256':hashlib.sha256(report.read_bytes()).hexdigest(),'official_sha256':hashlib.sha256((root/version/'official.json').read_bytes()).hexdigest()}
 dest=P/'reports/MODEL_COMPARISON_INTEGRITY_20260909.json';dest.write_text(json.dumps({'verified':out,'not_yet_complete':pending},indent=2)+'\n');print('Verified comparison provenance',list(out),'pending',pending)
if __name__=='__main__':main()
