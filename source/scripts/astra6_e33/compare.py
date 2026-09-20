import json
from pathlib import Path
import numpy as np
from scripts.analysis.current_official_evaluation import aggregate,write
P=Path(__file__).resolve().parents[2];RUN=P/'artifacts/astra6_e33_source_only_detector_20260910';METRICS=['PRECISION','RECALL','MCC','DICE','VOLSIM','HD95']
def main():
 from scripts.analysis.metric_support_attribution import main as support
 support('E33');root=P/'artifacts/current_official_20260909';names=['E17','E33'];official={n:json.loads((root/n/'official.json').read_text())['overall'] for n in names};cases={n:json.loads((root/n/'per_case.json').read_text()) for n in names};assert [r['case_id'] for r in cases['E17']]==[r['case_id'] for r in cases['E33']];rng=np.random.default_rng(20260910);samples=[]
 for _ in range(2000):
  ix=rng.integers(0,40,40);v={n:aggregate([cases[n][i]['raw'] for i in ix])['overall'] for n in names};samples.append([v['E33'][k]-v['E17'][k] for k in METRICS])
 delta={k:official['E33'][k]-official['E17'][k] for k in METRICS};comp={'before':official['E17'],'after':official['E33'],'delta':delta,'paired_bootstrap_95CI':{k:np.percentile(np.asarray(samples)[:,j],[2.5,97.5]).tolist() for j,k in enumerate(METRICS)}};write(RUN/'COMPARISONS.json',comp);m=json.loads((RUN/'evaluation/lesion_diagnostics.json').read_text())['E33'];f=json.loads((RUN/'FIXED_DOWNSTREAM.json').read_text());gates={'official_pareto':all(delta[k]>=-1e-10 for k in METRICS[:-1]) and delta['HD95']<=1e-10,'matched_at_least_53':m['matched']>=53,'location_correct_at_least_36':m['location_correct']>=36,'no_geometric_fallback_on_comparison':not any(c['legacy_empty_S_ellipse_count'] for c in f['cases'].values())};adopt=all(gates.values());write(RUN/'DECISION.json',{'decision':'candidate_pending_raw_T4_validation' if adopt else 'not_adopted','gates':gates,'metrics':m,'r2_unchanged':True,'planning_repair_is_not_performance_improvement_claim':True,'next':'If unmet, classify candidate misses vs false positives vs new-box shape/location shift before selecting the next learned module. Do not relax scores on MR40.'});print('E33 COMPLETE',adopt,gates,flush=True)
if __name__=='__main__':main()
