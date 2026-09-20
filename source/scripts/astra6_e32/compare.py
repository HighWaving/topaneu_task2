"""Predeclared E32 comparison: fixed D/C/F, E31 normalized shape control, retained E17."""
import json
import numpy as np
from scripts.analysis.current_official_evaluation import aggregate,write
from pathlib import Path
P=Path(__file__).resolve().parents[2];RUN=P/'artifacts/astra6_e32_image_only_segmentation_20260910'
METRICS=['PRECISION','RECALL','MCC','DICE','VOLSIM','HD95']
def main():
 from scripts.analysis.metric_support_attribution import main as support
 support('E32_normalized')
 root=P/'artifacts/current_official_20260909';names=['E17','E31_normalized','E32_normalized'];official={n:json.loads((root/n/'official.json').read_text())['overall'] for n in names};cases={n:json.loads((root/n/'per_case.json').read_text()) for n in names};ids=[r['case_id'] for r in cases['E17']];assert all([r['case_id'] for r in cases[n]]==ids for n in names)
 samples={n:[] for n in names[:-1]};rng=np.random.default_rng(20260910)
 for _ in range(2000):
  ix=rng.integers(0,40,40);v={n:aggregate([cases[n][i]['raw'] for i in ix])['overall'] for n in names}
  for b in samples:samples[b].append([v['E32_normalized'][k]-v[b][k] for k in METRICS])
 comparisons={}
 for b in samples:
  delta={k:official['E32_normalized'][k]-official[b][k] for k in METRICS};comparisons['E32_vs_'+b]={'before':official[b],'after':official['E32_normalized'],'delta':delta,'paired_bootstrap_95CI':{k:np.percentile(np.asarray(samples[b])[:,j],[2.5,97.5]).tolist() for j,k in enumerate(METRICS)},'pareto':all(delta[k]>=-1e-10 for k in METRICS[:-1]) and delta['HD95']<=1e-10}
 source=json.loads((RUN/'normalized/model/SOURCE_SELECTION.json').read_text());ref=P/'artifacts/astra6_e31_physical_candidate_segmentation_20260910/normalized';control=json.loads((ref/'model/SOURCE_SELECTION.json').read_text());a={r['case_id']:[] for r in source['rows']};b={r['case_id']:[] for r in control['rows']}
 for r in source['rows']:a[r['case_id']].append(r['Dice'])
 for r in control['rows']:b[r['case_id']].append(r['Dice'])
 assert set(a)==set(b);diff=np.array([np.mean(a[k])-np.mean(b[k]) for k in sorted(a)]);boot=[float(np.mean(rng.choice(diff,len(diff),replace=True))) for _ in range(5000)]
 d=json.loads((RUN/'normalized/evaluation/lesion_diagnostics.json').read_text());m=d['E32_normalized'];baseline_shape=json.loads((ref/'evaluation/lesion_diagnostics.json').read_text())['E31_normalized']['matched_binary_Dice_mean']
 no_new_miss=not any(r['before']['matched'] and not r['after']['matched'] for r in d['paired_rows']);gates={'source_noninferiority':source['source_case_mean_Dice']>=control['source_case_mean_Dice']-.02,'official_pareto_vs_E17':comparisons['E32_vs_E17']['pareto'],'matched_at_least_53':m['matched']>=53,'no_new_miss':no_new_miss,'shape_gain_vs_E31_normalized':m['matched_binary_Dice_mean']>=baseline_shape+.005};qualified=all(gates.values())
 write(RUN/'COMPARISONS.json',{'comparisons':comparisons,'source_case_Dice_delta':float(diff.mean()),'source_paired_bootstrap_95CI':np.percentile(boot,[2.5,97.5]).tolist(),'source_selection_bias':True,'source':source['source_case_mean_Dice'],'source_control':control['source_case_mean_Dice'],'known_supervised_planning_and_repeated_MR40_limits':True})
 write(RUN/'DECISION.json',{'decision':'candidate_pending_raw_runtime_validation' if qualified else 'not_adopted','gates':gates,'metrics':m,'r2_unchanged':True,'next':'Diagnose paired shape/volume/size and vessel reliance; continue principal capability research, no automatic threshold sweep.'});print('E32 COMPLETE',qualified,gates,flush=True)
if __name__=='__main__':main()
