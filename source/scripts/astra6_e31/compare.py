import json,subprocess
from pathlib import Path
import numpy as np
from scripts.analysis.current_official_evaluation import aggregate,write
P=Path(__file__).resolve().parents[2];RUN=P/'artifacts/astra6_e31_physical_candidate_segmentation_20260910';METRICS=['PRECISION','RECALL','MCC','DICE','VOLSIM','HD95']
def main():
 for arm in ['normalized','physical']:
  subprocess.run([str(P.parent/'conda_envs/nnunet_v100/bin/python'),'-u','-m','scripts.astra6_e31.verify_native','--arm',arm],cwd=P,check=True)
 from scripts.analysis.metric_support_attribution import main as support_attribution
 for version in ['E31_normalized','E31_physical']:support_attribution(version)
 root=P/'artifacts/current_official_20260909';names=['E17','E31_normalized','E31_physical'];official={n:json.loads((root/n/'official.json').read_text())['overall'] for n in names};cases={n:json.loads((root/n/'per_case.json').read_text()) for n in names};ids=[r['case_id'] for r in cases['E17']];assert all([r['case_id'] for r in cases[n]]==ids for n in names)
 pairs=[('E31_normalized','E17'),('E31_physical','E17'),('E31_physical','E31_normalized')];samples={p:[] for p in pairs};rng=np.random.default_rng(20260910)
 for _ in range(2000):
  ix=rng.integers(0,40,40);v={n:aggregate([cases[n][i]['raw'] for i in ix])['overall'] for n in names}
  for a,b in pairs:samples[a,b].append([v[a][k]-v[b][k] for k in METRICS])
 comparisons={}
 for a,b in pairs:
  delta={k:official[a][k]-official[b][k] for k in METRICS};comparisons[a+'_vs_'+b]={'before':official[b],'after':official[a],'delta':delta,'paired_bootstrap_95CI':{k:np.percentile(np.asarray(samples[a,b])[:,j],[2.5,97.5]).tolist() for j,k in enumerate(METRICS)},'pareto':all(delta[k]>=-1e-10 for k in METRICS[:-1]) and delta['HD95']<=1e-10}
 source={a:json.loads((RUN/a/'model/SOURCE_SELECTION.json').read_text()) for a in ['normalized','physical']};reference=json.loads((RUN/'SOURCE_REFERENCE.json').read_text());base=reference['case_mean_Dice'];gates={'normalized':source['normalized']['source_case_mean_Dice']>=base+.01,'physical':source['physical']['source_case_mean_Dice']>=max(base,source['normalized']['source_case_mean_Dice'])+.01};qualified=[]
 for arm in source:
  name='E31_'+arm;d=json.loads((RUN/arm/'evaluation/lesion_diagnostics.json').read_text());m=d[name];adopt=bool(gates[arm] and comparisons[name+'_vs_E17']['pareto'] and m['matched']>=53 and m['matched_binary_Dice_mean']>=.6731629207819424+.005)
  if adopt:qualified.append(name)
  write(RUN/arm/'evaluation/DECISION.json',{'decision':'candidate_pending_raw_runtime_validation' if adopt else 'not_adopted','source_gate':gates[arm],'official_pareto':comparisons[name+'_vs_E17']['pareto'],'metrics':m,'r2_unchanged':True})
 write(RUN/'COMPARISONS.json',{'comparisons':comparisons,'source':source,'source_reference':reference,'source_gate':gates,'n_bootstrap':2000,'known_supervised_planning_and_repeated_MR40_limits':True});write(RUN/'DECISION.json',{'qualified_development_candidates':qualified,'r2_unchanged':True,'next_step':'Validate raw pipeline and runtime if qualified; otherwise diagnose physical crop sampling, source-to-MR40 transfer and remaining anatomical/shape limitations. No automatic threshold sweep.'});print('E31 COMPLETE',qualified,flush=True)
if __name__=='__main__':main()
