import json,numpy as np
from pathlib import Path
from scripts.analysis.current_official_evaluation import aggregate,write
P=Path(__file__).resolve().parents[2];RUN=P/'artifacts/astra6_e30_expanded_CT_supervision_for_MR_20260910'
METRICS=['PRECISION','RECALL','MCC','DICE','VOLSIM','HD95']
def main():
 root=P/'artifacts/current_official_20260909';names=['E17','E30'];official={n:json.loads((root/n/'official.json').read_text())['overall'] for n in names};cases={n:json.loads((root/n/'per_case.json').read_text()) for n in names}
 assert [r['case_id'] for r in cases['E17']]==[r['case_id'] for r in cases['E30']]
 rng=np.random.default_rng(20260910);samples=[]
 for _ in range(2000):
  ix=rng.integers(0,40,40);v={n:aggregate([cases[n][i]['raw'] for i in ix])['overall'] for n in names};samples.append([v['E30'][k]-v['E17'][k] for k in METRICS])
 delta={k:official['E30'][k]-official['E17'][k] for k in METRICS};gate=all(delta[k]>=-1e-10 for k in METRICS[:-1]) and delta['HD95']<=1e-10
 d=json.loads((RUN/'evaluation/lesion_diagnostics.json').read_text());src=json.loads((RUN/'SOURCE_RESULT.json').read_text());adopt=bool(src['source_gate'] and gate and d['E30']['matched']>=53 and d['E30']['location_correct']>36)
 write(RUN/'COMPARISON.json',{'before':official['E17'],'after':official['E30'],'delta':delta,'paired_bootstrap_95CI':{k:np.percentile(np.asarray(samples)[:,j],[2.5,97.5]).tolist() for j,k in enumerate(METRICS)},'n_bootstrap':2000,'source_gate':src['source_gate'],'official_pareto_gate':gate})
 write(RUN/'DECISION.json',{'decision':'candidate_pending_raw_runtime_validation' if adopt else 'not_adopted','location_rescued':d['location_rescued'],'location_lost':d['location_lost'],'r2_unchanged':True,'known_planning_exposure_and_repeated_MR40_research':True,'next_step':'Investigate remaining capability bottleneck; no additional tree tuning on these cases.'})
 print('E30 complete',official['E30'],adopt,flush=True)
if __name__=='__main__':main()
