import json
import numpy as np
from scripts.analysis.current_official_evaluation import P,aggregate,write
R=P/'artifacts/astra6_e11_CT_expanded_fp_filter_20260909';root=P/'artifacts/current_official_20260909';metrics=['PRECISION','RECALL','MCC','DICE','VOLSIM','HD95']
b=json.loads((root/'CT_E09/official.json').read_text());a=json.loads((root/'CT_E11/official.json').read_text());delta={k:a['overall'][k]-b['overall'][k] for k in metrics};br=json.loads((root/'CT_E09/per_case.json').read_text());ar=json.loads((root/'CT_E11/per_case.json').read_text());assert [r['case_id'] for r in br]==[r['case_id'] for r in ar];rng=np.random.default_rng(20260909);samples=[]
for _ in range(2000):
 ix=rng.integers(0,len(br),len(br));before=aggregate([br[i]['raw'] for i in ix])['overall'];after=aggregate([ar[i]['raw'] for i in ix])['overall'];samples.append([after[k]-before[k] for k in metrics])
ci={k:np.percentile(np.array(samples)[:,i],[2.5,97.5]).tolist() for i,k in enumerate(metrics)}
write(R/'evaluation/paired_official_comparison.json',{'before':b['overall'],'after':a['overall'],'delta':delta,'paired_bootstrap_95CI':ci,'seed':20260909,'n_bootstrap':2000,'pareto_gate':all(delta[k]>=-1e-10 for k in metrics[:-1]) and delta['HD95']<=1e-10 and any(abs(x)>1e-8 for x in delta.values()),'threshold':json.loads((R/'model/THRESHOLD.json').read_text()),'interpretation':'Source threshold locked before repeated CT5 development comparison; no overall scalar average.'});print(json.dumps({'after':a['overall'],'delta':delta},indent=2),flush=True)
