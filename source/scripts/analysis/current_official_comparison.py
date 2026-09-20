import json,sys
from pathlib import Path
import numpy as np
from current_official_evaluation import aggregate,write,P
ROOT=P/'artifacts/current_official_20260909';metrics=['PRECISION','RECALL','MCC','DICE','VOLSIM','HD95'];out={}
for before,after in [('E01','E04'),('E02','E04'),('CT_E02','CT_E04')]:
 b=json.loads((ROOT/before/'per_case.json').read_text());a=json.loads((ROOT/after/'per_case.json').read_text());assert [r['case_id'] for r in a]==[r['case_id'] for r in b]
 ba=aggregate([r['raw'] for r in b])['overall'];aa=aggregate([r['raw'] for r in a])['overall'];delta={k:aa[k]-ba[k] for k in metrics};rng=np.random.default_rng(20260909);samples={k:[] for k in metrics}
 for _ in range(2000):
  ix=rng.integers(0,len(a),len(a));bs=aggregate([b[i]['raw'] for i in ix])['overall'];ass=aggregate([a[i]['raw'] for i in ix])['overall']
  for k in metrics:samples[k].append(ass[k]-bs[k])
 boot={k:{'low':float(np.percentile(v,2.5)),'high':float(np.percentile(v,97.5)),'mean':float(np.mean(v))} for k,v in samples.items()}
 out[f'{after}_vs_{before}']={'before':ba,'after':aa,'delta':delta,'pareto_improvement':all(delta[k]>=0 for k in metrics if k!='HD95') and delta['HD95']<=0 and any(delta[k]!=0 for k in metrics),'paired_case_bootstrap':boot,'n_cases':len(a),'bootstrap_n':2000,'bootstrap_seed':20260909}
 write(ROOT/'paired_comparison.json',out);print(after,before,delta,flush=True)
