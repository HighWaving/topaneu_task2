"""Explicit version arguments avoid copied-report version mismatches."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scripts.analysis.current_official_evaluation import P,RUNS,aggregate
METRICS=['PRECISION','RECALL','MCC','DICE','VOLSIM','HD95']
def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline',choices=list(RUNS),required=True);p.add_argument('--version',choices=list(RUNS),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=P/'artifacts/current_official_20260909';before=json.loads((root/a.baseline/'official.json').read_text())['overall'];after=json.loads((root/a.version/'official.json').read_text())['overall'];br=json.loads((root/a.baseline/'per_case.json').read_text());ar=json.loads((root/a.version/'per_case.json').read_text());assert [r['case_id'] for r in br]==[r['case_id'] for r in ar]
 for rows,expected in [(br,before),(ar,after)]:
  check=aggregate([r['raw'] for r in rows])['overall'];assert all(check[k]==expected[k] for k in METRICS)
 rng=np.random.default_rng(20260909);deltas=[]
 for _ in range(2000):
  ix=rng.integers(0,len(br),len(br));b=aggregate([br[i]['raw'] for i in ix])['overall'];n=aggregate([ar[i]['raw'] for i in ix])['overall'];deltas.append([n[k]-b[k] for k in METRICS])
 ci={k:np.percentile(np.asarray(deltas)[:,j],[2.5,97.5]).tolist() for j,k in enumerate(METRICS)};report={'baseline':a.baseline,'version':a.version,'before':before,'after':after,'delta':{k:after[k]-before[k] for k in METRICS},'paired_case_bootstrap_95CI':ci,'bootstrap_n':2000,'seed':20260909,'case_ids':[r['case_id'] for r in ar],'baseline_predictions':str(RUNS[a.baseline]),'new_predictions':str(RUNS[a.version]),'official_file_sha256':{v:hashlib.sha256((root/v/'official.json').read_bytes()).hexdigest() for v in [a.baseline,a.version]},'no_scalar_average':True,'caveat':'Repeatedly observed fixed development cohorts, not a new blind test; CT5 has only5 cases.'};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print('Paired final comparison saved',a.output,flush=True)
if __name__=='__main__':main()
