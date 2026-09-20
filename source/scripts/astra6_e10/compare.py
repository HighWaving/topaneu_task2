import json
import numpy as np
from scripts.analysis.current_official_evaluation import P,aggregate,write
R=P/'artifacts/astra6_e10_mask_contact_location_20260909';root=P/'artifacts/current_official_20260909'
b=json.loads((root/'E04/official.json').read_text());a=json.loads((root/'E10/official.json').read_text());metrics=['PRECISION','RECALL','MCC','DICE','VOLSIM','HD95'];delta={k:a['overall'][k]-b['overall'][k] for k in metrics}
br=json.loads((root/'E04/per_case.json').read_text());ar=json.loads((root/'E10/per_case.json').read_text());assert [x['case_id'] for x in br]==[x['case_id'] for x in ar];rng=np.random.default_rng(20260909);ds=[]
for _ in range(2000):
 ix=rng.integers(0,len(br),len(br));ba=aggregate([br[i]['raw'] for i in ix])['overall'];aa=aggregate([ar[i]['raw'] for i in ix])['overall'];ds.append([aa[k]-ba[k] for k in metrics])
ci={k:np.percentile(np.array(ds)[:,j],[2.5,97.5]).tolist() for j,k in enumerate(metrics)}
accepted=all(delta[k]>=-1e-10 for k in metrics if k!='HD95') and delta['HD95']<=1e-10 and any(abs(v)>1e-8 for v in delta.values())
report={'baseline':'E04','new':'E10','before':b['overall'],'after':a['overall'],'delta':delta,'paired_case_bootstrap_95CI':ci,'bootstrap_n':2000,'seed':20260909,'accepted_by_pareto_gate':accepted,'no_scalar_average':True,'source_validation':json.loads((R/'evaluation/source_validation.json').read_text()),'limitations':'Repeatedly observed positive-only MR40 development comparison; no center2 fit. Organizer TA36 patient-level lineage unverified.'};write(R/'evaluation/paired_official_comparison.json',report);print(json.dumps(report,indent=2),flush=True)
