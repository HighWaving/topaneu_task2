import json
import numpy as np
from scripts.analysis.current_official_evaluation import P,aggregate,write
R=P/'artifacts/astra6_e25_oof_shape_fp_filter_20260909';root=P/'artifacts/current_official_20260909';metrics=['PRECISION','RECALL','MCC','DICE','VOLSIM','HD95']
b=json.loads((root/'E17/official.json').read_text());a=json.loads((root/'E25/official.json').read_text());delta={k:a['overall'][k]-b['overall'][k] for k in metrics};br=json.loads((root/'E17/per_case.json').read_text());ar=json.loads((root/'E25/per_case.json').read_text());assert [r['case_id'] for r in br]==[r['case_id'] for r in ar];rng=np.random.default_rng(20260909);samples=[]
for _ in range(2000):
 ix=rng.integers(0,len(br),len(br));before=aggregate([br[i]['raw'] for i in ix])['overall'];after=aggregate([ar[i]['raw'] for i in ix])['overall'];samples.append([after[k]-before[k] for k in metrics])
ci={k:np.percentile(np.array(samples)[:,i],[2.5,97.5]).tolist() for i,k in enumerate(metrics)}
write(R/'evaluation/paired_official_comparison.json',{'before':b['overall'],'after':a['overall'],'delta':delta,'paired_bootstrap_95CI':ci,'seed':20260909,'n_bootstrap':2000,'pareto_gate':all(delta[k]>=-1e-10 for k in metrics[:-1]) and delta['HD95']<=1e-10 and any(abs(x)>1e-8 for x in delta.values()),'source_validation':json.loads((R/'evaluation/SOURCE_RESULT.json').read_text()),'interpretation':'OOF shape-aware F source-selected and fully fitted before MR40 evaluation; D/C/S fixed to E17 baseline. MR40 has known legacy GT-anchor planning exposure and repeated research use; this is shared-plan developmental evidence, not independent generalization; no rawmetric overall average.'});print(json.dumps({'after':a['overall'],'delta':delta},indent=2),flush=True)

source=json.loads((R/'evaluation/SOURCE_RESULT.json').read_text());attribution=json.loads((R/'evaluation/PAIRED_ERROR_ATTRIBUTION.json').read_text());comparison=json.loads((R/'evaluation/paired_official_comparison.json').read_text());diag=json.loads((R/'evaluation/lesion_diagnostics.json').read_text())['E25'];lesion_gate=diag['matched']>=53 and diag['location_correct']>=36
adopt=bool(source['source_gate_passed'] and comparison['pareto_gate'] and lesion_gate)
evidence=[]
if source['negative_candidates']<10:evidence.append('Too few source operating negatives for the prespecified source test; hypothesis insufficiently tested, not disproved.')
if source['matched_components']<10:evidence.append('Too few source matched lesions to establish coverage-preserving calibration.')
if source['source_gate_passed'] and not comparison['pareto_gate']:evidence.append('Source benefit failed to transfer with identical deployed D/S/C; source-to-deployment candidate/S shift or small-source selection instability remain plausible, not yet separately identified.')
if attribution['GT_transitions']['lost']:evidence.append('Locked F removes previously matched true lesions; examine their source-versus-deployment probability and size distributions before any further fitting.')
if attribution['FP_transitions']['added_FP']>attribution['FP_transitions']['removed_FP']:evidence.append('Net false positives increased; source-derived operating threshold failed to control deployed negatives.')
if not evidence:evidence.append('Inspect paired FP and location/size strata; data and shape representation remain confounded in this experiment.')
write(R/'evaluation/DECISION.json',{'decision':'development_candidate_pending_clean_validation_and_runtime' if adopt else 'not_adopted','source_gate':bool(source['source_gate_passed']),'official_pareto_gate':comparison['pareto_gate'],'lesion_gate':lesion_gate,'valid_MR_baseline_for_next_research':'E25' if adopt else 'E17','evidence':evidence,'GT_transitions':attribution['GT_transitions'],'FP_transitions':attribution['FP_transitions'],'next_step':'Resolve dominant remaining failure using paired diagnostics. E28 joint image-anatomy representation is prepared if ICA/MCA remains dominant. Do not end at rejection or alter MR40 thresholds.','release_r2_unchanged_until_full_runtime_verified':True})
# Continue sequentially into a capability-changing location experiment only if
# the valid baseline still has the predeclared ICA/MCA error burden.
import subprocess,os
subprocess.run([str(P.parent/'conda_envs/nnunet_v100/bin/python'),'-m','scripts.astra6_e28.chain'],cwd=P,env=os.environ.copy(),check=True)
