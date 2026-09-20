"""Measured detector throughput plus explicitly provisional downstream budgets."""
from pathlib import Path
import json,time,datetime,statistics,math
P=Path(__file__).resolve().parents[2];O=P/'artifacts/astra6_e23_MR_oof_candidates_20260909';R=P/'artifacts/research_audit_20260909'
def snapshot():
 now=datetime.datetime.now(datetime.timezone.utc);folds={}
 for f in [0,1]:
  rows=[]
  for line in (O/f'logs/fold{f}_workflow_resumed.log').read_text().splitlines():
   if line.startswith('E23_BATCH_PROGRESS '):
    try:rows.append(json.loads(line.split(' ',1)[1]))
    except json.JSONDecodeError:pass
  last=rows[-1];window=max(1500,last['global_step']-1000);deltas=[(b['timestamp']-a['timestamp'])/(b['global_step']-a['global_step']) for a,b in zip(rows,rows[1:]) if a['global_step']>=window and b['global_step']>a['global_step'] and a['epoch_zero_based']==b['epoch_zero_based']];assert len(deltas)>=5;med=statistics.median(deltas);q=statistics.quantiles(deltas,n=10,method='inclusive');overheads=[max(0,b['timestamp']-a['timestamp']-(b['global_step']-a['global_step'])*med) for a,b in zip(rows,rows[1:]) if b['epoch_zero_based']==a['epoch_zero_based']+1 and 0<b['global_step']-a['global_step']<=50];validation=statistics.median(overheads) if overheads else 120.;updates=max(0,45000-last['global_step']);epochs=max(0,60-last['epoch_zero_based'])
  if (O/f'checkpoints/fold{f}_TRAINED.json').exists():updates=0;epochs=0
  point=(updates*med+epochs*validation)/3600;low=(updates*q[1]+epochs*validation*.8)/3600;high=(updates*q[-2]+epochs*validation*1.3)/3600;folds[str(f)]={'global_step':last['global_step'],'planned_updates':45000,'recent_window_start_step':window,'sample_intervals':len(deltas),'median_seconds_per_update':med,'p20_p80_seconds_per_update':[q[1],q[-2]],'measured_epoch_validation_checkpoint_overhead_seconds':validation,'observed_validation_boundaries':len(overheads),'remaining_training_hours_point':point,'remaining_training_hours_scenario_range':[low,high],'ETA_training_UTC':(now+datetime.timedelta(hours=point)).isoformat()}
 trainpoint=max(v['remaining_training_hours_point'] for v in folds.values());trainrange=[max(v['remaining_training_hours_scenario_range'][i] for v in folds.values()) for i in [0,1]];downstream={'two_source_disjoint_S_helpers_parallel':[.05,.25],'OOF_detector_inference_parallel_two_folds':[1.7,4.5],'E25_feature_preparation':[.5,1.5],'E25_development_and_final_training':[.5,4.0],'MR40_inference_official6_diagnostics_and_comparison':[.5,1.0],'selected_release_runtime_checks_if_needed':[.5,1.5]};
 measurements={};F=P/'artifacts/astra6_e25_oof_shape_fp_filter_20260909';C=P/'artifacts/astra6_e28_joint_anatomy_location_20260909'
 if all((O/f'segmenters/fold{f}/model/LOCKED.json').exists() for f in [0,1]):downstream['two_source_disjoint_S_helpers_parallel']=[0.,0.]
 inference_ranges=[]
 for f in [0,1]:
  files=sorted((O/f'oof_boxes/fold{f}').glob('*_boxes.pkl'),key=lambda p:p.stat().st_mtime);n_expected=134 if f==0 else 133;missing=max(0,n_expected-len(files));times=[p.stat().st_mtime for p in files[-11:]];intervals=[b-a for a,b in zip(times,times[1:]) if b>a]
  if len(intervals)>=9:
   med=statistics.median(intervals);q=statistics.quantiles(intervals,n=5,method='inclusive');bounds=[missing*q[0]/3600,missing*q[-1]/3600];measurements[f'OOF_fold{f}']={'complete_cases':len(files),'recent_case_seconds_median':med,'remaining_hours_scenario':bounds}
  else:bounds=[missing*45/3600,missing*120/3600]
  inference_ranges.append(bounds)
 downstream['OOF_detector_inference_parallel_two_folds']=[max(x[i] for x in inference_ranges) for i in [0,1]]
 done_features=(F/'features/READY.json').exists()
 if done_features:downstream['E25_feature_preparation']=[0.,0.]
 if (F/'model/LOCKED.json').exists():downstream['E25_development_and_final_training']=[0.,0.]
 elif (F/'model/development_history.json').exists():
  h=json.loads((F/'model/development_history.json').read_text());sec=statistics.median(r['seconds'] for r in h[-5:])*1.2;best=min(h,key=lambda r:r['dev_balanced_BCE'])['epoch'];n=h[-1]['epoch'];final_h=json.loads((F/'model/final_history.json').read_text()) if (F/'model/final_history.json').exists() else [];final_sec=statistics.median(r['seconds'] for r in final_h[-5:]) if final_h else sec*2.5
  if (F/'model/THRESHOLD.json').exists():
   selected=json.loads((F/'model/THRESHOLD.json').read_text())['selected_epoch'];left=max(0,selected-(final_h[-1]['epoch'] if final_h else 0));bounds=[left*final_sec*.8/3600,left*final_sec*1.3/3600]
  else:bounds=[(max(0,max(30,best+20)-n)*sec+best*final_sec)*.8/3600,((100-n)*sec+100*final_sec)*1.3/3600]
  downstream['E25_development_and_final_training']=bounds;measurements['E25']={'development_epochs_completed':n,'recent_epoch_seconds_with_validation_allowance':sec,'final_epoch_seconds_measured_or_extrapolated':final_sec,'remaining_hours_scenario':bounds}
 if (F/'evaluation/DECISION.json').exists():downstream['MR40_inference_official6_diagnostics_and_comparison']=[0.,0.]
 conditional_C=[.5,8.]
 if (C/'evaluation/DECISION.json').exists() or (C/'NEXT_BOTTLENECK_REVIEW_REQUIRED.json').exists():conditional_C=[0.,0.]
 remaining=[trainrange[i]+sum(v[i] for v in downstream.values()) for i in [0,1]];out={'computed_UTC':now.isoformat(),'measured_downstream_updates':measurements,'conditional_E28_additional_hours_scenario':conditional_C,'scope':'Primary remaining range covers E23/E25 and potential delivery checks. E28 additional range is conditional; mandatory train-only planning repair/retraining, later CT research and platform access are not yet forecast.','folds':folds,'detector_training_remaining_hours_point':trainpoint,'subsequent_hours_provisional':downstream,'whole_chain_remaining_hours_scenario_range':remaining,'whole_chain_ETA_UTC_scenario_range':[(now+datetime.timedelta(hours=x)).isoformat() for x in remaining],'basis':'Training range uses recent same-epoch update-time quantiles plus measured validation/save overhead. OOF inference provisional45-120sec/case x134 on each parallelGPU, informed by existing99.8sec representative and125.1sec maximum rawMR D including preprocessing; OOF avoids preprocessing. S helpers informed by E17 full5368row epoch~14sec. Later F time remains config-budget estimate until OOF rowcount and firstfull epoch are known.','limitations':'Scenario range is not a statistical confidence interval or completion guarantee. Includes downstream stages and one locked end-to-end evaluation even if the source gate fails. Queue/contention and changed candidate counts can alter it. Re-estimate after first10OOFcases and firstfullE25epoch. No budgets shortened.'};R.mkdir(exist_ok=True);tmp=R/'BUDGET.tmp';tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.replace(R/'BUDGET_CURRENT.json');return out
if __name__=='__main__':print(json.dumps(snapshot(),indent=2),flush=True)
