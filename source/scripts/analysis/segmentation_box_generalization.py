"""Summarize saved paired source masks; no image inference, fitting, or GT input."""
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

P=Path(__file__).resolve().parents[2]

def main():
    source=P/'artifacts/astra6_e23_MR_oof_candidates_20260909/segmentation_source_audit/RESULT.json'
    d=json.loads(source.read_text());qualified=set(d['paired_source_rows'])
    rows=[r for r in d['all_rows'] if r['source_row'] in qualified]
    summary={'n':len(rows),'cases':len({r['case_id'] for r in rows})}
    for name in ['old','OOF','oracle32','oracle48']:
        summary[name]={k:statistics.mean(r[name][k] for r in rows) for k in ['native_Dice','relative_volume_error','GT_inside_box_fraction']}
    bycase={c:[r['OOF']['native_Dice']-r['old']['native_Dice'] for r in rows if r['case_id']==c] for c in sorted({r['case_id'] for r in rows})}
    rng=random.Random(20260910);keys=list(bycase);samples=[]
    for _ in range(2000):
        selected=[keys[rng.randrange(len(keys))] for j in keys]
        samples.append(statistics.mean(v for c in selected for v in bycase[c]))
    samples.sort();summary['case_bootstrap_95_percentile_Dice_delta']=[samples[49],samples[1949]]
    x=[];y=[]
    for r in rows:
        a,b=r['old_box'],r['OOF_box'];ea=[a[1][i]-a[0][i] for i in range(3)];eb=[b[1][i]-b[0][i] for i in range(3)]
        r['box_volume_ratio_OOF_over_old']=math.prod(eb)/math.prod(ea)
        r['center_shift_old_box_diagonal_fraction']=math.sqrt(sum(((a[0][i]+a[1][i]-b[0][i]-b[1][i])/2)**2 for i in range(3)))/math.sqrt(sum(v*v for v in ea))
        r['Dice_delta']=r['OOF']['native_Dice']-r['old']['native_Dice']
        x.append(math.log(r['box_volume_ratio_OOF_over_old']))
        y.append(math.log((1+r['OOF']['relative_volume_error'])/(1+r['old']['relative_volume_error'])))
    mx,my=statistics.mean(x),statistics.mean(y)
    summary['log_box_volume_change_vs_log_predicted_volume_change_Pearson']=sum((a-mx)*(b-my) for a,b in zip(x,y))/math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
    fields=['case_id','source_row','class','GT_voxels','Dice_delta','box_volume_ratio_OOF_over_old','center_shift_old_box_diagonal_fraction','old','OOF','oracle32','oracle48']
    summary['largest_losses']=[{k:r[k] for k in fields} for r in sorted(rows,key=lambda r:r['Dice_delta'])[:6]]
    summary['source_result_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    summary['fixed_S_sha256']=d['plan']['fixed_S_sha256']
    summary['limitations']=['Source-only paired diagnostic. Old versus OOF D differs in training amount/weights and own-case exposure; legacy GT planning shared. Not a causal isolation of detector overfitting.',
        'Oracle32/48 uses GT for diagnostic round-trip only; native routine can use ellipse when empty and these values are not strict learned-model upper bounds.',
        'Bootstrap clustered by source case, descriptive and not an independent validation or training selection criterion.']
    summary['preliminary_next_capability_hypothesis']='Reduce exact detector-box scale dependence using physically controlled image sampling and explicit candidate localization; evaluate on unchanged candidates. Do not start a new segmentation branch before E25/main location results prioritize the bottleneck.'
    out=P/'artifacts/review_current_model_20260910/SEGMENTATION_BOX_GENERALIZATION.json'
    out.write_text(json.dumps(summary,indent=2)+'\n')
    print({k:summary[k] for k in ['n','cases','case_bootstrap_95_percentile_Dice_delta','log_box_volume_change_vs_log_predicted_volume_change_Pearson']})

if __name__=='__main__':main()
