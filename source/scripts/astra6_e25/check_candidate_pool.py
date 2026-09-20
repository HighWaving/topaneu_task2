"""Regression check: every deployed candidate must exist in the source pool."""
import numpy as np
from scripts.astra6_e25.source_candidate_bias import wide_indices
from scripts.astra6_e01.e01_common import select_candidates,write_json
from scripts.astra6_e25.common import RUN
rng=np.random.default_rng(20260909);example=None
for trial in range(200):
 scores=rng.choice([0.,.04,.05,.1,.3,.5,.8,1.],size=200);boxes=np.tile([0,0,1,1,0,1],(len(scores),1));operating={i for i,_,_,_ in select_candidates(boxes,scores)};wide=wide_indices(scores);assert operating<=set(wide) and len(wide)<=20 and all(scores[i]>=.05 for i in wide)
 old=[i for i,s in sorted(enumerate(scores),key=lambda x:(-x[1],x[0])) if s>=.05][:20]
 if example is None and not operating<=set(old):example={'trial':trial,'operating_indices':sorted(operating),'old_pool_omitted_operating_indices':sorted(operating-set(old)),'correct_pool_indices':wide}
assert wide_indices(np.array([]))==[]
write_json(RUN/'CANDIDATE_POOL_CONTRACT_CHECK.json',{'synthetic_quantized_score_trials':200,'all_operating_candidates_in_wide_pool':True,'tie_mismatch_example':example,'empty_pool_valid':True,'no_fit_no_GT_no_real_prediction_change':True})
print('Candidate pool contract passed; old tie mismatch example:',example,flush=True)
