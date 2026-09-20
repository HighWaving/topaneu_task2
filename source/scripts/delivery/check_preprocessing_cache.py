import copy,json
from pathlib import Path
import numpy as np,torch
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from scripts.delivery.ta36_cached_preprocessing import CachedAdapter,OriginalAdapter
P=Path(__file__).resolve().parents[2];root=P.parent/'topaneu-task1-algorithm-model/ta36_models';rng=np.random.default_rng(20260909);arr=rng.normal(size=(1,32,36,40)).astype(np.float32);checks=[]
for d in sorted(root.iterdir()):
 pm=PlansManager(str(d/'plans.json'));dj=json.loads((d/'dataset.json').read_text());cm=pm.get_configuration('3d_fullres');props={'spacing':[.55,.4,.4]}
 expected=next(OriginalAdapter([arr],[None],[copy.deepcopy(props)],[None],pm,dj,cm,num_threads_in_multithreaded=1,verbose=False));actual=next(CachedAdapter([arr],[None],[copy.deepcopy(props)],[None],pm,dj,cm,num_threads_in_multithreaded=1,verbose=False));equal=torch.equal(expected['data'],actual['data']);assert equal
 serialize=lambda o:json.dumps(o,sort_keys=True,default=lambda x:x.tolist())
 assert serialize(expected['data_properties'])==serialize(actual['data_properties']);checks.append({'model':d.name,'bit_exact_tensor':equal,'identical_properties':True})
result={'checks':checks,'hits':CachedAdapter.hits,'misses':CachedAdapter.misses};assert result['hits']==2 and result['misses']==1;(P/'reports/ta36_preprocessing_equivalence_20260909.json').write_text(json.dumps(result,indent=2));print(result)
