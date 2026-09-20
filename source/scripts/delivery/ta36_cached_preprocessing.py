"""Reuse exactly identical image-only DefaultPreprocessor results across TA36 models.
Model execution, resampled per-model softmax averaging and cleanup stay unchanged.
"""
import copy,json,runpy,sys
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy as OriginalAdapter

class CachedAdapter:
 cache={};hits=0;misses=0
 def __init__(self,*args,**kwargs):
  images,segs,props,outputs,pm,dj,cm=args[:7]
  valid=len(images)==1 and (segs is None or segs==[None]) and cm.preprocessor_class.__name__=='DefaultPreprocessor'
  conf=cm.configuration
  names=['preprocessor_name','spacing','normalization_schemes','use_mask_for_norm','resampling_fn_data','resampling_fn_data_kwargs','resampling_fn_seg','resampling_fn_seg_kwargs']
  sig={'config':{k:conf.get(k) for k in names},'transpose':pm.transpose_forward,'intensity':pm.foreground_intensity_properties_per_channel,'dataset':dj,'spacing':props[0]['spacing']}
  key=(id(images[0]),json.dumps(sig,sort_keys=True,default=lambda x:x.tolist())) if valid else None
  if key is not None and key in self.cache:
   CachedAdapter.hits+=1;c=self.cache[key];self.result={**copy.deepcopy({k:v for k,v in c.items() if k!='data'}),'data':c['data'].clone()};print('TA36 exact preprocessing cache HIT',flush=True)
  else:
   CachedAdapter.misses+=1;self.result=next(OriginalAdapter(*args,**kwargs))
   if key is not None:
    # Retain source to prevent an id from being recycled for another scan.
    CachedAdapter.cache={key:{**copy.deepcopy({k:v for k,v in self.result.items() if k!='data'}),'data':self.result['data'].clone()}};CachedAdapter.source=images[0]
   print('TA36 exact preprocessing cache MISS',flush=True)
 def __iter__(self):return self
 def __next__(self):return self.result

def main():
 import nnunetv2.houjing_scripts.infer_ppl_parallel_npz as pipeline
 pipeline.PreprocessAdapterFromNpy=CachedAdapter
 entry=sys.argv[1];sys.argv=[entry,*sys.argv[2:]];runpy.run_path(entry,run_name='__main__')
if __name__=='__main__':main()
