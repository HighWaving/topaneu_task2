"""Expand the unchanged E04 segmenter source data with42 eligible CT cases."""
from pathlib import Path
import json,shutil
import numpy as np
from scipy.ndimage import map_coordinates
from scripts.astra6_e01.e01_common import P,DATA,sha256_file,load_nifti,write_json
from scripts.astra6_e04.run_e04 import crop_coordinates,SUPPORT,load_image,component_records_fast,tests
RUN=P/'artifacts/astra6_e16_CT_expanded_segmentation_20260909';PARENT=P/'artifacts/astra6_e04_crop_segmentation_20260909T031146Z';RECORDS=P/'artifacts/astra6_e09_CT_expanded_location_20260909'
def main():
 for s in ['features','model','evaluation','predictions_ct']:(RUN/s).mkdir(parents=True,exist_ok=True)
 write_json(RUN/'config.json',{'experiment':'E16','hypothesis':'Adding42 eligible CT source cases improves CT tumor shape quality while keeping the existing E04 architecture and training recipe','change':'only expand positive segmentation source218to260cases; newCT source62to104; MR deployment remains frozenE04','inputs':'raw image32cube plus box prior, same two channels as E04; TA36 anatomy handled by frozen downstream classifiers/filters','training':'sameE04 optimizer/BCE+Dice/batch32; fixed13epochs for source helper and final, matching establishedE04 selected duration; no new duration sweep','source_validation':'E05 group-disjoint old development cases plus E09 eight newCT source-development cases; baseline helperE10 excludes old source-dev and never saw newCT','budget_hours':3,'checkpoint':'everyepoch model optimizer loader CPU/CUDA RNG','adoption':'CT5 all-six Pareto gain versus CT_E13, no matched-lesion loss; frozenD,C,F and selection','no_CT5_or_MR40_fit':True});write_json(RUN/'GEOMETRY_TEST.json',tests())
 rec=[json.loads(s) for s in (RECORDS/'features/train_records.jsonl').read_text().splitlines()];split=json.loads((RECORDS/'source_split.json').read_text());test=set(split['fixed_CT5']);assert len(rec)==5994 and not {r['case_id'] for r in rec}&test and not any('center2_mr' in r['case_id'] for r in rec);groups={};mapping=[];mir=[]
 for r in rec:
  key=(r['case_id'],r['source_class_id'],r['component_id'],r['sample_index'])
  if key not in groups:groups[key]=(len(groups),r)
  mapping.append(groups[key][0]);mir.append(r['view']=='mirror')
 assert len(groups)==2997;oldmap=np.load(PARENT/'features/rows.npz');assert np.array_equal(np.array(mapping[:4986]),oldmap['crop_index']);np.savez_compressed(RUN/'features/rows.npz',crop_index=mapping,mirrored=mir);shutil.copy2(RECORDS/'features/train_records.jsonl',RUN/'features/train_records.jsonl');progress_path=RUN/'features/progress.json';progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'completed_new_cases':[],'normalization':{}}
 oldx=np.load(PARENT/'features/images.npy',mmap_mode='r');oldy=np.load(PARENT/'features/targets.npy',mmap_mode='r');x=np.lib.format.open_memmap(RUN/'features/images.npy',mode='r+' if progress_path.exists() else 'w+',dtype=np.float16,shape=(2997,32,32,32));y=np.lib.format.open_memmap(RUN/'features/targets.npy',mode='r+' if progress_path.exists() else 'w+',dtype=np.uint8,shape=x.shape)
 if not progress_path.exists():x[:2493]=oldx;y[:2493]=oldy;x.flush();y.flush();write_json(progress_path,progress)
 bycase={}
 for idx,r in groups.values():
  if idx>=2493:bycase.setdefault(r['case_id'],[]).append((idx,r))
 assert len(bycase)==42
 for n,(cid,items) in enumerate(sorted(bycase.items()),1):
  if cid in progress['completed_new_cases']:continue
  arr,aff,norm=load_image(cid);gt,ga,shape=load_nifti(DATA/f'location_masks/{cid}.nii.gz');assert arr.shape==gt.shape and np.allclose(aff,ga,atol=1e-4);comps=component_records_fast(gt);cache={}
  for idx,r in items:
   key=(r['source_class_id'],r['component_id'])
   if key not in cache:
    base=next(z for _,z in items if z['source_class_id']==key[0] and z['component_id']==key[1] and z['sample_index']==0);comp=next(c for c in comps if c['class_id']==key[0] and np.array_equal(c['coords'].min(0)-.5,base['low']) and np.array_equal(c['coords'].max(0)+.5,base['high']));lo=comp['coords'].min(0);hi=comp['coords'].max(0)+1;small=np.zeros(tuple(hi-lo),np.uint8);small[tuple((comp['coords']-lo).T)]=1;cache[key]=(lo-1,np.pad(small,1))
   lo,small=cache[key];coords=crop_coordinates(r['low'],r['high']);x[idx]=map_coordinates(arr,coords.reshape(3,-1),order=1,mode='constant',cval=0,prefilter=False).reshape(32,32,32);target=map_coordinates(small,(coords-lo[:,None,None,None]).reshape(3,-1),order=0,mode='constant',cval=0,prefilter=False).reshape(32,32,32);valid=np.all((coords>=0)&(coords<=(np.array(shape)-1)[:,None,None,None]),axis=0);y[idx]=target*SUPPORT*valid;assert y[idx].sum()>0
  x.flush();y.flush();progress['completed_new_cases'].append(cid);progress['normalization'][cid]=norm;write_json(progress_path,progress);print('CT_new_segmentation_crops',n,42,cid,flush=True)
 assert np.array_equal(x[:2493],oldx) and np.array_equal(y[:2493],oldy) and np.isfinite(x).all();oldsplit=json.loads((P/'artifacts/astra6_e05_learned_location_splits_20260909/source_split.json').read_text());dev=set(oldsplit['development_cases'])|set(split['source_development_cases']);tr=[i for i,r in enumerate(rec) if r['case_id'] not in dev];dv=[i for i,r in enumerate(rec) if r['case_id'] in dev and r['view']=='original' and r['sample_index']==0];assert len(dv)==67 and not {rec[i]['case_id'] for i in tr}&dev;write_json(RUN/'source_split.json',{'development_cases':sorted(dev),'train_rows':tr,'development_rows':dv,'fixed_CT5':sorted(test),'source_group_rule':'oldE05 CT scan suffix groups plus newCT2 unique case IDs'});write_json(RUN/'features/SOURCE_READY.json',{'source_positive_cases':260,'CT_source_cases':104,'old_crops_bit_identical':True,'n_crops':2997,'n_rows':5994,'images_sha256':sha256_file(RUN/'features/images.npy'),'targets_sha256':sha256_file(RUN/'features/targets.npy'),'records_sha256':sha256_file(RUN/'features/train_records.jsonl')});print('SOURCE_READY',flush=True)
if __name__=='__main__':main()
