"""CT anatomical FP filter features: actual TA36 predictions and frozen E11 image scores."""
from pathlib import Path
import json,shutil
import numpy as np,torch
from scripts.astra6_e01.e01_common import P,DATA,compute_feature,load_nifti_geometry,sha256_file,write_json
from scripts.astra6_e02.run_e02 import multiscale
from scripts.astra6_e06.common import Filter
from scripts.delivery.geometry import vessel_geometry_fast
RUN=P/'artifacts/astra6_e13_CT_anatomical_fp_filter_20260909';PARENT=P/'artifacts/astra6_e11_CT_expanded_fp_filter_20260909'
def image_probabilities(weights,dest):
 if dest.exists():return np.load(dest)
 x=np.load(PARENT/'features/images.npy',mmap_mode='r');model=Filter().cuda();model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=False)['state_dict']);model.eval();ps=[]
 with torch.inference_mode():
  for start in range(0,len(x),64):ps.extend(model(torch.from_numpy(np.asarray(x[start:start+64],np.float32)).cuda()).sigmoid().cpu().tolist())
 out=np.asarray(ps,np.float32);np.save(dest,out);del model;torch.cuda.empty_cache();return out

def main():
 torch.set_num_threads(4)
 for s in ['features/cases','model','evaluation','predictions_ct']:(RUN/s).mkdir(parents=True,exist_ok=True)
 config={'experiment':'E13','hypothesis':'anatomical context from actual TA36 predictions can reject CT false positives left by the E11 image-only filter','control':'CT E09C/E04S/E11F','inputs':'same943 geometric vessel features as E02 plus frozen E11 image probability; no GT vessel input at training or inference','source':'same104CT cases as E11; actual TA36 predictions already available for these cases','source_validation':'same E11 source case groups; E11 development checkpoint for development features; final checkpoint for final and runtime features','classifier':{'type':'ExtraTrees','n_estimators':512,'max_depth':12,'min_samples_leaf':5,'max_features':.5,'bootstrap':False,'seed':20260909,'class_weight':'balanced'},'budget_hours':3,'checkpoint_every_trees':64,'threshold':'minimum positive anatomical score among source-dev original deployment candidates passing source E11 threshold; fixed beforeCT5','adoption':'no matched-lesion loss and all-six Pareto gain over CT E11; retain E11 otherwise','no_CT5_or_MR40_fit':True}
 write_json(RUN/'config.json',config);shutil.copy2(PARENT/'source_split.json',RUN/'source_split.json');shutil.copy2(PARENT/'features/records.jsonl',RUN/'features/records.jsonl');rec=[json.loads(s) for s in (RUN/'features/records.jsonl').read_text().splitlines()];test=set(json.loads((RUN/'source_split.json').read_text())['fixed_CT5']);assert len({r['case_id'] for r in rec})==104 and not {r['case_id'] for r in rec}&test
 probs={stage:image_probabilities(PARENT/f'model/{name}.pt',RUN/f'features/{stage}_image_probability.npy') for stage,name in [('development','development_best'),('final','final_last')]};groups={}
 for i,r in enumerate(rec):groups.setdefault(r['case_id'],[]).append(i)
 X=np.empty((len(rec),943),np.float32);vhash={}
 for n,(cid,ix) in enumerate(sorted(groups.items()),1):
  dest=RUN/f'features/cases/{cid}.npz';vp=P/f'artifacts/ta36_ct_output/{cid}.nii.gz';vhash[cid]=sha256_file(vp)
  if dest.exists():data=np.load(dest);assert np.array_equal(data['indices'],ix);X[ix]=data['X'];continue
  aff,shape=load_nifti_geometry(DATA/f'images/{cid}_0000.nii.gz');geom=vessel_geometry_fast(vp,shape,aff);xx=[]
  for i in ix:
   lo=np.array(rec[i]['low']);hi=np.array(rec[i]['high']);xx.append(np.concatenate([compute_feature(geom,aff,lo,hi,'CT'),multiscale(geom,aff,lo,hi)]))
  xx=np.asarray(xx,np.float32);X[ix]=xx;np.savez_compressed(dest,X=xx,indices=ix);print('anatomical_source',n,104,cid,flush=True)
 for stage,pp in probs.items():np.savez_compressed(RUN/f'features/{stage}.npz',X=np.column_stack([X,pp]),y=np.array([r['y'] for r in rec],np.uint8))
 assert np.isfinite(X).all();write_json(RUN/'features/READY.json',{'source_cases':104,'rows':len(rec),'feature_dim':944,'TA36_source_prediction_hashes':vhash,'development_features_sha256':sha256_file(RUN/'features/development.npz'),'final_features_sha256':sha256_file(RUN/'features/final.npz'),'records_sha256':sha256_file(RUN/'features/records.jsonl'),'image_development_checkpoint_sha256':sha256_file(PARENT/'model/development_best.pt'),'image_final_checkpoint_sha256':sha256_file(PARENT/'model/final_last.pt'),'parent_source':json.loads((PARENT/'features/READY.json').read_text())});print('SOURCE_READY',flush=True)
if __name__=='__main__':main()
