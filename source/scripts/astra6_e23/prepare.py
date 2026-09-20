"""Build isolated two-fold MR OOF detector data; verify current raw image/label identity."""
import json,pickle,re,shutil,hashlib,subprocess,os
from scipy.ndimage import label
from pathlib import Path
import numpy as np,nibabel as nib
from scripts.astra6_e01.e01_common import P,DATA,sha256_file,write_json
RUN=P/'artifacts/astra6_e23_MR_oof_candidates_20260909';OLD=P.parent/'nndet_data/Task030FG_TopAneuMR';TASK='Task130FG_TopAneuMR_OOF';TARGET=RUN/'data'/TASK

def group(cid):return re.sub(r'(_(?:ct|mr)_\d+)_\d+$',r'\1',cid)
def main():
 for d in ['logs','models','checkpoints','oof_boxes','evidence']:(RUN/d).mkdir(parents=True,exist_ok=True)
 config={'experiment':'E23','hypothesis':'Detector in-sample source proposals underestimate out-of-sample false positives and miscalibrate downstream F/C; create OOF proposals before more downstream training','modality':'MR first','source_cases':267,'excluded_comparison_cases':40,'folds':2,'initialization':'from scratch, never from a TopAneu-trained checkpoint','split':'case-group disjoint, stratified center and positive/negative; official case IDs are the available patient linkage','detector':'same RetinaUNetV001 and existing D3V001_3d architecture/spacing plan','preprocessing_disclosure':'Reuses legacy supervised planning: all historical MR GT boxes informed anchors, including comparison and OOF cases. Per-case gradient exclusion is not whole-pipeline independence; current run is a legacy-plan developmental control.','training_budget':{'main_epochs':50,'swa_epochs':10,'train_batches_per_epoch':750,'validation_batches_per_epoch':100,'total_optimizer_updates_per_fold':45000,'maximum_expected_hours_per_fold':24},'checkpoint':'every epoch last with optimizer/scheduler, best validation proxy for diagnostics, fixed final epoch60 for OOF generation; no holdout epoch sweep','inference':'After each completed fold, infer every held-out source case, retaining200raw proposals; sourceGT labels candidates only afterward','downstream':'OOF hard negatives and detector positives for a shape-aware FP model and location model; deployment detector remains frozen until separate evidence justifies replacement','GPUs':['GPU-a643ded3-193b-58e8-b362-be93dc8eac14','GPU-f7491bbb-3972-1264-755a-63dec96b4a7d'],'no_MR40_or_CT5_fit':True};write_json(RUN/'config.json',config)
 if (RUN/'SOURCE_READY.json').exists():return
 ids=sorted(f.name.removesuffix('_0000.nii.gz') for f in (DATA/'images').glob('*_mr_*_0000.nii.gz') if 'center2_' not in f.name);assert len(ids)==267;comparison=json.loads((P/'artifacts/m1_center2_mr_case_ids.json').read_text());assert not set(ids)&set(comparison)
 checks={};checkpath=RUN/'evidence/source_identity.json'
 if checkpath.exists():checks=json.loads(checkpath.read_text())
 for n,cid in enumerate(ids,1):
  if cid in checks:continue
  a=DATA/f'images/{cid}_0000.nii.gz';b=OLD/f'raw_splitted/imagesTr/{cid}_0000.nii.gz';ha,hb=sha256_file(a),sha256_file(b);assert ha==hb,(cid,'raw images differ; must reprocess before training')
  g=nib.load(DATA/f'location_masks/{cid}.nii.gz');old=nib.load(OLD/f'raw_splitted/labelsTr/{cid}.nii.gz');ga=np.asarray(g.dataobj)>0;oa=np.asarray(old.dataobj)>0;assert ga.shape==oa.shape;assert np.allclose(g.affine,old.affine,atol=.03),(cid,'label physical geometry changed');count=int(ga.sum());corrected=not np.array_equal(ga,oa)
  if corrected:
   dest=RUN/'corrected_native_labels';dest.mkdir(exist_ok=True);inst,ninst=label(ga,structure=np.ones((3,3,3),np.uint8));assert ninst<256;im=nib.Nifti1Image(inst.astype(np.uint8),g.affine,g.header);im.set_data_dtype(np.uint8);im.to_filename(dest/f'{cid}.nii.gz');write_json(dest/f'{cid}.json',{'instances':{str(i):0 for i in range(1,ninst+1)}});write_json(RUN/f'evidence/{cid}_LABEL_CORRECTION.json',{'case_id':cid,'old_foreground_voxels':int(oa.sum()),'current_foreground_voxels':count,'different_voxels':int(np.count_nonzero(ga!=oa)),'old_label_sha256':sha256_file(OLD/f'raw_splitted/labelsTr/{cid}.nii.gz'),'current_label_sha256':sha256_file(DATA/f'location_masks/{cid}.nii.gz'),'original_files_untouched':True});del inst,im
   env=os.environ.copy();env['PYTHONPATH']=str(P.parent/'external/nnDetection');env['OMP_NUM_THREADS']='4';env['OPENBLAS_NUM_THREADS']='4';subprocess.run([str(P.parent/'conda_envs/nndet/bin/python'),str(P/'scripts/astra6_e23/reprocess_case.py'),'--case',cid,'--root',str(RUN)],env=env,check=True)
  del ga,oa,g,old
  checks[cid]={'image_sha256':ha,'current_location_mask_sha256':sha256_file(DATA/f'location_masks/{cid}.nii.gz'),'detector_instance_label_sha256':sha256_file(OLD/f'raw_splitted/labelsTr/{cid}.nii.gz'),'binary_target_voxels_identical':not corrected,'rebuilt_from_current_GT':corrected,'foreground_voxels':count,'group':group(cid)};write_json(checkpath,checks);print('E23_SOURCE_IDENTITY',n,len(ids),cid,flush=True)
 groups={}
 for cid in ids:groups.setdefault(group(cid),[]).append(cid)
 strata={}
 for key,cases in groups.items():strata.setdefault((key.split('_')[1],any(checks[c]['foreground_voxels'] for c in cases)),[]).append(key)
 rng=np.random.default_rng(20260909);parts=[[],[]]
 for key,gg in sorted(strata.items()):
  gg=sorted(gg);rng.shuffle(gg)
  for n,g in enumerate(gg):parts[n%2].extend(groups[g])
 splits=[]
 for fold in range(2):
  va=sorted(parts[fold]);tr=sorted(parts[1-fold]);assert not {group(c) for c in tr}&{group(c) for c in va};splits.append({'train':tr,'val':va})
 assert set(parts[0])|set(parts[1])==set(ids) and not set(parts[0])&set(parts[1])
 prep=TARGET/'preprocessed';dest=prep/'D3V001_3d/imagesTr';dest.mkdir(parents=True,exist_ok=True)
 for cid in ids:
  source=RUN/'reprocessed'/cid if checks[cid].get('rebuilt_from_current_GT') else OLD/'preprocessed/D3V001_3d/imagesTr'
  files=list(source.glob(cid+'.*'))+list(source.glob(cid+'_*'))
  assert any(f.name==cid+'.npz' for f in files) and any(f.name==cid+'_seg.npy' for f in files),cid
  for f in files:
   link=dest/f.name
   if not link.exists():link.symlink_to(f)
 # Dataset enumeration itself contains no comparison images. Shared immutable source arrays are read-only by workflow contract.
 actual={f.name.removesuffix('.npz') for f in dest.glob('*.npz')};assert actual==set(ids)
 shutil.copy2(OLD/'dataset.json',TARGET/'dataset.json');dataset=json.loads((TARGET/'dataset.json').read_text());dataset.update(name='TopAneuMR_OOF',task=TASK);write_json(TARGET/'dataset.json',dataset);shutil.copy2(OLD/'preprocessed/D3V001_3d.pkl',prep/'D3V001_3d.pkl');(prep/'splits_final.pkl').write_bytes(pickle.dumps(splits));write_json(RUN/'source_split.json',{'folds':splits,'comparison_excluded':comparison,'group_rule':'suffix-grouped case IDs; unavailable cross-ID patient linkage not invented','source_only':True});write_json(RUN/'SOURCE_READY.json',{'n_source':len(ids),'fold_sizes':[{'train':len(s['train']),'val':len(s['val'])} for s in splits],'current_image_and_binary_GT_verified':True,'reprocessed_updated_labels':[c for c,v in checks.items() if v.get('rebuilt_from_current_GT')],'identity_manifest_sha256':sha256_file(checkpath),'split_sha256':sha256_file(RUN/'source_split.json'),'preprocessing_plan_sha256':sha256_file(prep/'D3V001_3d.pkl')});print('E23_SOURCE_READY',flush=True)
if __name__=='__main__':main()
