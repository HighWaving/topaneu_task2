from pathlib import Path
import json,hashlib,pickle,collections
P=Path(__file__).resolve().parents[1];V=P.parent;D=V/'data_topaneu26'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
R=P/'artifacts/astra6_e04_crop_segmentation_20260909T031146Z';C=read(R/'config.json')
out={'root':str(P),'files':{},'rules':{'source':'https://topaneu-26.grand-challenge.org/participation/','checked':'2026-09-09','seconds_per_case':720,'RAM_GB':32,'reserved_RAM_GB':1,'GPU':'T4 16GB','output':'single-channel uint8 0..52','external_data':'allowed with disclosure; no-external models required for manuscript comparison'}}
for p,h in C['scorer_hashes'].items():
 actual=sha(Path(p));assert actual==h;out['files'][p]=actual
ids=sorted(x.name.removesuffix('_0000.nii.gz') for x in (D/'images').glob('*.nii.gz'))
out['counts']={'images':len(ids),'modalities':dict(collections.Counter('MR' if '_mr_' in x else 'CT' for x in ids))}
out['dataset_index']=[{'case_id':c,'image_bytes':(D/f'images/{c}_0000.nii.gz').stat().st_size,'image_mtime_ns':(D/f'images/{c}_0000.nii.gz').stat().st_mtime_ns,'label_sha256':sha(D/f'location_masks/{c}.nii.gz')} for c in ids]
for f in ['README.md','CHANGELOG.txt','location_mapping.json','vessel_mapping.json']:out['files'][str(D/f)]=sha(D/f)
records=[json.loads(x) for x in (R/'features/train_records.jsonl').read_text().splitlines()];source={x['case_id'] for x in records};ev=set(read(P/'artifacts/m1_center2_mr_case_ids.json'));assert not source&ev and not any('center2' in c for c in source)
out['split']={'source_cases':sorted(source),'comparison_cases':sorted(ev),'no_overlap':True,'development':read(R/'source_split.json')}
cache=read(R/'features/SOURCE_READY.json');assert sha(R/'features/images.npy')==cache['images_sha256'] and sha(R/'features/targets.npy')==cache['targets_sha256'];out['cache']=cache
out['detector_checkpoint_sha256']=sha(P/'artifacts/fold1_checkpoint_snapshots/epoch60/model_last.ckpt');assert out['detector_checkpoint_sha256']=='f84488d9dfce235fc993c5dc9faca4fd35501b134b1a26a0248b4a309443980e'
out['candidate_files']={str(p):sha(p) for p in sorted((P/'artifacts/fold1_eval_center2_epoch60').glob('*boxes.pkl'))}
for task in ['Task030FG_TopAneuMR','Task031FG_TopAneuCT']:
 for p in (V/'nndet_data'/task).rglob('splits_final.pkl'):
  with p.open('rb') as f:splits=pickle.load(f)
  out.setdefault('detector_splits',{})[str(p)]=[{'train':list(s['train']),'val':list(s['val'])} for s in splits]
(P/'TAKEOVER_AUDIT_20260909.json').write_text(json.dumps(out,indent=2,default=str)+'\n')
print(out['counts']);print('cache, scorer and checkpoint hashes verified; split disjoint')
