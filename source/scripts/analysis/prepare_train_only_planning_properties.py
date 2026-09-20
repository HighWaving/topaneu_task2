"""Prepare train-only MR planner inputs, never modify active E23 data or weights.

Reuses verified per-case measurements only, not learned global plans. Rebuilds
all aggregate statistics; corrects the changed 702 native instance mask first.
This creates properties, NOT a trained detector or a clean-validation claim.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS','1')
import json,pickle,hashlib,copy,re
from pathlib import Path
from collections import OrderedDict,defaultdict
import numpy as np
import SimpleITK as sitk
P=Path(__file__).resolve().parents[2];V=P.parent;E=P/'artifacts/astra6_e23_MR_oof_candidates_20260909';R=P/'artifacts/train_only_planning_repair_20260909'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def group(c):return re.sub(r'(_(?:mr|ct)_\d+)_\d+$',r'\1',c)
def main():
 split=json.loads((E/'source_split.json').read_text());identity=json.loads((E/'evidence/source_identity.json').read_text());oldpath=V/'nndet_data/Task030FG_TopAneuMR/preprocessed/D3V001_3d.pkl';old=pickle.loads(oldpath.read_bytes())['dataset_properties'];source=set(identity);assert len(source)==267
 per={c:copy.deepcopy(old['instance_props_per_patient'][c]) for c in sorted(source)};changes={}
 for c,q in identity.items():
  assert q['binary_target_voxels_identical'] or q.get('rebuilt_from_current_GT',False)
  if not q.get('rebuilt_from_current_GT',False):continue
  native=E/f'corrected_native_labels/{c}.nii.gz';mapping=E/f'corrected_native_labels/{c}.json';im=sitk.ReadImage(str(native));a=sitk.GetArrayFromImage(im);prop=per[c]
  assert tuple(a.shape)==tuple(prop['original_size_of_raw_data']);assert np.allclose(im.GetSpacing(),prop['itk_spacing'],atol=1e-5);assert int((a>0).sum())==q['foreground_voxels']
  crop=a[tuple(slice(*pair) for pair in prop['crop_bbox'])];assert int((crop>0).sum())==int((a>0).sum());ids=np.unique(crop);ids=ids[ids>0];instances=json.loads(mapping.read_text())['instances'];assert set(map(int,instances))==set(map(int,ids));boxes=[];vol=[]
  for inst in ids:
   coords=np.argwhere(crop==inst);lo=coords.min(0)-1;hi=coords.max(0)+1;boxes.append([lo[0],lo[1],hi[0],hi[1],lo[2],hi[2]]);vol.append(len(coords)*float(np.prod(im.GetSpacing())))
  b=np.asarray(boxes);low=b[:,[0,1,4]];high=b[:,[2,3,5]];inter=np.maximum(np.minimum(high[:,None],high[None])-np.maximum(low[:,None],low[None]),0).prod(2);size=(high-low).prod(1);iou=inter/(size[:,None]+size[None]-inter);iou=iou[~np.eye(len(b),dtype=bool)].reshape(len(b),-1)
  changes[c]={'native_label_sha256':sha(native),'old_boxes':np.asarray(prop['boxes']).tolist(),'current_boxes':b.tolist(),'foreground_voxels':q['foreground_voxels']}
  prop.update(instances=instances,num_instances={0:len(ids)},has_classes=[0],volume_per_class={0:sum(vol)},region_volume_per_class={0:vol},boxes=b,all_ious=iou,class_ious={0:iou},seg_file=str(native));del a,crop,im
 scopes={f'fold{f}':s['train'] for f,s in enumerate(split['folds'])};scopes['final_source267']=sorted(source);summary={'status':'planner_inputs_ready_not_models_or_validated_plans','source_identity_sha256':sha(E/'evidence/source_identity.json'),'source_split_sha256':sha(E/'source_split.json'),'old_per_case_measurement_container_sha256':sha(oldpath),'updated_native_properties':changes,'scopes':{},'remaining':['Run fresh architecture/anchor planning on each scope after E23 GPU work; no old anchors or global CT intensity reuse.','Preprocess with resulting plans and formally train new weights from scratch.','Rebuild affected downstream source features and fits; audit TA36 separately.','Repeated MR40 remains developmental even after known planning exposure is repaired.']}
 for name,cases in scopes.items():
  cases=sorted(cases);assert set(cases)<=source and not set(cases)&set(split['comparison_excluded']);excluded=sorted(source-set(cases));assert not {group(c) for c in cases}&{group(c) for c in excluded};props=OrderedDict((c,per[c]) for c in cases)
  counts=defaultdict(int);ci=defaultdict(list)
  for q in props.values():
   for k,n in q['num_instances'].items():counts[k]+=n
   for k,x in q['class_ious'].items():ci[k].append(x.flatten())
  d={'dim':3,'all_sizes':[q['size_after_cropping'] for q in props.values()],'all_spacings':[q['original_spacing'] for q in props.values()],'size_reductions':OrderedDict((c,float(np.prod(q['size_after_cropping'])/np.prod(q['original_size_of_raw_data']))) for c,q in props.items()),'modalities':copy.deepcopy(old['modalities']),'class_dct':copy.deepcopy(old['class_dct']),'all_classes':copy.deepcopy(old['all_classes']),'instance_props_per_patient':props,'num_instances':dict(counts),'class_ious':{k:np.concatenate(v) for k,v in ci.items()},'all_ious':np.concatenate([q['all_ious'].flatten() for q in props.values()]),'intensity_properties':None}
  assert d['modalities']=={0:'TOF'} # MR per-image normalization: no global GT intensity collection needed.
  dest=R/name/'preprocessed/properties';dest.mkdir(parents=True,exist_ok=True);file=dest/'dataset_properties.pkl';content=pickle.dumps(d)
  if file.exists():assert file.read_bytes()==content,'Refusing to overwrite differing prepared properties'
  else:file.write_bytes(content)
  summary['scopes'][name]={'cases':cases,'excluded_source_cases':excluded,'comparison_excluded':split['comparison_excluded'],'n_cases':len(cases),'n_instances':sum(counts.values()),'dataset_properties_sha256':sha(file),'intensity_properties':None,'no_legacy_global_summary_or_anchors_copied':True}
 R.mkdir(exist_ok=True);(R/'PROPERTIES_READY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:{'n_cases':v['n_cases'],'n_instances':v['n_instances']} for k,v in summary['scopes'].items()},indent=2),flush=True)
if __name__=='__main__':main()
