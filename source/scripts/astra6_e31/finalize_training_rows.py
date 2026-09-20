"""Pre-fit source-only shape-quality exclusion, identical in both arms.
Keep original annotations, all raw feature arrays and all validation rows.
"""
import json,time,shutil
from scripts.astra6_e31.common import RUN
from scripts.astra6_e01.e01_common import write_json,sha256_file

def main():
 marker=RUN/'SOURCE_SHAPE_QUALITY_EXCLUSION.json'
 if marker.exists():return
 assert not list(RUN.glob('*/model/*_resume.pt')),'Never alter fit indices after optimizer updates'
 rows=[json.loads(s) for s in (RUN/'features/records.jsonl').read_text().splitlines()];split=json.loads((RUN/'source_split.json').read_text());exclude={i for i,r in enumerate(rows) if r['GT_voxels']==1};assert exclude
 assert not exclude&set(split['validation_rows']),'Validation is retained; a singleton there needs a separate sampling review'
 vanished=[i for i,r in enumerate(rows) if not r['normalized_target_voxels'] or not r['physical_target_voxels']];assert set(vanished)<=exclude,'A nonsingleton target vanished: stop and investigate sampling'
 original_split=RUN/'source_split_before_singleton_quality_exclusion.json';shutil.copy2(RUN/'source_split.json',original_split);shutil.copy2(RUN/'features/records.jsonl',RUN/'features/records_before_singleton_quality_exclusion.jsonl');oldcfg=sha256_file(RUN/'config.json')
 before={k:len(split[k]) for k in ['train_rows','validation_rows','final_rows']}
 for k in ['train_rows','final_rows']:split[k]=[i for i in split[k] if i not in exclude]
 split['shape_training_excluded_rows']=sorted(exclude);split['source_fit_cases']=sorted({rows[i]['case_id'] for i in split['train_rows']})
 for i,r in enumerate(rows):
  r['use_final']=i in set(split['final_rows'])
  if i in exclude:r['use_train']=False;r['shape_training_exclusion']='singleton component has no resolved 3D shape at proposed physical sampling; GT preserved'
 (RUN/'features/records.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n');write_json(RUN/'source_split.json',split)
 cfg=json.loads((RUN/'config.json').read_text());cfg['source_shape_quality_preflight_rule']={'rule':'exclude GT component size==1 from both training arms only; preserve original GT and all source validation','observed_vanished_crops':len(vanished),'excluded_crops':len(exclude),'no_optimizer_update_before_rule':True,'not_a_clinical_false_positive_adjudication':True};write_json(RUN/'config.json',cfg)
 ready=json.loads((RUN/'features/READY.json').read_text());ready['raw_train_rows_before_quality_exclusion']=ready['train_rows'];ready['train_rows']=len(split['train_rows']);ready['final_rows']=len(split['final_rows']);ready['records_sha256']=sha256_file(RUN/'features/records.jsonl');ready['source_split_sha256']=sha256_file(RUN/'source_split.json');ready['config_sha256']=sha256_file(RUN/'config.json');write_json(RUN/'features/READY.json',ready)
 write_json(marker,{'timestamp':time.time(),'before':before,'after':{k:len(split[k]) for k in before},'vanished_crops':vanished,'excluded_rows':sorted(exclude),'components':sorted({(rows[i]['case_id'],rows[i]['component_index'],rows[i]['GT_voxels']) for i in exclude}),'original_config_sha256':oldcfg,'source_only_pre_fit':True,'raw_arrays_and_GT_unchanged':True,'validation_rows_unchanged':True,'same_rule_both_arms':True,'clinical_annotation_error_not_established':True});print('E31 source shape quality exclusions',len(exclude),flush=True)
if __name__=='__main__':main()
