"""Fresh source-only detector planning; never reuse legacy learned global plans."""
import json,pickle,hashlib,shutil,time,random
from pathlib import Path
import numpy as np,torch
from omegaconf import OmegaConf
from nndet.planning.experiment.v001 import D3V001
P=Path(__file__).resolve().parents[2];RUN=P/'artifacts/astra6_e33_source_only_detector_20260910';TASK='Task133FG_TopAneuMR_TrainOnly';PREP=RUN/'data'/TASK/'preprocessed'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);random.seed(20260910);np.random.seed(20260910);torch.manual_seed(20260910)
 RUN.mkdir(exist_ok=True);(PREP/'properties').mkdir(parents=True,exist_ok=True)
 if (RUN/'PLAN_READY.json').exists():return
 repair=P/'artifacts/train_only_planning_repair_20260909';summary=json.loads((repair/'PROPERTIES_READY.json').read_text());scope=summary['scopes']['final_source267'];source=repair/'final_source267/preprocessed/properties/dataset_properties.pkl';assert sha(source)==scope['dataset_properties_sha256'];props=pickle.loads(source.read_bytes());cases=sorted(props['instance_props_per_patient']);assert cases==scope['cases'] and len(cases)==267;assert not set(cases)&set(scope['comparison_excluded']);assert props['intensity_properties'] is None
 dest=PREP/'properties/dataset_properties.pkl'
 if dest.exists():assert sha(dest)==sha(source)
 else:shutil.copyfile(source,dest)
 oldcfg=P/'artifacts/astra6_e23_MR_oof_candidates_20260909/models/Task130FG_TopAneuMR_OOF/RetinaUNetV001_D3V001_3d/fold0/config.yaml';cfg=OmegaConf.load(oldcfg);model_cfg=OmegaConf.to_container(cfg.model_cfg,resolve=True)
 contract={'experiment':'E33','stage':'fresh_source_only_planning','timestamp':time.time(),'hypothesis':'Full source267 scratch detector with source-only geometry/anchors removes known MR40 supervised planning exposure; evaluate proposal recall and downstream distribution shift separately from S/F/C changes. No guarantee of improvement.','scope':scope,'budget':{'main_epochs':50,'SWA_epochs':10,'updates_per_epoch':750,'total_updates':45000,'validation_batches':100,'checkpoint':'each epoch with pre-SWA recovery archive; fixed final epoch60, no MR40 checkpoint selection'},'comparison':'First compare D replacement with E17 C02/S17/F14 fixed, retain old D outputs and r2. E32 result evaluated separately; no simultaneous D+S adoption.','initialization':'scratch only; planner synthetic memory-measurement weights discarded','limits':['MR40 already repeatedly observed; source-only planning repair does not make it untouched validation.','TA36 upstream training lineage unresolved.','Final267 detector is deployment candidate, not OOF source-data generator. Existing E23 outputs retain legacy planning limitation.','F14 training uses old detector outputs; new-D distribution mismatch must be reported.'],'model_cfg':model_cfg,'planner_properties_sha256':sha(dest),'old_config_use':'fixed hyperparameters only; no old plan, anchors, global statistics or weights loaded'}
 if not (RUN/'PROTOCOL_BEFORE_PLANNING.json').exists():(RUN/'PROTOCOL_BEFORE_PLANNING.json').write_text(json.dumps(contract,indent=2)+'\n')
 if (PREP/'D3V001_3d.pkl').exists():
  saved=pickle.loads((PREP/'D3V001_3d.pkl').read_bytes());plans=['D3V001_3d']
  if saved.get('trigger_lr1'):assert (PREP/'D3V001_3dlr1.pkl').exists();plans.append('D3V001_3dlr1')
 else:
  planner=D3V001(PREP);plans=planner.plan_experiment(model_name='RetinaUNetV001',model_cfg=model_cfg)
 result=[]
 for name in plans:
  file=PREP/(name if name.endswith('.pkl') else name+'.pkl');plan=pickle.loads(file.read_bytes());assert sorted(plan['dataset_properties']['instance_props_per_patient'])==cases;assert plan['dataset_properties']['intensity_properties'] is None
  result.append({'name':name,'sha256':sha(file),'patch_size':np.asarray(plan['patch_size']).tolist(),'target_spacing':np.asarray(plan['target_spacing']).tolist(),'batch_size':plan['batch_size'],'anchors':plan['anchors']})
 (RUN/'PLAN_READY.json').write_text(json.dumps({'timestamp':time.time(),'source_cases':267,'excluded_comparison_cases':scope['comparison_excluded'],'plans':result,'properties_sha256':sha(dest),'no_legacy_supervised_plan_loaded':True},indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n');print('E33 PLAN READY',result,flush=True)
if __name__=='__main__':main()
