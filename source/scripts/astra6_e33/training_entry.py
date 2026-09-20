"""Fixed-budget scratch source267 D; protect the verified pre-SWA restart point."""
import sys,importlib.util,os,faulthandler,signal,json,time
from pathlib import Path
faulthandler.enable();faulthandler.register(signal.SIGUSR1,all_threads=True)
import torch,pytorch_lightning as pl
P=Path(__file__).resolve().parents[2];RUN=P/'artifacts/astra6_e33_source_only_detector_20260910';torch.set_num_threads(1);torch.set_num_interop_threads(1);fold=int(sys.argv[1]);args=sys.argv[2:];pl.seed_everything(20260910,workers=True);sys.path.insert(0,str(P))
from scripts.astra6_e23.progress_callback import Progress as BaseProgress
from scripts.astra6_e33.safe_swa_resume import snapshot
class Progress(BaseProgress):
 def on_train_epoch_start(self,trainer,pl_module):
  if trainer.current_epoch==48:
   result=snapshot(fold);print('E33_PROTECTED_PRE_SWA',json.dumps(result),flush=True)
 def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx,dataloader_idx=0):
  super().on_train_batch_end(trainer,pl_module,outputs,batch,batch_idx,dataloader_idx)
 def on_save_checkpoint(self,trainer,pl_module,checkpoint):
  checkpoint['E33_source_only_fit']={'source_ready':json.loads((RUN/'SOURCE_READY.json').read_text()),'full_budget_updates':45000,'seed':20260910,'validation_proxy_is_in_sample':True}
  return {}
original=pl.Trainer.__init__
def tracked(self,*args,**kwargs):
 kwargs['callbacks']=list(kwargs.get('callbacks',[]))+[Progress(RUN/'logs/batch_progress.json')];return original(self,*args,**kwargs)
pl.Trainer.__init__=tracked
source=P.parent/'external/nnDetection/scripts/train.py';spec=importlib.util.spec_from_file_location('topaneu_source_only_train',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);sys.argv=['nndet_train',args[0],'-o',*args[1:]];module.train()
