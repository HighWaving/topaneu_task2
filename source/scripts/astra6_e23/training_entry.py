"""Set a fixed fold seed, then use the installed nnDetection training entry."""
import sys,importlib.util,os,faulthandler,signal
faulthandler.enable()
faulthandler.register(signal.SIGUSR1,all_threads=True)
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from pathlib import Path
import pytorch_lightning as pl
fold=int(sys.argv[1]);args=sys.argv[2:];pl.seed_everything(20260909+fold,workers=True)
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.astra6_e23.progress_callback import Progress
_original_trainer_init=pl.Trainer.__init__
def _tracked_trainer_init(self,*args,**kwargs):
 kwargs['callbacks']=list(kwargs.get('callbacks',[]))+[Progress(Path(__file__).resolve().parents[2]/f'artifacts/astra6_e23_MR_oof_candidates_20260909/logs/fold{fold}_batch_progress.json')]
 return _original_trainer_init(self,*args,**kwargs)
pl.Trainer.__init__=_tracked_trainer_init
source=Path(__file__).resolve().parents[3]/'external/nnDetection/scripts/train.py';spec=importlib.util.spec_from_file_location('topaneu_oof_detector_train',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);sys.argv=['nndet_train',args[0],'-o',*args[1:]];module.train()
