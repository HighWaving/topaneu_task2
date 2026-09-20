"""Lightweight optimizer-step evidence for the resumed full-budget detector run."""
import json,time,os
from pathlib import Path
import torch
import pytorch_lightning as pl
class Progress(pl.Callback):
 def __init__(self,path):self.path=Path(path);self.started=time.monotonic()
 def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx,dataloader_idx=0):
  if batch_idx%25:return
  loss=outputs.get('loss') if isinstance(outputs,dict) else outputs
  value=float(loss.detach().cpu()) if isinstance(loss,torch.Tensor) and loss.numel()==1 else None
  if value is not None:assert __import__('math').isfinite(value),'nonfinite training loss'
  row={'epoch_zero_based':int(trainer.current_epoch),'batch_index':int(batch_idx),'global_step':int(trainer.global_step),'loss':value,'seconds_since_resume':time.monotonic()-self.started,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'timestamp':time.time()};tmp=self.path.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2)+'\n');os.replace(tmp,self.path);print('E23_BATCH_PROGRESS',json.dumps(row),flush=True)
