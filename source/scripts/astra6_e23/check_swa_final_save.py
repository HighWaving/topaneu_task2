"""Tiny CPU callback transaction test; not a medical-model experiment."""
import json,tempfile
from pathlib import Path
import torch,pytorch_lightning as pl
from torch.utils.data import DataLoader,TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from nndet.training.swa import SWACycleLinear
P=Path(__file__).resolve().parents[2]
class Tiny(pl.LightningModule):
 def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.tensor(.1));self.epoch_end_weights=[]
 def training_step(self,batch,batch_idx):return (self.w-1).square()
 def validation_step(self,batch,batch_idx):self.log('val_loss',(self.w-1).square())
 def on_train_epoch_end(self):self.epoch_end_weights.append(float(self.w.detach()))
 def configure_optimizers(self):
  o=torch.optim.SGD(self.parameters(),lr=.1);return {'optimizer':o,'lr_scheduler':{'scheduler':torch.optim.lr_scheduler.StepLR(o,step_size=100),'interval':'step'}}
 def configure_callbacks(self):return [SWACycleLinear(swa_epoch_start=3,cycle_initial_lr=.01,cycle_final_lr=.001,num_iterations_per_epoch=2)]
def main():
 torch.set_num_threads(1)
 with tempfile.TemporaryDirectory(prefix='e23_swa_final_save_check_') as td:
  m=Tiny();cb=ModelCheckpoint(dirpath=td,filename='best',save_last=True,monitor='val_loss',mode='min');loader=DataLoader(TensorDataset(torch.ones(2)),batch_size=1,num_workers=0)
  trainer=pl.Trainer(max_epochs=4,gpus=0,logger=False,callbacks=[cb],progress_bar_refresh_rate=0,weights_summary=None,num_sanity_val_steps=0);trainer.fit(m,loader,loader);saved=torch.load(Path(td)/'last.ckpt',map_location='cpu');out={'medical_training':False,'GPU_used':False,'steps':trainer.global_step,'epoch_end_unaveraged_weights':m.epoch_end_weights,'in_memory_final_weight':float(m.w.detach()),'saved_final_weight':float(saved['state_dict']['w']),'saved_epoch':int(saved['epoch']),'saved_step':int(saved['global_step']),'saved_matches_final_SWA':torch.equal(saved['state_dict']['w'],m.w.detach().cpu()),'callback_order':[type(x).__name__ for x in trainer.callbacks]}
 (P/'artifacts/research_audit_20260909/SWA_FINAL_SAVE_CHECK.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
