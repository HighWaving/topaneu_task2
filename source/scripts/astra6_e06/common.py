from pathlib import Path
import numpy as np,torch
from torch import nn
from scripts.astra6_e01.e01_common import P,DATA
from scripts.astra6_e03.run_e03 import load_image,sample_crop
from scripts.astra6_e04.run_e04 import crop_volume
RUN=P/'artifacts/astra6_e06_image_fp_filter_20260909'
SEED=20260909

def crops(arr,aff,lo,hi):
 return np.stack([crop_volume(arr,lo,hi,1),sample_crop(arr,aff,lo,hi)[0]]).astype(np.float32)
class Filter(nn.Module):
 def __init__(self):
  super().__init__();parts=[];cin=1
  for c in [16,32,64]:parts.extend([nn.Conv3d(cin,c,3,2,1,bias=False),nn.GroupNorm(8,c),nn.SiLU()]);cin=c
  self.encoder=nn.Sequential(*parts,nn.AdaptiveAvgPool3d(1),nn.Flatten());self.head=nn.Sequential(nn.Linear(128,32),nn.SiLU(),nn.Dropout(.25),nn.Linear(32,1))
 def forward(self,x):return self.head(self.encoder(x.reshape(-1,1,32,32,32)).reshape(x.shape[0],128)).squeeze(1)

def load_filter(weights,device='cpu'):
 checkpoint=torch.load(weights,map_location='cpu',weights_only=False);architecture=checkpoint.get('architecture')
 if architecture=='E14_mean3':
  from scripts.astra6_e14.common import Filter as EnsembleFilter
  model=EnsembleFilter()
 elif architecture in (None,'E06_shared_multiscale_CNN'):model=Filter()
 else:raise ValueError(f'Unsupported image filter architecture: {architecture}')
 model.load_state_dict(checkpoint['state_dict']);return model.to(device).eval()
