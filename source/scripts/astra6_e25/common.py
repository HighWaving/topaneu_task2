"""OOF image and predicted-shape filter. No label image is an input."""
from pathlib import Path
import torch
from torch import nn
P=Path(__file__).resolve().parents[2];RUN=P/'artifacts/astra6_e25_oof_shape_fp_filter_20260909';OOF=P/'artifacts/astra6_e23_MR_oof_candidates_20260909'
def encoder(cin):
 parts=[]
 for cout in [16,32,64]:parts.extend([nn.Conv3d(cin,cout,3,2,1,bias=False),nn.GroupNorm(8,cout),nn.SiLU()]);cin=cout
 return nn.Sequential(*parts,nn.AdaptiveAvgPool3d(1),nn.Flatten())
class Filter(nn.Module):
 def __init__(self):
  super().__init__();self.local=encoder(2);self.context=encoder(1);self.head=nn.Sequential(nn.Linear(128,32),nn.SiLU(),nn.Dropout(.25),nn.Linear(32,1))
 def forward(self,x):return self.head(torch.cat([self.local(x[:,[0,2]]),self.context(x[:,1:2])],dim=1)).squeeze(1)
