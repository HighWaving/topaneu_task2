import torch
from torch import nn
from scripts.astra6_e06.common import Filter as SingleFilter
from scripts.astra6_e01.e01_common import P
RUN=P/'artifacts/astra6_e14_MR_filter_seed_stability_20260909'
class Filter(nn.Module):
 def __init__(self):
  super().__init__();self.models=nn.ModuleList([SingleFilter() for _ in range(3)])
 def forward(self,x):
  mean=torch.stack([m(x).sigmoid() for m in self.models]).mean(0);return torch.logit(mean.clamp(1e-7,1-1e-7))
