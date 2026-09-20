"""Image/aneurysm joint representation, no ground-truth vessel inference input."""
import numpy as np
from scipy.ndimage import map_coordinates
import torch
from torch import nn
from torch.nn import functional as F

def anatomy_center_indices(foreground_labels, rng):
 """Two centers: one voxel-proportional, one uniform over present labels."""
 labels=np.asarray(foreground_labels)
 if not len(labels):return []
 assert np.all((labels>0)&(labels<=36))
 voxel_index=int(rng.integers(len(labels)))
 selected_label=int(rng.choice(np.unique(labels)))
 label_indices=np.flatnonzero(labels==selected_label)
 return [voxel_index,int(label_indices[rng.integers(len(label_indices))])]

def balanced_anatomy_ce(logits, silver):
 """Half background, half foreground with sqrt-count label mass.

 Each foreground voxel gets inverse-sqrt class frequency. Unlike a full
 class-macro mean, a tiny noisy silver label does not get equal total mass.
 """
 ce=F.cross_entropy(logits,silver.long(),reduction='none');fg=silver>0
 if fg.any():
  counts=torch.bincount(silver[fg].long(),minlength=logits.shape[1]).float()
  weights=counts.clamp_min(1).rsqrt()[silver[fg].long()]
  foreground=(ce[fg].float()*weights).sum()/weights.sum()
 else:foreground=ce.sum()*0
 background=ce[~fg].mean() if (~fg).any() else ce.sum()*0
 return .5*(foreground+background)

def ras_grid(affine,center_native,width_mm,resolution):
 center=np.asarray(affine)[:3,:3]@np.asarray(center_native)+np.asarray(affine)[:3,3]
 world=np.stack(np.meshgrid(*[center[i]+(np.arange(resolution)-(resolution-1)/2)*width_mm/resolution for i in range(3)],indexing='ij'))
 inv=np.linalg.inv(affine)
 return (inv[:3,:3]@world.reshape(3,-1)+inv[:3,3,None]).reshape(3,resolution,resolution,resolution)
def resample_ras(array,affine,center_native,width_mm,resolution,order=1):
 grid=ras_grid(affine,center_native,width_mm,resolution)
 return map_coordinates(array,grid.reshape(3,-1),order=order,mode='constant',cval=0,prefilter=False).reshape(resolution,resolution,resolution)
def shape_to_ras(probability,low,high,affine,width_mm=24,resolution=48):
 low,high=np.asarray(low),np.asarray(high);center=(low+high)/2;grid=ras_grid(affine,center,width_mm,resolution);extent=2*np.maximum(high-low,1)
 coords=(grid-center[:,None,None,None])/extent[:,None,None,None]*32+15.5
 # Shape helper probabilities are restricted to original candidate support,
 # matching deployed S; probabilities outside the box are not anatomy evidence.
 support=np.all(np.abs(grid-center[:,None,None,None])<=np.maximum(high-low,1)[:,None,None,None]/2,axis=0)
 return map_coordinates(probability,coords.reshape(3,-1),order=1,mode='constant',cval=0,prefilter=False).reshape(resolution,resolution,resolution)*support

def block(a,b,stride=1):
 return nn.Sequential(nn.Conv3d(a,b,3,stride,1,bias=False),nn.GroupNorm(4,b),nn.SiLU(),nn.Conv3d(b,b,3,1,1,bias=False),nn.GroupNorm(4,b),nn.SiLU())
class JointAnatomyLocation(nn.Module):
 """Dense anatomy supervision precedes shape fusion; no label autoencoding."""
 def __init__(self):
  super().__init__();self.local_stem=block(1,16);self.local_deep=nn.Sequential(block(16,32,2),block(32,64,2));self.vessel_head=nn.Conv3d(16,37,1);self.anatomy_deep=nn.Conv3d(64,16,1);self.anatomy_context=nn.Linear(64*8,16);self.anatomy_decoder=block(48,16)
  self.context=nn.Sequential(block(1,16,2),block(16,32,2),block(32,64,2),nn.AdaptiveAvgPool3d(2),nn.Flatten());self.classifier=nn.Sequential(nn.Linear(64*3+64*8,128),nn.SiLU(),nn.Dropout(.25),nn.Linear(128,52))
 @staticmethod
 def pool(features,weight):
  return (features*weight).sum((2,3,4))/weight.sum((2,3,4)).clamp_min(1e-6)
 def forward(self,local_image,context_image,aneurysm_probability,return_anatomy=True):
  stem=self.local_stem(local_image);features=self.local_deep(stem);core=F.interpolate(aneurysm_probability,size=features.shape[-3:],mode='trilinear',align_corners=False).clamp(0,1);ring=(F.max_pool3d(core,3,1,1)-core).clamp(0,1)
  # Core/perilesional/global pools preserve distinct lesion/parent context.
  context_features=self.context(context_image)
  joined=torch.cat([self.pool(features,core),self.pool(features,ring),features.mean((2,3,4)),context_features],dim=1)
  result={'location_logits':self.classifier(joined)}
  if return_anatomy:
   # Semantic artery supervision must reach deep and broad-context image
   # features, not only a shallow local texture stem. No shape/label input.
   deep=F.interpolate(self.anatomy_deep(features),size=stem.shape[-3:],mode='trilinear',align_corners=False);context_condition=self.anatomy_context(context_features)[:,:,None,None,None].expand(-1,-1,*stem.shape[-3:]);result['vessel_logits']=self.vessel_head(self.anatomy_decoder(torch.cat([stem,deep,context_condition],dim=1)))
  return result

def training_loss(output,location_class,vessel_silver,class_weights=None,anatomy_weight=.2):
 """location_class uses1..52; -1 for anatomy-only source crops."""
 valid=location_class>0
 classification=F.cross_entropy(output['location_logits'][valid],location_class[valid]-1,weight=class_weights) if valid.any() else output['location_logits'].sum()*0
 # Treat organizer predictions as noisy supervision, not ground truth.
 anatomy=balanced_anatomy_ce(output['vessel_logits'],vessel_silver)
 return classification+anatomy_weight*anatomy,{'classification':classification.detach(),'silver_anatomy':anatomy.detach()}

def refine_prediction(baseline_class,probabilities):
 """Keep effective C02 artery family/laterality; learn ICA/MCA subdivision."""
 candidates=range(22,36) if 22<=baseline_class<=35 else range(45,53) if 45<=baseline_class<=52 else []
 candidates=[c for c in candidates if c%2==baseline_class%2]
 return max(candidates,key=lambda c:probabilities[c-1]) if candidates else baseline_class
