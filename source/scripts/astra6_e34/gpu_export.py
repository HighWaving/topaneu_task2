"""Preserve per-model resample-logits -> softmax -> ordered fp16 sum semantics.
Only interpolation/softmax executes on CUDA; accumulated probabilities stay on CPU.
"""
import numpy as np,torch
import torch.nn.functional as F
SLAB_BYTES=256*1024**2

def resample_slab(src,z0,z1,d_out,hw_out):
 C,d_in,h_in,w_in=src.shape
 if d_out==d_in:lerped=src[:,z0:z1].float()
 else:
  zc=(torch.arange(z0,z1,dtype=torch.float64,device=src.device)+.5)*(d_in/d_out)-.5;zf=torch.floor(zc);w=(zc-zf).float().view(1,-1,1,1);i0=zf.long().clamp_(0,d_in-1);i1=(zf.long()+1).clamp_(0,d_in-1);lerped=src[:,i0].float()*(1-w)+src[:,i1].float()*w
 if (h_in,w_in)==tuple(hw_out):return lerped.contiguous()
 nz=lerped.shape[1];return F.interpolate(lerped.reshape(1,C*nz,h_in,w_in),size=tuple(hw_out),mode='bilinear',antialias=False).reshape(C,nz,*hw_out)

@torch.inference_mode()
def streamed_gpu(sources,n_sources,pm,cm,lm,props,fp16=False):
 shape=[int(i) for i in props['shape_after_cropping_and_before_resampling']];d_out,h_out,w_out=shape;acc=None;seg=np.empty(shape,np.uint8)
 for src in sources:
  # Export never overlaps the next model's convolution; at most one source is on GPU.
  gpu=src.to('cuda:0');C=gpu.shape[0];nz=min(d_out,max(1,SLAB_BYTES//(C*h_out*w_out*4)))
  if n_sources>1 and acc is None:acc=torch.zeros((C,*shape),dtype=torch.float16 if fp16 else torch.float32)
  for z0 in range(0,d_out,nz):
   z1=min(z0+nz,d_out);slab=resample_slab(gpu,z0,z1,d_out,(h_out,w_out))
   if n_sources==1:seg[z0:z1]=torch.argmax(slab,dim=0).cpu().numpy().astype(np.uint8)
   else:acc[:,z0:z1]+=lm.apply_inference_nonlin(slab).to(dtype=acc.dtype,device='cpu')
   del slab
  del src,gpu
 if acc is not None:
  for z0 in range(0,d_out,nz):seg[z0:min(z0+nz,d_out)]=torch.argmax(acc[:,z0:min(z0+nz,d_out)],dim=0).numpy().astype(np.uint8)
  del acc
 full=np.zeros([int(i) for i in props['shape_before_cropping']],np.uint8);full[tuple(slice(b[0],b[1]) for b in props['bbox_used_for_cropping'])]=seg;return full.transpose(pm.transpose_backward)
