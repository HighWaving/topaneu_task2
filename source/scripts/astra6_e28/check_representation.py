"""CPU-only meaningful geometry and forward/backward checks; no model fit."""
from pathlib import Path
import json,os,time
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np,torch
from scripts.astra6_e28.representation import ras_grid,resample_ras,shape_to_ras,JointAnatomyLocation,training_loss
P=Path(__file__).resolve().parents[2];R=P/'artifacts/astra6_e28_joint_anatomy_location_20260909';torch.set_num_threads(1);torch.manual_seed(20260909);start=time.time()
# Rotated, anisotropic, translated image: resampled world coordinate ramp
# must reproduce the analytic RAS ramp, catching index/world-axis mistakes.
aff=np.array([[0,-.7,0,40],[.4,0,0,-20],[0,0,1.3,5],[0,0,0,1.]])
shape=(96,96,96);idx=np.indices(shape);arr=(aff[0,:3]@idx.reshape(3,-1)+aff[0,3]).reshape(shape).astype(np.float32);center=np.array([48,48,48]);crop=resample_ras(arr,aff,center,12,24);world_center=aff[:3,:3]@center+aff[:3,3];expected=np.broadcast_to(world_center[0]+((np.arange(24)-11.5)*.5)[:,None,None],crop.shape);assert np.allclose(crop,expected,atol=1e-4)
prob=shape_to_ras(np.ones((32,32,32),np.float32),center-3,center+3,aff);assert 0<prob.sum()<prob.size and np.isfinite(prob).all()
model=JointAnatomyLocation();local=torch.randn(1,1,48,48,48);context=torch.randn(1,1,64,64,64);mask=torch.from_numpy(prob.astype(np.float32))[None,None];out=model(local,context,mask);assert out['location_logits'].shape==(1,52) and out['vessel_logits'].shape==(1,37,48,48,48)
silver=torch.zeros((1,48,48,48),dtype=torch.long);silver[:,20:28,20:28,:]=34;loss,terms=training_loss(out,torch.tensor([25]),silver);loss.backward();assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());assert model.vessel_head.weight.grad.abs().sum()>0 and model.local_stem[0].weight.grad.abs().sum()>0
# Anatomy-only source cases must actually train the deep/local and broad
# context image branches; the auxiliary prediction must not copy shape input.
model.zero_grad(set_to_none=True);anatomy_out=model(local,context,mask);anatomy_loss,_=training_loss(anatomy_out,torch.tensor([-1]),silver);anatomy_loss.backward();assert model.local_deep[0][0].weight.grad.abs().sum()>0 and model.context[0][0].weight.grad.abs().sum()>0
model.eval()
with torch.inference_mode():assert torch.equal(model(local,context,mask)['vessel_logits'],model(local,context,torch.zeros_like(mask))['vessel_logits'])
# Absent/uncertain shape must remain numerically valid without GT fallback.
with torch.inference_mode():assert torch.isfinite(model(local,context,torch.zeros_like(mask),False)['location_logits']).all()
result={'status':'CPU representation checks passed; no optimizer steps, no trained model','analytic_RAS_geometry':True,'shape_probability_support':True,'finite_multitask_forward_backward':True,'anatomy_only_gradient_reaches_deep_and_context':True,'anatomy_prediction_independent_of_shape_input':True,'empty_predicted_shape_supported':True,'parameters':sum(p.numel() for p in model.parameters()),'local':list(local.shape),'context':list(context.shape),'seconds':time.time()-start,'GPU_memory_not_yet_verified':True};(R/'model/REPRESENTATION_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
