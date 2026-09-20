"""CPU statistical and gradient checks of branch supervision; no fitting."""
import json
from pathlib import Path
import numpy as np
import torch
from scripts.astra6_e28.representation import anatomy_center_indices, balanced_anatomy_ce

def main():
    torch.set_num_threads(1)
    labels=np.array([1]*99+[33])
    rng=np.random.default_rng(20260910)
    draws=np.array([anatomy_center_indices(labels,rng) for _ in range(4000)])
    rare_rates=(labels[draws]==33).mean(0)
    assert .002<rare_rates[0]<.025 and .45<rare_rates[1]<.55,rare_rates
    assert anatomy_center_indices([],rng)==[]
    assert len(anatomy_center_indices([33],rng))==2
    assert anatomy_center_indices(labels,np.random.default_rng(7))==anatomy_center_indices(labels,np.random.default_rng(7))
    target=torch.tensor([1]*81+[33]*9+[0]*10).reshape(1,1,1,100)
    logits=torch.zeros((1,37,1,1,100),requires_grad=True)
    loss=balanced_anatomy_ce(logits,target)
    assert torch.allclose(loss,torch.tensor(np.log(37),dtype=torch.float32),atol=1e-6)
    loss.backward()
    common_mass=logits.grad[0,1,0,0,:81].abs().sum()
    rare_mass=logits.grad[0,33,0,0,81:90].abs().sum()
    ratio=float(common_mass/rare_mass)
    assert abs(ratio-3)<1e-5,(ratio,'sqrt class-count mass expected; old voxel mass ratio=9')
    for label in (0,33):
        x=torch.zeros((1,37,2,2,2),requires_grad=True)
        value=balanced_anatomy_ce(x,torch.full((1,2,2,2),label))
        value.backward()
        assert torch.isfinite(value) and torch.isfinite(x.grad).all()
    result={'no_optimizer_steps':True,'synthetic_1pct_label_center_rates':rare_rates.tolist(),
            'gradient_class_mass_ratio_81_to_9_voxels':ratio,
            'expected_sqrt_mass_ratio':3.,'empty_and_single_label_sampling':True,
            'all_background_and_all_foreground_finite':True}
    p=Path(__file__).resolve().parents[2]/'artifacts/review_current_model_20260910/ANATOMY_BALANCE_CHECK.json'
    p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
