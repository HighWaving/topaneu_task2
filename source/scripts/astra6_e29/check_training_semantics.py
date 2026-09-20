"""Verify transfer/freeze behavior before formal E29 training."""
import torch
from scripts.astra6_e29.common import RUN
from scripts.astra6_e29.train import frozen_encoder_model
from scripts.astra6_e28.representation import JointAnatomyLocation, balanced_anatomy_ce
from scripts.astra6_e01.e01_common import write_json


def main():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    anatomy = JointAnatomyLocation()
    for p in anatomy.classifier.parameters():
        p.requires_grad_(False)
    local, context, shape = [torch.rand(2, 1, 16, 16, 16) for _ in range(3)]
    logits = anatomy(local, context, torch.zeros_like(shape), True)['vessel_logits']
    loss = balanced_anatomy_ce(logits, torch.randint(0, 37, (2, 16, 16, 16)))
    loss.backward()
    assert all(p.grad is None for p in anatomy.classifier.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in anatomy.local_deep.parameters())
    pretrained = {k: v.detach().clone() for k, v in anatomy.state_dict().items()}
    for k in pretrained:
        if k.startswith('classifier.'):
            pretrained[k].fill_(7)
    torch.manual_seed(20260910)
    control = JointAnatomyLocation()
    torch.manual_seed(20260910)
    staged = frozen_encoder_model(pretrained)
    for k, v in staged.state_dict().items():
        assert torch.equal(v, control.state_dict()[k] if k.startswith('classifier.') else pretrained[k])
    before = {k: v.detach().clone() for k, v in staged.state_dict().items()}
    optimizer = torch.optim.AdamW(staged.parameters(), lr=3e-4, weight_decay=.01)
    loss = torch.nn.functional.cross_entropy(staged(local, context, shape, False)['location_logits'], torch.tensor([23, 46]))
    loss.backward()
    assert all(p.grad is None for n, p in staged.named_parameters() if not n.startswith('classifier.'))
    optimizer.step()
    assert all(torch.equal(v, before[k]) for k, v in staged.state_dict().items() if not k.startswith('classifier.'))
    assert any(not torch.equal(v, before[k]) for k, v in staged.state_dict().items() if k.startswith('classifier.'))
    write_json(RUN / 'TRAINING_SEMANTICS_CHECK.json', {
        'passed': True, 'CPU_only': True, 'synthetic_data_only': True,
        'anatomy_phase_no_classifier_gradient': True, 'anatomy_gradient_reaches_deep_encoder': True,
        'fresh_classifier_matches_control_initialization': True,
        'encoder_exactly_unchanged_after_C_optimizer_step': True,
        'classifier_changes_after_C_optimizer_step': True,
        'formal_training_updates': 0,
        'geometry_and_actual_GPU_architecture_preflight': 'Inherited unchanged E28 representation and verified image arrays; E28 actual batch4 GPU preflight passed.'})
    print('E29 training semantics passed', flush=True)


if __name__ == '__main__':
    main()
