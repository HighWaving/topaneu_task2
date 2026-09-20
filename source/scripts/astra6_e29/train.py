"""Matched joint control versus anatomy pretraining and frozen-encoder C fit."""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts.astra6_e29.common import P, RUN, PREVIOUS, SEED, arm_path
from scripts.astra6_e28.representation import JointAnatomyLocation, balanced_anatomy_ce, refine_prediction
from scripts.astra6_e01.e01_common import write_json, sha256_file
import scripts.astra6_e28.train as engine


def frozen_encoder_model(pretrained_state):
    """Transfer image/anatomy weights only; preserve fresh seeded C initialization."""
    model = JointAnatomyLocation()
    result = model.load_state_dict({k: v for k, v in pretrained_state.items()
                                   if not k.startswith('classifier.')}, strict=False)
    assert set(result.missing_keys) == {k for k in model.state_dict() if k.startswith('classifier.')}
    assert not result.unexpected_keys
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith('classifier.'))
    return model


def anatomy_fit(stage, indices, records):
    output = arm_path('staged') / 'anatomy'
    output.mkdir(parents=True, exist_ok=True)
    final_path = output / f'{stage}_last.pt'
    if final_path.exists():
        assert torch.load(final_path, map_location='cpu', weights_only=False)['epoch'] == 80
        return final_path
    torch.set_num_threads(2)
    random.seed(SEED + 1000); np.random.seed(SEED + 1000)
    torch.manual_seed(SEED + 1000); torch.cuda.manual_seed_all(SEED + 1000)
    torch.backends.cudnn.benchmark = False
    model = JointAnatomyLocation().cuda()
    for p in model.classifier.parameters():
        p.requires_grad_(False)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    generator = torch.Generator().manual_seed(SEED + 1000)
    loader = DataLoader(engine.Crops(indices, records), batch_size=4, shuffle=True,
                        generator=generator, num_workers=0, pin_memory=True)
    _, vessel_map = engine.mappings()
    assert all(records[i]['class'] < 0 and records[i]['kind'] == 'silver_anatomy_only' for i in indices)
    history, start = [], 1
    resume = output / f'{stage}_resume.pt'
    if resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['state_dict']); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); scaler.load_state_dict(saved['scaler'])
        history, start = saved['history'], saved['epoch'] + 1
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
        np.random.set_state(saved['numpy_rng']); random.setstate(saved['python_rng'])
        generator.set_state(saved['sampler_rng'])
    for epoch in range(start, 81):
        model.train(); begin = time.monotonic(); losses = []
        for batch, _, _ in loader:
            local = batch['local'][:, None].cuda().float()
            context = batch['context'][:, None].cuda().float()
            silver = batch['silver'].cuda().long()
            gamma = torch.empty((len(local), 1, 1, 1, 1), device='cuda').uniform_(.8, 1.25)
            local = (local.clamp(0, 1).pow(gamma) + .01 * torch.randn_like(local)).clamp(0, 1)
            context = (context.clamp(0, 1).pow(gamma) + .01 * torch.randn_like(context)).clamp(0, 1)
            if torch.rand((), device='cuda') < .5:
                local = local.flip(2); context = context.flip(2); silver = vessel_map[silver.flip(1)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(local, context, torch.zeros_like(local), True)['vessel_logits']
                loss = balanced_anatomy_ce(logits, silver)
            assert torch.isfinite(loss)
            scaler.scale(loss).backward()
            assert all(p.grad is None for p in model.classifier.parameters())
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(parameters, 5)
            scaler.step(optimizer); scaler.update(); losses.append(float(loss))
        scheduler.step()
        row = {'epoch': epoch, 'silver_anatomy_loss': float(np.mean(losses)),
               'seconds': time.monotonic() - begin, 'updates': len(loader),
               'C_targets_or_gradients': False, 'GPU_peak_bytes': torch.cuda.max_memory_allocated()}
        history.append(row)
        write_json(output / f'{stage}_history.json', history)
        engine.save(resume, {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                            'epoch': epoch, 'history': history, 'torch_rng': torch.get_rng_state(),
                            'cuda_rng': torch.cuda.get_rng_state_all(), 'numpy_rng': np.random.get_state(),
                            'python_rng': random.getstate(), 'sampler_rng': generator.get_state()})
        print('E29 anatomy', stage, json.dumps(row), flush=True)
    engine.save(final_path, {'state_dict': model.state_dict(), 'epoch': 80,
                            'training': 'fixed-budget image-only anatomy silver pretraining; no C targets'})
    return final_path


def main(name):
    assert (PREVIOUS / 'evaluation/DECISION.json').exists(), 'Finish E28 paired result before a new formal fit'
    assert json.loads((RUN / 'TRAINING_SEMANTICS_CHECK.json').read_text())['passed']
    arm = arm_path(name)
    if (arm / 'model/LOCKED.json').exists():
        from scripts.astra6_e29.anatomy_diagnostics import main as diagnose_completed_anatomy
        diagnose_completed_anatomy(name)
        return
    ready = json.loads((RUN / 'features/READY.json').read_text())
    assert ready['config_sha256'] == sha256_file(RUN / 'config.json')
    assert ready['source_split_sha256'] == sha256_file(RUN / 'source_split.json')
    assert ready['records_sha256'] == sha256_file(RUN / 'features/records.jsonl')
    records = [json.loads(s) for s in (RUN / 'features/records.jsonl').read_text().splitlines()]
    split = json.loads((RUN / 'source_split.json').read_text())
    assert not {records[i]['case_id'] for i in split['train_rows']} & {records[i]['case_id'] for i in split['validation_rows']}
    engine.RUN = arm; engine.SEED = SEED
    engine.JointAnatomyLocation = JointAnatomyLocation
    pretraining = {}
    if name == 'staged':
        from scripts.astra6_e29.anatomy_diagnostics import main as diagnose_anatomy
        diagnose_anatomy('joint')
        path = anatomy_fit('development', split['anatomy_train_rows'], records)
        pretraining['development'] = sha256_file(path)
        state = torch.load(path, map_location='cpu', weights_only=False)['state_dict']
        engine.JointAnatomyLocation = lambda: frozen_encoder_model(state)
    selected_epoch = engine.fit('development', split['train_rows'], records, split['validation_rows'])
    predictions = json.loads((arm / 'model/development_predictions.json').read_text())
    rows = []
    for pred in predictions:
        before = records[pred['row']]['source_baseline_class']
        after = refine_prediction(before, pred['probabilities'])
        rows.append({**pred, 'baseline_prediction': before, 'fine_refined_prediction': after})
    write_json(arm / 'evaluation/SOURCE_RESULT.json', {
        'arm': name, 'selected_epoch': selected_epoch, 'n_candidates': len(rows),
        'baseline_correct': sum(r['baseline_prediction'] == r['class'] for r in rows),
        'new_correct': sum(r['fine_refined_prediction'] == r['class'] for r in rows),
        'rows': rows, 'source_gate_requires_both_arms': True})
    if name == 'staged':
        path = anatomy_fit('final', split['anatomy_final_rows'], records)
        pretraining['final'] = sha256_file(path)
        state = torch.load(path, map_location='cpu', weights_only=False)['state_dict']
        engine.JointAnatomyLocation = lambda: frozen_encoder_model(state)
    engine.fit('final', split['final_rows'], records, epochs=selected_epoch)
    write_json(arm / 'model/LOCKED.json', {
        'arm': name, 'epochs': selected_epoch, 'full_formal_training_complete': True,
        'model_sha256': sha256_file(arm / 'model/final_last.pt'), 'anatomy_pretraining_sha256': pretraining,
        'config_sha256': sha256_file(RUN / 'config.json'),
        'source_split_sha256': sha256_file(RUN / 'source_split.json'),
        'training_wrapper_sha256': sha256_file(Path(__file__)),
        'classification_engine_sha256': sha256_file(Path(engine.__file__)),
        'no_MR40_CT5_fit': True, 'known_supervised_planning_exposure': True})
    print('E29 formal arm complete', name, selected_epoch, flush=True)
    if name == 'staged':
        diagnose_anatomy('staged')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--arm', choices=['joint', 'staged'], required=True)
    main(parser.parse_args().arm)
