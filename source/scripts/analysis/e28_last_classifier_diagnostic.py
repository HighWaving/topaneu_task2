"""One fixed last-checkpoint source diagnosis, separating NLL from decisions.

Never evaluates this unselected checkpoint on MR40 or changes epoch selection.
"""
import json
import numpy as np
import torch
from scripts.astra6_e28.common import RUN
from scripts.astra6_e28.representation import JointAnatomyLocation, refine_prediction
from scripts.astra6_e01.e01_common import write_json, sha256_file


def main():
    torch.set_num_threads(1); torch.backends.cudnn.benchmark = False
    assert (RUN / 'evaluation/DECISION.json').exists()
    source = json.loads((RUN / 'evaluation/SOURCE_RESULT.json').read_text())
    split = json.loads((RUN / 'source_split.json').read_text())
    records = [json.loads(s) for s in (RUN / 'features/records.jsonl').read_text().splitlines()]
    byrow = {r['row']: r for r in source['rows']}
    indices = split['validation_rows']
    arrays = {k: np.load(RUN / f'features/{k}.npy', mmap_mode='r') for k in ['local', 'context', 'shape']}
    checkpoint = RUN / 'model/development_last.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = JointAnatomyLocation().cuda().eval(); model.load_state_dict(saved['state_dict'])
    rows = []
    for start in range(0, len(indices), 4):
        ix = indices[start:start + 4]
        batch = {k: torch.from_numpy(np.array(a[ix], np.float32))[:, None].cuda() for k, a in arrays.items()}
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            logits = model(batch['local'], batch['context'], batch['shape'], False)['location_logits']
        probabilities = logits.float().softmax(1).cpu().numpy()
        for i, pp in zip(ix, probabilities):
            old = byrow[i]; baseline = old['baseline_prediction']; target = old['class']
            allowed = range(22, 36) if 22 <= baseline <= 35 else range(45, 53) if 45 <= baseline <= 52 else []
            allowed = [c for c in allowed if c % 2 == baseline % 2]
            loss = float(-np.log(max(float(pp[target - 1]) / max(float(pp[np.array(allowed) - 1].sum()), 1e-12), 1e-12))) if target in allowed else None
            rows.append({'row': i, 'case_id': records[i]['case_id'], 'class': target,
                         'baseline_prediction': baseline, 'selected_prediction': old['fine_refined_prediction'],
                         'last_prediction': refine_prediction(baseline, pp), 'last_conditional_CE': loss})
    eligible = [r for r in rows if r['last_conditional_CE'] is not None]
    cases = sorted({r['case_id'] for r in eligible})
    ce = float(np.mean([np.mean([r['last_conditional_CE'] for r in eligible if r['case_id'] == c]) for c in cases]))
    history = json.loads((RUN / 'model/development_history.json').read_text())
    assert abs(ce - history[-1]['source_case_CE']) < .01
    result = {'last_checkpoint_sha256': sha256_file(checkpoint), 'last_epoch': saved['epoch'],
              'selected_C_epoch': json.loads((RUN / 'model/LOCKED.json').read_text())['epochs'],
              'baseline_correct': source['baseline_correct'], 'selected_correct': source['new_correct'],
              'last_correct': sum(r['last_prediction'] == r['class'] for r in rows),
              'n': len(rows), 'last_case_conditional_CE': ce,
              'correct_rescued_vs_selected': sum(r['selected_prediction'] != r['class'] and r['last_prediction'] == r['class'] for r in rows),
              'correct_lost_vs_selected': sum(r['selected_prediction'] == r['class'] and r['last_prediction'] != r['class'] for r in rows),
              'rows': rows, 'no_MR40_inference_no_selection_change': True,
              'interpretation': 'Tests whether worsening source NLL also means worse deployed fine-class decisions. Confidence overfitting and decision accuracy can differ; this is one unselected last-checkpoint diagnostic, not a sweep or a replacement model.'}
    write_json(RUN / 'evaluation/LAST_CLASSIFIER_DIAGNOSTIC.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
