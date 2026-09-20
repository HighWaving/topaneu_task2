"""Saved-artifact comparison of anatomy learning and C selection timing."""
from pathlib import Path
import hashlib
import json
import statistics

P = Path(__file__).resolve().parents[2]
RUN = P / 'artifacts/astra6_e28_joint_anatomy_location_20260909'


def main():
    paths = [RUN / 'evaluation' / name for name in
             ['SOURCE_ANATOMY_DIAGNOSTICS.json', 'SOURCE_ANATOMY_LAST_DIAGNOSTICS.json']]
    selected, last = [json.loads(p.read_text()) for p in paths]
    history = json.loads((RUN / 'model/development_history.json').read_text())
    groups = {}
    for group in selected['groups']:
        a = selected['groups'][group]['per_class']
        b = last['groups'][group]['per_class']
        assert all(a[k]['reference_voxels'] == b[k]['reference_voxels'] for k in a)
        keys = [k for k in a if a[k]['reference_voxels'] > 0]
        groups[group] = {
            'reference_present_classes': len(keys),
            'pooled_reference_present_class_macro_Dice': {
                'selected': statistics.mean(a[k]['Dice'] for k in keys),
                'last': statistics.mean(b[k]['Dice'] for k in keys)},
            'OA_AChA': {k: {'name': a[k]['name'], 'reference_voxels': a[k]['reference_voxels'],
                           'selected_Dice': a[k]['Dice'], 'last_Dice': b[k]['Dice'],
                           'selected_case_mean_Dice': a[k]['case_mean_Dice'],
                           'last_case_mean_Dice': b[k]['case_mean_Dice']}
                        for k in ['31', '32', '33', '34']},
        }
    result = {
        'input_sha256': {str(p.relative_to(P)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'selected_C_epoch': selected['selected_C_epoch'],
        'last_development_epoch': last['checkpoint_epoch'],
        'selected_training_history': history[selected['selected_C_epoch'] - 1],
        'last_training_history': history[-1],
        'groups': groups,
        'interpretation': 'Anatomy silver agreement improves substantially after the C-selected early epoch while held-out C loss worsens. This supports testing anatomy pretraining before C fitting, rather than treating the image representation as incapable. It does not prove staged training will improve Task2.',
        'constraints': ['No checkpoint substitution, threshold change or additional model selection.',
                        'Silver agreement is not true-vessel accuracy or an official Task2 score.',
                        'Same source C-selection cases are reused diagnostically, not independent validation.',
                        'A staged-training experiment should use a separately declared source partition with a matched joint-training control, then fixed-upstream E2E comparison.'],
    }
    followup = RUN / 'evaluation/LAST_CLASSIFIER_DIAGNOSTIC.json'
    if followup.exists():
        result['decision_accuracy_followup'] = json.loads(followup.read_text())
        result['input_sha256'][str(followup.relative_to(P))] = hashlib.sha256(followup.read_bytes()).hexdigest()
        result['interpretation'] = 'Anatomy silver agreement improves after the C-selected early epoch while held-out C NLL worsens. Last-source fine accuracy14/20 versus selected13/20 equals geometry baseline14/20; worsening NLL does not establish worsening decision accuracy. Test anatomy maturity, confidence/selection mismatch and small-source instability; staged/frozen training is not guaranteed to improve Task2.'
    out = P / 'artifacts/review_current_model_20260910/E28_LEARNING_TIMING.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v['pooled_reference_present_class_macro_Dice'] for k, v in groups.items()}))


if __name__ == '__main__':
    main()
