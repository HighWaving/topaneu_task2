"""Official six paired contrasts and predeclared candidate gates; no score mean."""
import json
import numpy as np
from scripts.astra6_e29.common import P, RUN, SEED, arm_path
from scripts.analysis.current_official_evaluation import aggregate, write

METRICS = ['PRECISION', 'RECALL', 'MCC', 'DICE', 'VOLSIM', 'HD95']


def pareto(delta):
    return all(delta[k] >= -1e-10 for k in METRICS[:-1]) and delta['HD95'] <= 1e-10 and any(abs(v) > 1e-8 for v in delta.values())


def main():
    anatomy = {n: json.loads((arm_path(n) / 'evaluation/SOURCE_ANATOMY_DIAGNOSTICS.json').read_text()) for n in ['joint', 'staged']}
    anatomy_comparison = {}
    for pool in anatomy['joint']['groups']:
        a = anatomy['joint']['groups'][pool]['per_class']
        b = anatomy['staged']['groups'][pool]['per_class']
        assert all(a[k]['reference_voxels'] == b[k]['reference_voxels'] for k in a)
        keys = [k for k in a if a[k]['reference_voxels'] > 0]
        anatomy_comparison[pool] = {
            'joint_reference_present_class_macro_pooled_Dice': float(np.mean([a[k]['Dice'] for k in keys])),
            'staged_reference_present_class_macro_pooled_Dice': float(np.mean([b[k]['Dice'] for k in keys])),
            'per_class': {k: {'name': a[k]['name'], 'reference_voxels': a[k]['reference_voxels'],
                              'joint_Dice': a[k]['Dice'], 'staged_Dice': b[k]['Dice']} for k in a}}
    freeze = json.loads((arm_path('staged') / 'evaluation/ACTUAL_ENCODER_FREEZE_CHECK.json').read_text())
    assert freeze['passed']
    write(RUN / 'ANATOMY_COMPARISON.json', {'groups': anatomy_comparison, 'actual_encoder_freeze_check': freeze,
          'interpretation': 'Same new source cases and silver targets. Diagnostic of anatomy capability, not true-vessel accuracy or an official Task2 metric; not a checkpoint selection criterion.'})
    root = P / 'artifacts/current_official_20260909'
    names = ['E17', 'E29_joint', 'E29_staged']
    official = {n: json.loads((root / n / 'official.json').read_text())['overall'] for n in names}
    cases = {n: json.loads((root / n / 'per_case.json').read_text()) for n in names}
    ids = [r['case_id'] for r in cases['E17']]
    assert all([r['case_id'] for r in cases[n]] == ids for n in names)
    pairs = [('E29_joint', 'E17'), ('E29_staged', 'E17'), ('E29_staged', 'E29_joint')]
    samples = {pair: [] for pair in pairs}
    rng = np.random.default_rng(SEED)
    for _ in range(2000):
        ix = rng.integers(0, len(ids), len(ids))
        values = {n: aggregate([cases[n][i]['raw'] for i in ix])['overall'] for n in names}
        for pair in pairs:
            a, b = pair
            samples[pair].append([values[a][k] - values[b][k] for k in METRICS])
    comparisons = {}
    for a, b in pairs:
        delta = {k: official[a][k] - official[b][k] for k in METRICS}
        comparisons[a + '_vs_' + b] = {
            'before': official[b], 'after': official[a], 'delta': delta,
            'paired_bootstrap_95CI': {k: np.percentile(np.asarray(samples[(a, b)])[:, j], [2.5, 97.5]).tolist() for j, k in enumerate(METRICS)},
            'pareto_gate': pareto(delta), 'n_bootstrap': 2000, 'seed': SEED}
    source = {n: json.loads((arm_path(n) / 'evaluation/SOURCE_RESULT.json').read_text()) for n in ['joint', 'staged']}
    assert source['joint']['n_candidates'] == source['staged']['n_candidates']
    assert [r['row'] for r in source['joint']['rows']] == [r['row'] for r in source['staged']['rows']]
    assert source['joint']['baseline_correct'] == source['staged']['baseline_correct']
    base_correct = source['joint']['baseline_correct']
    source_gate = {
        'joint': source['joint']['n_candidates'] >= 20 and source['joint']['new_correct'] >= base_correct + 2,
        'staged': source['staged']['n_candidates'] >= 20 and source['staged']['new_correct'] >= max(base_correct, source['joint']['new_correct']) + 2}
    baseline = json.loads((P / 'artifacts/astra6_e17_MR_detector_crop_segmentation_20260909/evaluation/lesion_diagnostics.json').read_text())['E17']
    qualified = []
    for arm in ['joint', 'staged']:
        name = 'E29_' + arm
        d = json.loads((arm_path(arm) / 'evaluation/lesion_diagnostics.json').read_text())
        comparison = comparisons[name + '_vs_E17']
        adopt = bool(source_gate[arm] and comparison['pareto_gate'] and d[name]['matched'] >= baseline['matched'] and d[name]['location_correct'] > baseline['location_correct'])
        if adopt:
            qualified.append(name)
        write(arm_path(arm) / 'evaluation/paired_official_comparison.json', comparison)
        write(arm_path(arm) / 'evaluation/DECISION.json', {
            'decision': 'development_candidate_pending_clean_validation_and_runtime' if adopt else 'not_adopted',
            'source_gate': source_gate[arm], 'official_pareto_gate': comparison['pareto_gate'],
            'location_rescued': d['location_rescued'], 'location_lost': d['location_lost'],
            'r2_unchanged': True, 'known_planning_exposure_and_repeated_MR40_research': True})
    selected = qualified[0] if len(qualified) == 1 else (
        'E29_staged' if len(qualified) == 2 and comparisons['E29_staged_vs_E29_joint']['pareto_gate'] else None)
    write(RUN / 'COMPARISONS.json', {
        'comparisons': comparisons,
        'source': {n: {k: v for k, v in s.items() if k != 'rows'} for n, s in source.items()},
        'source_gate': source_gate,
        'interpretation': 'Matched new source split; staged adds anatomy pretraining compute. D/S/F and native foreground fixed. MR40 has legacy GT planning exposure and repeated research use; no independent-generalization or ranking claim. No raw-six average.'})
    write(RUN / 'DECISION.json', {
        'qualified_development_candidates': qualified, 'selected_development_candidate': selected,
        'retain_E17_until_verified': True,
        'next_step': 'If qualified, verify raw pipeline/runtime and create a fresh candidate release. Otherwise diagnose anatomy agreement, source fit and MR40 transfer; continue the most supported capability change. Known planning repair, CT and T4 constraints remain.'})
    print(json.dumps({'qualified': qualified, 'selected': selected, 'source_gate': source_gate}, indent=2), flush=True)


if __name__ == '__main__':
    main()
