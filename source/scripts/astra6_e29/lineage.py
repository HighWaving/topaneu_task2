"""Actual case roles and learned-upstream gradient exclusions for E29."""
import hashlib
import json
import re
from scripts.astra6_e29.common import P, RUN


def group(cid):
    return re.sub(r'(_(?:mr|ct)_\d+)_\d+$', r'\1', cid)


def main():
    config = json.loads((RUN / 'config.json').read_text())
    records = [json.loads(s) for s in (RUN / 'features/records.jsonl').read_text().splitlines()]
    part = config['source_partition']
    fit, dev = set(part['C_fit_cases']), set(part['epoch_selection_cases'])
    dev_groups = {group(c) for c in dev}
    oof = P / 'artifacts/astra6_e23_MR_oof_candidates_20260909'
    folds = json.loads((oof / 'source_split.json').read_text())['folds']
    helpers = {f: json.loads((oof / f'segmenters/fold{f}/source_split.json').read_text()) for f in [0, 1]}
    assert not dev_groups & {group(c) for c in folds[1]['train']}
    assert not dev_groups & {group(c) for c in helpers[1]['training_cases']}
    result = []
    for cid in sorted({r['case_id'] for r in records}):
        rows = [r for r in records if r['case_id'] == cid]
        fold = rows[0]['OOF_fold']
        assert all(r['OOF_fold'] == fold for r in rows)
        d_seen = group(cid) in {group(c) for c in folds[fold]['train']}
        s_seen = group(cid) in {group(c) for c in helpers[fold]['training_cases']}
        assert not d_seen and not s_seen
        components = {r['component_index'] for r in rows if r['class'] > 0}
        result.append({'case_id': cid, 'available_patient_group': group(cid),
                       'OOF_fold': fold, 'own_D_gradient_fit': d_seen, 'own_S_helper_gradient_fit': s_seen,
                       'C_development_fit_eligible': cid in fit,
                       'C_development_supervised_components': len(components) if cid in fit else 0,
                       'source_selection_case': cid in dev,
                       'anatomy_development_fit': cid not in dev,
                       'final_anatomy_fit': True, 'final_C_supervised_components': len(components),
                       'anatomy_only_rows': sum(r['class'] < 0 for r in rows),
                       'GT_centered_training_rows': sum(r['kind'] == 'GT_training_crop' for r in rows),
                       'OOF_positive_rows': sum(r['kind'] == 'OOF_positive' for r in rows),
                       'anatomy_crop_D_S_dependency': False,
                       'organizer_silver_predictor_training_history': 'unknown',
                       'D_uses_shared_GT_informed_legacy_plan': True})
    output = {'cases': result, 'source_C_upstream_all_selection_cases_gradient_excluded': True,
              'same_case_roles_for_joint_and_staged': True,
              'records_sha256': hashlib.sha256((RUN / 'features/records.jsonl').read_bytes()).hexdigest(),
              'limitations': ['GT-centered/jittered crops are supervised examples, not OOF detector proposals.',
                             'Anatomy-only rows use raw image and organizer silver targets, with zero shape input; D/S exclusion is not needed for these unused outputs.',
                             'Shared supervised planning and unknown silver history prevent an end-to-end leakage-free claim.',
                             'Source comparison uses development weights, not final weights that fit all267source cases.',
                             'Patient grouping only uses available identifiers; unknown cross-ID linkage remains.']}
    (RUN / 'CASE_LINEAGE.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'cases': len(result), 'C_fit_eligible': len(fit), 'selection': len(dev),
                      'all_own_D_S_gradient_exclusions_passed': True}), flush=True)
    from scripts.analysis import e28_source_support as support
    support.RUN = RUN
    support.main()


if __name__ == '__main__':
    main()
