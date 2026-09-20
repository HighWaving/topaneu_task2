"""Reuse immutable E28 image crops with a new, declared fold1 source split."""
import json
import re
from pathlib import Path
import numpy as np
import nibabel as nib
import joblib
from sklearn.base import clone
from scripts.astra6_e29.common import P, RUN, PREVIOUS, arm_path
from scripts.astra6_e01.e01_common import DATA, write_json, sha256_file, compute_feature
from scripts.astra6_e02.run_e02 import E01, extended_schema, multiscale
from scripts.astra6_e28.common import BASE, OOF
from scripts.delivery.geometry import vessel_geometry_fast


def group(cid):
    return re.sub(r'(_(?:mr|ct)_\d+)_\d+$', r'\1', cid)


def main():
    if (RUN / 'features/READY.json').exists():
        return
    config = json.loads((RUN / 'config.json').read_text())
    assert sha256_file(PREVIOUS / 'features/READY.json') == config['cached_E28_features_READY_sha256']
    ready = json.loads((PREVIOUS / 'features/READY.json').read_text())
    for key, digest in ready['arrays_sha256'].items():
        assert sha256_file(PREVIOUS / f'features/{key}.npy') == digest
    part = config['source_partition']
    fit, dev = set(part['C_fit_cases']), set(part['epoch_selection_cases'])
    assert not {group(c) for c in fit} & {group(c) for c in dev}
    dsplit = json.loads((OOF / 'source_split.json').read_text())['folds'][1]
    ssplit = json.loads((OOF / 'segmenters/fold1/source_split.json').read_text())
    assert fit | dev == set(dsplit['val'])
    for upstream in [dsplit['train'], ssplit['training_cases']]:
        assert not {group(c) for c in upstream} & {group(c) for c in fit | dev}
    records = [json.loads(s) for s in (PREVIOUS / 'features/records.jsonl').read_text().splitlines()]
    chosen = {}
    for i, row in enumerate(records):
        row.pop('source_baseline_class', None)
        row.pop('baseline_geometry_features', None)
        row['use_C_fit'] = row['case_id'] in fit and row['class'] > 0
        row['use_C_validation'] = row['case_id'] in dev and row['kind'] == 'OOF_positive' and row['operating']
        row['use_anatomy_fit'] = row['case_id'] not in dev
        if row['use_C_fit'] or row['use_C_validation']:
            assert row['OOF_fold'] == 1
        if row['use_C_validation']:
            key = row['case_id'], row['component_index']
            old = chosen.get(key)
            if old is None or (row['score'], -row['candidate_index']) > (records[old]['score'], -records[old]['candidate_index']):
                chosen[key] = i
    train = [i for i, r in enumerate(records) if r['use_C_fit'] or (r['class'] < 0 and r['use_anatomy_fit'])]
    validation = sorted(chosen.values())
    assert train and validation
    schema = json.loads((E01 / 'feature_schema.json').read_text())
    _, vp, _ = extended_schema(schema)
    for cid in sorted({records[i]['case_id'] for i in validation}):
        image = nib.load(str(DATA / f'images/{cid}_0000.nii.gz'))
        aff = image.affine
        geometry = vessel_geometry_fast(DATA / f'vessel_masks/{cid}.nii.gz', image.shape, aff)
        for i in validation:
            r = records[i]
            if r['case_id'] == cid:
                lo, hi = np.array(r['low']), np.array(r['high'])
                r['baseline_geometry_features'] = np.concatenate([
                    compute_feature(geometry, aff, lo, hi, 'MR', False, vp),
                    multiscale(geometry, aff, lo, hi)]).astype(np.float32).tolist()
        print('E29 source geometry', cid, flush=True)
    source_rows = [json.loads(s) for s in (BASE / 'features/train_records.jsonl').read_text().splitlines()]
    ix = [i for i, r in enumerate(source_rows) if r['case_id'] in fit]
    data = np.load(BASE / 'features/train.npz')
    comparator = clone(joblib.load(BASE / 'model/classifier.joblib')).set_params(n_jobs=1)
    comparator.fit(data['X'][ix], data['y'][ix], sample_weight=data['sample_weight'][ix])
    (RUN / 'features').mkdir(parents=True, exist_ok=True)
    joblib.dump(comparator, RUN / 'features/source_C02_comparator.joblib')
    predictions = []
    for i in validation:
        r = records[i]
        before = int(comparator.predict(np.array(r['baseline_geometry_features'], np.float32)[None])[0])
        r['source_baseline_class'] = before
        predictions.append({'row': i, 'prediction': before, 'class': r['class'], 'case_id': r['case_id']})
    write_json(RUN / 'SOURCE_BASELINE.json', {
        'rows': predictions, 'correct': sum(r['prediction'] == r['class'] for r in predictions),
        'actual_fit_cases': sorted({source_rows[i]['case_id'] for i in ix}),
        'n_original_augmentation_rows': len(ix), 'same_comparator_for_both_arms': True})
    split = {'train_rows': train, 'validation_rows': validation, 'final_rows': list(range(len(records))),
             'training_C_cases': sorted(fit), 'development_cases': sorted(dev),
             'anatomy_train_rows': [i for i, r in enumerate(records) if r['class'] < 0 and r['case_id'] not in dev],
             'anatomy_final_rows': [i for i, r in enumerate(records) if r['class'] < 0]}
    write_json(RUN / 'source_split.json', split)
    (RUN / 'features/records.jsonl').write_text('\n'.join(json.dumps(r) for r in records) + '\n')
    for name in ['joint', 'staged']:
        arm = arm_path(name)
        for folder in ['features', 'model', 'evaluation']:
            (arm / folder).mkdir(parents=True, exist_ok=True)
        for key in ready['arrays_sha256']:
            target = arm / f'features/{key}.npy'
            if not target.exists():
                target.symlink_to(PREVIOUS / f'features/{key}.npy')
        write_json(arm / 'source_split.json', split)
        write_json(arm / 'config.json', {**config, 'arm': name})
    write_json(RUN / 'features/READY.json', {
        'rows': len(records), 'training_rows': len(train), 'validation_components': len(validation),
        'records_sha256': sha256_file(RUN / 'features/records.jsonl'),
        'source_split_sha256': sha256_file(RUN / 'source_split.json'),
        'config_sha256': sha256_file(RUN / 'config.json'), 'reused_array_hashes': ready['arrays_sha256'],
        'known_supervised_planning_exposure': True, 'no_MR40_CT5_fit': True})
    print('E29 sources ready', len(train), len(validation), flush=True)


if __name__ == '__main__':
    main()
