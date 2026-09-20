"""Read-only actual E28 supervision inventory, deduplicated by GT component.

Run after features/READY.json. No fit, model selection, or input alteration.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json

P = Path(__file__).resolve().parents[2]
RUN = P / 'artifacts/astra6_e28_joint_anatomy_location_20260909'


def main():
    assert (RUN / 'features/READY.json').exists()
    path = RUN / 'features/records.jsonl'
    records = [json.loads(s) for s in path.read_text().splitlines()]
    split = json.loads((RUN / 'source_split.json').read_text())
    mapping_path = P.parent / 'data_topaneu26/location_mapping.json'
    labels = json.loads(mapping_path.read_text())['labels']
    mirror = {}
    for name, c in labels.items():
        partner = ('L-' + name[2:] if name.startswith('R-') else
                   'R-' + name[2:] if name.startswith('L-') else name)
        mirror[c] = labels[partner]
    assert all(mirror[mirror[c]] == c for c in mirror)
    summary = {}
    for stage, key in [('development_fit', 'train_rows'),
                       ('development_selection', 'validation_rows'),
                       ('final_fit', 'final_rows')]:
        rows = [records[i] for i in split[key]]
        components = {}
        for row in rows:
            if row['class'] <= 0:
                continue
            identity = row['case_id'], row['component_index']
            if identity in components:
                assert components[identity] == row['class']
            components[identity] = row['class']
        counts = Counter(components.values())
        # Read named laterality: midline labels break any global parity rule.
        # Count a self-mapped midline component once.
        paired = {str(c): counts[c] + (counts[mirror[c]] if mirror[c] != c else 0)
                  for c in range(1, 53)}
        anatomy = [r for r in rows if r['kind'] == 'silver_anatomy_only']
        summary[stage] = {
            'rows': len(rows), 'cases': len({r['case_id'] for r in rows}),
            'classification_crop_rows': sum(r['class'] > 0 for r in rows),
            'unique_classification_components': len(components),
            'original_component_counts': {str(c): counts[c] for c in range(1, 53)},
            'mirror_pair_component_support': paired,
            'anatomy_only_rows': len(anatomy),
            'anatomy_selected_center_labels': dict(Counter(r['anatomy_center_label'] for r in anatomy)),
            'anatomy_selected_label_lost_after_resampling': [
                {'case_id': r['case_id'], 'label': r['anatomy_center_label'],
                 'sampling': r['anatomy_center_sampling']}
                for r in anatomy if not r['center_label_present_after_resampling']],
        }
    train = summary['development_fit']
    validation = summary['development_selection']
    absent = [int(c) for c, n in validation['original_component_counts'].items()
              if n and not train['mirror_pair_component_support'][c]]
    result = {
        'records_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'mapping_sha256': hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
        'stages': summary,
        'validation_classes_without_development_mirror_pair_support': absent,
        'no_model_or_threshold_change': True,
        'limitations': [
            'Mirror support counts original components from either side, not additional independent lesions.',
            'Anatomy center labels describe the chosen centers, not all labels visible in each crop.',
            'A label surviving resampling is not evidence of anatomically correct silver supervision.',
            'Not every validation class is eligible for deployed same-family/same-side refinement.',
            'Known upstream supervised planning exposure and silver-model history limitations remain.',
        ],
    }
    out = RUN / 'evaluation/SOURCE_SUPPORT_AUDIT.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'stages': {k: {'rows': v['rows'],
          'unique_components': v['unique_classification_components']} for k, v in summary.items()},
          'validation_classes_without_mirror_support': absent}), flush=True)


if __name__ == '__main__':
    main()
