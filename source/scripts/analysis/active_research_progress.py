"""Bounded read-only observation of the explicitly registered active experiment."""
from pathlib import Path
import json
import statistics

P = Path(__file__).resolve().parents[2]


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def snapshot():
    pointer = read(P / 'LATEST_ACTIVE_RESEARCH.json')
    if not pointer:
        return None
    root = P / pointer['root']
    assert root.resolve().is_relative_to(P.resolve())
    result = {**pointer, 'workflow': read(root / 'WORKFLOW_PROGRESS.json'), 'arms': {}}
    for name in pointer.get('arms', ['']):
        arm = root / name
        stages = {}
        for stage in ['model/development', 'model/final', 'anatomy/development', 'anatomy/final']:
            history = read(arm / (stage + '_history.json'), [])
            stages[stage] = {'latest': history[-1] if history else None,
                             'median_recent_epoch_seconds': statistics.median(r['seconds'] for r in history[-5:]) if history else None}
        result['arms'][name or 'main'] = {'stages': stages, 'locked_model': read(arm / 'model/LOCKED.json'),
                                         'source_result_complete': ((arm / 'evaluation/SOURCE_RESULT.json').exists() or (arm / 'SOURCE_RESULT.json').exists()),
                                         'prediction_cases': len(list((arm / 'predictions/mr_center2_k05').glob('*.nii.gz'))),
                                         'decision': read(arm / 'evaluation/DECISION.json')}
    result['decision'] = read(root / 'DECISION.json')
    return result
