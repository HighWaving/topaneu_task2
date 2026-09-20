"""Measured E28 budget scenarios; observes artifacts without controlling training."""
from pathlib import Path
import datetime
import json
import statistics

P = Path(__file__).resolve().parents[2]
RUN = P / 'artifacts/astra6_e28_joint_anatomy_location_20260909'


def read(path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def snapshot():
    history = read(RUN / 'model/development_history.json', [])
    final = read(RUN / 'model/final_history.json', [])
    config = read(RUN / 'config.json', {})['planned_budget']
    split = read(RUN / 'source_split.json', {})
    result = {
        'computed_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'development_epochs_completed': history[-1]['epoch'] if history else 0,
        'final_epochs_completed': final[-1]['epoch'] if final else 0,
        'features_ready': (RUN / 'features/READY.json').exists(),
        'limits': 'Scenarios, not confidence intervals or completion guarantees. Training seconds exclude validation/save; use a 15% allowance. Later planning repair, CT research and T4 submission validation excluded.',
        'training_budget_unchanged': True,
    }
    if history:
        recent = history[-5:]
        seconds = statistics.median(r['seconds'] for r in recent) * 1.15
        best = min(history, key=lambda r: r['source_case_CE'])['epoch']
        n = history[-1]['epoch']
        ceiling = config['max_development_epochs']
        earliest_stop = min(ceiling, max(config['minimum_stop_epoch'], best + config['patience']))
        development_done = (RUN / 'evaluation/SOURCE_RESULT.json').exists()
        remaining_dev = [0, 0] if development_done else [max(0, earliest_stop - n), max(0, ceiling - n)]
        # Batch size is unchanged across development/final; ratio uses actual rows.
        step_ratio = ((len(split['final_rows']) + 3) // 4) / ((len(split['train_rows']) + 3) // 4)
        final_seconds = statistics.median(r['seconds'] for r in final[-5:]) * 1.15 if final else seconds * step_ratio
        final_n = final[-1]['epoch'] if final else 0
        remaining_final = [max(0, best - final_n), max(0, (best if development_done else ceiling) - final_n)]
        hours = [(remaining_dev[i] * seconds + remaining_final[i] * final_seconds) / 3600 for i in [0, 1]]
        if (RUN / 'model/LOCKED.json').exists():
            hours = [0., 0.]
        result.update(recent_complete_epochs=len(recent), recent_epoch_seconds_with_allowance=seconds,
                      current_best_source_epoch=best, final_epoch_seconds_measured_or_extrapolated=final_seconds,
                      training_remaining_hours_scenario=hours,
                      minimum_scenario_assumption='No future source improvement; existing best epoch retained. Upper scenario completes max development and max selected-final epochs.')
    result['inference_official_diagnostics_remaining_hours_provisional'] = (
        [0., 0.] if (RUN / 'evaluation/DECISION.json').exists() else [.25, 1.])
    path = RUN / 'BUDGET_CURRENT.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, indent=2) + '\n')
    tmp.replace(path)
    return result


if __name__ == '__main__':
    print(json.dumps(snapshot(), indent=2))
