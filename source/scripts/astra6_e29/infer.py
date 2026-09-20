"""E29 uses the verified E28 C-only renderer with an explicitly isolated run root."""
import argparse
import json
from pathlib import Path
from scripts.astra6_e29.common import P, RUN, arm_path
from scripts.astra6_e01.e01_common import write_json, sha256_file, sha256_tree


def main(name):
    arm = arm_path(name)
    marker = arm / 'PREDICTIONS_HASHED_BEFORE_EVAL_GT.json'
    if marker.exists():
        old = json.loads(marker.read_text())
        assert old['prediction_tree_sha256'] == sha256_tree(arm / 'predictions')
        expected = json.loads((arm / 'model/LOCKED.json').read_text())['model_sha256']
        assert old.get('model_sha256', old.get('C28_sha256')) == expected
        assert old.get('experiment') in [None, 'E29_' + name]
        if old.get('experiment') is None:
            # Renderer finished but wrapper was interrupted before metadata normalization.
            assert json.loads((arm / 'RESEARCH_START.json').read_text())['experiment'] == 'E29_' + name
            old['model_sha256'] = old.pop('C28_sha256')
            old['experiment'] = 'E29_' + name
            write_json(marker, old)
        return
    baseline = P / 'artifacts/astra6_e17_MR_detector_crop_segmentation_20260909'
    write_json(arm / 'RESEARCH_START.json', {
        'baseline': 'E17', 'baseline_candidates_sha256': sha256_file(baseline / 'candidate_predictions.jsonl'),
        'experiment': 'E29_' + name, 'source_config_sha256': sha256_file(RUN / 'config.json'),
        'fixed_detector_segmenter_filter': True})
    import scripts.astra6_e28.infer as renderer
    renderer.RUN = arm
    print('E29 inference with unchanged C-only renderer', name, flush=True)
    renderer.main()
    lock = json.loads(marker.read_text())
    lock['model_sha256'] = lock.pop('C28_sha256')
    lock['experiment'] = 'E29_' + name
    lock['renderer_sha256'] = sha256_file(Path(renderer.__file__))
    write_json(marker, lock)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--arm', choices=['joint', 'staged'], required=True)
    main(p.parse_args().arm)
