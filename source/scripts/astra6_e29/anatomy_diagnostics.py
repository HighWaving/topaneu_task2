"""Same-source anatomy comparison and actual frozen-weight verification."""
import argparse
import json
import torch
from scripts.astra6_e29.common import RUN, arm_path
from scripts.astra6_e01.e01_common import write_json, sha256_file


def main(name):
    torch.set_num_threads(2)
    arm = arm_path(name)
    assert (arm / 'model/LOCKED.json').exists()
    if name == 'staged' and not (arm / 'evaluation/ACTUAL_ENCODER_FREEZE_CHECK.json').exists():
        checks = {}
        for stage, c_name in [('development', 'development_best'), ('final', 'final_last')]:
            ap = arm / f'anatomy/{stage}_last.pt'
            cp = arm / f'model/{c_name}.pt'
            a = torch.load(ap, map_location='cpu', weights_only=False)['state_dict']
            c = torch.load(cp, map_location='cpu', weights_only=False)['state_dict']
            keys = [k for k in a if not k.startswith('classifier.')]
            assert all(torch.equal(a[k], c[k]) for k in keys), 'Frozen encoder changed during C training'
            checks[stage] = {'anatomy_checkpoint_sha256': sha256_file(ap),
                             'C_checkpoint_sha256': sha256_file(cp),
                             'n_exactly_equal_state_entries': len(keys)}
        write_json(arm / 'evaluation/ACTUAL_ENCODER_FREEZE_CHECK.json', {'passed': True, 'stages': checks})
    output = arm / 'evaluation/SOURCE_ANATOMY_DIAGNOSTICS.json'
    if not output.exists():
        records = [json.loads(s) for s in (RUN / 'features/records.jsonl').read_text().splitlines()]
        split = json.loads((RUN / 'source_split.json').read_text())
        import scripts.astra6_e28.anatomy_diagnostics as diagnosis
        diagnosis.RUN = arm
        diagnosis.run(records, split)
    result = json.loads(output.read_text())
    result['experiment'] = 'E29_' + name
    result['source_partition'] = 'E29 fold1 source selection; common across both arms'
    write_json(output, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--arm', choices=['joint', 'staged'], required=True)
    main(parser.parse_args().arm)
