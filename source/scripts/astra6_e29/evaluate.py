"""Read-only lesion and size attribution for a locked E29 arm."""
import argparse
import json
from scripts.astra6_e29.common import arm_path
from scripts.astra6_e01.e01_common import write_json


def main(name):
    arm = arm_path(name)
    output = arm / 'evaluation/lesion_diagnostics.json'
    version = 'E29_' + name
    if output.exists():
        if version in json.loads(output.read_text()):
            return
    else:
        import scripts.astra6_e28.evaluate as diagnosis
        diagnosis.RUN = arm
        diagnosis.main()
    result = json.loads(output.read_text())
    result[version] = result.pop('E28')
    result['size'] = {}
    for size in sorted({r['size_bin'] for r in result['paired_rows']}):
        rows = [r for r in result['paired_rows'] if r['size_bin'] == size]
        result['size'][size] = {'n': len(rows),
                                'location_rescued': sum(r['location_rescued'] for r in rows),
                                'location_lost': sum(r['location_lost'] for r in rows),
                                'before_correct': sum(r['before'].get('class_correct', False) for r in rows),
                                'after_correct': sum(r['after'].get('class_correct', False) for r in rows)}
    result['interpretation'] = 'C-only diagnostic; E17 foreground/native parity verified. MR40 is shared-plan developmental evidence, not independent generalization.'
    write_json(output, result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--arm', choices=['joint', 'staged'], required=True)
    main(p.parse_args().arm)
