"""Bounded source-only audit of anatomy sampling; no model fitting or GPU."""
import hashlib
import json
from pathlib import Path
import numpy as np
import nibabel as nib

P = Path(__file__).resolve().parents[2]
DATA = P.parent / 'data_topaneu26'
R = P / 'artifacts/review_current_model_20260910'

def main():
    cfg = json.loads((P / 'artifacts/astra6_e28_joint_anatomy_location_20260909/config.json').read_text())
    fit = cfg['source_partition']['C_fit_cases']
    # Predefined case selection independent of evaluation images or outcomes.
    selected = []
    for center in ('center1', 'center5'):
        cases = [c for c in fit if center in c]
        selected.extend(sorted(cases, key=lambda c: hashlib.sha256(('review_anatomy_20260910:' + c).encode()).hexdigest())[:2])
    rows = []
    for cid in selected:
        path = DATA / 'vessel_masks' / (cid + '.nii.gz')
        image = nib.load(str(path))
        arr = np.asanyarray(image.dataobj)
        counts = np.zeros(37, dtype=np.int64)
        # Avoid an additional whole-volume int64 array.
        for z in range(arr.shape[2]):
            values = arr[:, :, z].astype(np.int64, copy=False).ravel()
            assert values.min() >= 0 and values.max() <= 36
            counts += np.bincount(values, minlength=37)
        del arr
        total = int(counts[1:].sum())
        present = int((counts[1:] > 0).sum())
        rows.append({'case_id': cid, 'silver_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'foreground_voxels': total, 'present_labels': present,
                     'label_counts': counts.tolist(),
                     'OA_AChA': {str(k): {'voxels': int(counts[k]),
                       'old_two_centers_probability_at_least_one': float(1-(1-counts[k]/max(total,1))**2),
                       'uniform_present_label_two_centers_probability': float(1-(1-1/max(present,1))**2) if counts[k] else 0.}
                       for k in (31,32,33,34)}})
        print(cid, rows[-1]['OA_AChA'], flush=True)
    result = {'selection': 'Two SHA-ranked C-fit cases per source center; no MR40 or source epoch-selection cases',
              'scope': 'Four-case code-mechanism check, not cohort prevalence or model-performance evidence',
              'rows': rows, 'interpretation': 'Original anatomy crop centers and foreground CE both weight labels by voxel count. Center probability is not probability that a label appears anywhere in a crop; missing silver labels cannot be recovered by balancing.'}
    R.mkdir(parents=True, exist_ok=True)
    (R / 'ANATOMY_SAMPLING_SOURCE_AUDIT.json').write_text(json.dumps(result, indent=2)+'\n')

if __name__ == '__main__':
    main()
