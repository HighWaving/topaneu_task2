"""Diagnostic orthogonal slices for the three locked E25 decision changes.

No model fitting, relabeling, or operating-point selection.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.astra6_e01.e01_common import P, DATA, TA36_DIR, load_nifti


def main():
    run = P / 'artifacts/astra6_e25_oof_shape_fp_filter_20260909'
    base = P / 'artifacts/astra6_e17_MR_detector_crop_segmentation_20260909'
    out = P / 'artifacts/review_current_model_20260910/E25_changed_candidates'
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(s) for s in (run / 'candidate_predictions.jsonl').read_text().splitlines()]
    manifest = []
    for row in rows:
        if row['filter_keep'] == row['baseline_filter_keep']:
            continue
        cid = row['case_id']
        arr, aff, _ = load_nifti(DATA / f'images/{cid}_0000.nii.gz')
        center = np.rint((np.array(row['low']) + row['high']) / 2).astype(int)
        radius = np.ceil(12 / np.linalg.norm(aff[:3, :3], axis=0)).astype(int)
        low = np.maximum(center - radius, 0)
        high = np.minimum(center + radius + 1, arr.shape)
        crop = tuple(slice(l, h) for l, h in zip(low, high))
        image = arr[crop].copy()
        del arr
        masks = {}
        for name, path in [('GT', DATA / f'location_masks/{cid}.nii.gz'),
                           ('old', base / f'predictions/mr_center2_k05/{cid}.nii.gz'),
                           ('new', run / f'predictions/mr_center2_k05/{cid}.nii.gz'),
                           ('vessel', TA36_DIR / f'{cid}.nii.gz')]:
            data, ma, _ = load_nifti(path)
            assert np.allclose(ma, aff, atol=1e-4)
            masks[name] = data[crop].copy()
            del data
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        vmin, vmax = np.percentile(image, [1, 99])
        for axis in range(3):
            index = int(center[axis] - low[axis])
            plane = np.take(image, index, axis=axis).T
            for ax in axes[:, axis]:
                ax.imshow(plane, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
                ax.set_axis_off()
            for name, color in [('GT', 'red'), ('old', 'lime'), ('new', 'yellow')]:
                plane_mask = np.take(masks[name], index, axis=axis).T > 0
                if plane_mask.any() and not plane_mask.all():
                    axes[0, axis].contour(plane_mask, levels=[.5], colors=[color], linewidths=1)
            silver = np.take(masks['vessel'], index, axis=axis).T
            axes[1, axis].imshow(np.ma.masked_equal(silver, 0), cmap='tab20', vmin=1, vmax=36,
                                 alpha=.6, origin='lower', interpolation='nearest')
            axes[0, axis].set_title(f'Native axis {axis}; candidate center')
        fig.suptitle(f'{cid}: keep {row["baseline_filter_keep"]} -> {row["filter_keep"]}\n'
                     'Top: GT red / old green / new yellow. Bottom: predicted vessel labels.\n'
                     '24 mm crop; native voxel aspect; three slices only, no clinical adjudication.')
        fig.tight_layout()
        path = out / f'{cid}.png'
        fig.savefig(path, dpi=140)
        plt.close(fig)
        manifest.append({'case_id': cid, 'candidate_index': row['original_index'],
                         'center_native': center.tolist(), 'panel': str(path),
                         'predicted_vessel_is_diagnostic_only': True})
        print(cid, flush=True)
    (out / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
