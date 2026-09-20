from pathlib import Path
import json
import nibabel as nib,numpy as np
P=Path(__file__).resolve().parents[2];D=P.parent/'data_topaneu26';out=[]
for p in sorted((D/'images').glob('*.nii.gz')):
 im=nib.load(p);out.append({'case_id':p.name.removesuffix('_0000.nii.gz'),'shape':list(im.shape),'spacing':list(map(float,im.header.get_zooms())),'voxels':int(np.prod(im.shape)),'compressed_bytes':p.stat().st_size})
(P/'reports/delivery_case_fingerprint_20260909.json').write_text(json.dumps(out,indent=2));print({m:max([r for r in out if f'_{m}_' in r['case_id']],key=lambda r:r['voxels']) for m in ['mr','ct']})
