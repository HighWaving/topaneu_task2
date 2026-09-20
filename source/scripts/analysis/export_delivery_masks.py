from pathlib import Path
import json,hashlib
import SimpleITK as sitk,numpy as np
P=Path(__file__).resolve().parents[2];D=P.parent/'data_topaneu26';out=P/'delivery_20260909';ledger=[]
for cohort,source in [('MR40',P/'artifacts/astra6_e04_crop_segmentation_20260909T031146Z/predictions/mr_center2_k05'),('CT5',P/'artifacts/ct_independent5_20260909/E04')]:
 for p in sorted(source.glob('*.nii.gz')):
  cid=p.name.removesuffix('.nii.gz');ref=sitk.ReadImage(str(D/f'images/{cid}_0000.nii.gz'));im=sitk.ReadImage(str(p));a=sitk.GetArrayFromImage(im);assert a.min()>=0 and a.max()<=52;assert im.GetSize()==ref.GetSize() and np.allclose(im.GetSpacing(),ref.GetSpacing()) and np.allclose(im.GetDirection(),ref.GetDirection()) and np.allclose(im.GetOrigin(),ref.GetOrigin())
  im=sitk.Cast(im,sitk.sitkUInt8);im.CopyInformation(ref);target=out/f'predictions/{cohort}/{cid}.mha';target.parent.mkdir(parents=True,exist_ok=True);sitk.WriteImage(im,str(target),True);check=sitk.ReadImage(str(target));assert np.array_equal(sitk.GetArrayFromImage(check),a) and check.GetPixelID()==sitk.sitkUInt8
  ledger.append({'cohort':cohort,'case_id':cid,'file':str(target.relative_to(out)),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'native_geometry_verified':True,'labels_preserved':True})
(out/'PREDICTION_MANIFEST.json').write_text(json.dumps(ledger,indent=2));print(len(ledger),'MHA contract outputs verified')
