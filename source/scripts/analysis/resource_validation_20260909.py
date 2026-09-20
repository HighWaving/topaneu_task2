from pathlib import Path
import json
import numpy as np,SimpleITK as sitk
P=Path(__file__).resolve().parents[2];out={'hardware':'V10016GB; not T4','runtime_contract_seconds':720,'ram_contract_GiB_available':31,'run_results':{},'equivalence':{},'container_status':'blocked: Buildah and buildah unshare both fail layer extraction with remount / flags0x44000 permission denied exit125','T4_tested':False}
for tag in ['delivery_max_mr_20260909','delivery_max_mr_fast_20260909','delivery_max_ct_20260909','delivery_max_ct_fast_20260909','delivery_negative_mr_20260909','delivery_max_mr_final_20260909']:
 r=json.loads((P/'artifacts'/tag/'runtime.json').read_text());r['child_peak_rss_GiB']=r['child_peak_rss_kb']/1024**2;r['note']='maximum individual child RSS, not an isolated-container aggregate';out['run_results'][tag]=r
for before,after in [('delivery_max_mr_20260909','delivery_max_mr_fast_20260909'),('delivery_max_ct_20260909','delivery_max_ct_fast_20260909'),('delivery_max_mr_20260909','delivery_max_mr_final_20260909')]:
 a=sitk.ReadImage(str(P/f'artifacts/{before}_output.mha'));b=sitk.ReadImage(str(P/f'artifacts/{after}_output.mha'));aa=sitk.GetArrayFromImage(a);bb=sitk.GetArrayFromImage(b);diff=int(np.count_nonzero(aa!=bb));assert diff==0 and a.GetSize()==b.GetSize() and np.allclose(a.GetSpacing(),b.GetSpacing()) and np.allclose(a.GetOrigin(),b.GetOrigin()) and np.allclose(a.GetDirection(),b.GetDirection());assert b.GetPixelID()==sitk.sitkUInt8 and bb.max()<=52;out['equivalence'][after]={'different_mask_voxels':diff,'geometry_preserved':True,'uint8_0_52':True}
r=P/'artifacts/gc_socket_test_20260909';im=sitk.ReadImage(str(r/'input/images/head-mr-angio/input.mha'));pr=sitk.ReadImage(str(r/'output/images/aneurysm-segmentation/output.mha'));assert not np.any(sitk.GetArrayFromImage(pr)) and pr.GetPixelID()==sitk.sitkUInt8 and pr.GetSize()==im.GetSize() and np.allclose(pr.GetSpacing(),im.GetSpacing()) and np.allclose(pr.GetOrigin(),im.GetOrigin()) and np.allclose(pr.GetDirection(),im.GetDirection());out['socket_test']={'valid':True,'output':'/output/images/aneurysm-segmentation/output.mha','no_tumor_case_zero_output':True,'case_is_functional_not_generalization':True,'runtime':json.loads(next((r/'work').rglob('runtime.json')).read_text())}
log=P/'logs/delivery_gpu_memory_20260909.jsonl';peaks={}
for line in log.read_text().splitlines():
 for row in json.loads(line)['gpus']:
  idx,mem,util=[int(x.strip()) for x in row.split(',')];peaks[str(idx)]=max(peaks.get(str(idx),0),mem)
for line in (P/'logs/delivery_final_gpu_memory_20260909.jsonl').read_text().splitlines():
 for row in json.loads(line)['gpus'].splitlines():
  idx,mem=[int(x.strip()) for x in row.split(',')]
  if idx==3:peaks[str(idx)]=max(peaks.get(str(idx),0),mem)
out['observed_GPU_memory_peak_MiB']=peaks;out['GPU_measurement_limit']='2-second polling, not CUDA allocator exact high-water mark; largest complete reruns covered; no OOM occurred'
for dest in [P/'reports/RESOURCE_VALIDATION_20260909.json',P/'delivery_20260909/RESOURCE_VALIDATION_20260909.json']:dest.write_text(json.dumps(out,indent=2))
print(json.dumps({'equivalence':out['equivalence'],'GPU_peak':peaks,'socket_test':True},indent=2))
