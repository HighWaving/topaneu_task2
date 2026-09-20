"""Same-real-logits CPU vs CUDA export profile; returns CPU control to original entry."""
import json,runpy,sys,time,resource
from pathlib import Path
import numpy as np,torch,SimpleITK as sitk
from scripts.astra6_e34.gpu_export import streamed_gpu,resample_slab
from scripts.delivery.ta36_cached_preprocessing import CachedAdapter
P=Path(__file__).resolve().parents[2];RUN=P/'artifacts/astra6_e34_gpu_export_runtime_20260910';APP=P.parent/'topaneu-task1-sanity-submission';WORK=RUN/'MR_worst_profile'
def main():
 torch.set_num_threads(4);WORK.mkdir(exist_ok=True)
 import nnunetv2.houjing_scripts.infer_ppl_parallel_npz as pipeline
 original=pipeline._streamed_seg_from_sources;pipeline.PreprocessAdapterFromNpy=CachedAdapter
 torch.manual_seed(20260910);src=torch.randn(7,8,11,13);a=pipeline._resample_slab_trilinear(src,3,9,17,(19,23));b=resample_slab(src.cuda(),3,9,17,(19,23)).cpu();error=float((a-b).abs().max());assert error<1e-5
 (WORK/'SYNTHETIC_PREFLIGHT.json').write_text(json.dumps({'max_logit_error_CPU_vs_CUDA':error,'same_half_pixel_geometry':True,'not_a_model_training_experiment':True})+'\n')
 def paired(sources,n_sources,pm,cm,lm,props,fp16=False):
  begin=time.monotonic();values=list(sources);torch.cuda.synchronize();network=time.monotonic()-begin
  t=time.monotonic();control=original(iter(values),n_sources,pm,cm,lm,props,fp16=fp16);cpu=time.monotonic()-t
  torch.cuda.reset_peak_memory_stats();t=time.monotonic();candidate=streamed_gpu(iter(values),n_sources,pm,cm,lm,props,fp16=fp16);torch.cuda.synchronize();gpu=time.monotonic()-t;different=control!=candidate
  np.save(WORK/'CPU_raw_seg.npy',control);np.save(WORK/'GPU_raw_seg.npy',candidate)
  report={'n_sources':n_sources,'fp16_accumulator':fp16,'source_shapes':[list(v.shape) for v in values],'native_shape':list(control.shape),'same_logits_network_seconds':network,'CPU_export_seconds':cpu,'GPU_export_seconds':gpu,'export_speedup':cpu/gpu,'different_voxels':int(different.sum()),'different_fraction':float(different.mean()),'foreground_union_voxels':int(((control>0)|(candidate>0)).sum()),'gpu_peak_bytes_during_export':torch.cuda.max_memory_allocated(),'peak_rss_kb_profile_only':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'profile_holds_all_source_logits_for_paired_comparison':True,'production_implementation_streams_one_source':True,'T4_validated':False}
  (WORK/'PAIRED_EXPORT.json').write_text(json.dumps(report,indent=2)+'\n');print('E34 PAIRED EXPORT',json.dumps(report),flush=True);return control
 pipeline._streamed_seg_from_sources=paired
 image=P/'artifacts/submission_bundle_MR_E17_max_20260909/ta36_input/case_0000.nii.gz';output=WORK/'CPU_control';sys.argv=[str(APP/'ta36/run_inference.py'),'--input',str(image),'--output',str(output),'--suffix','_0000.nii.gz','--sequential','--n_infer_workers','1','--n_pre_post_workers','1'];runpy.run_path(str(APP/'ta36/run_inference.py'),run_name='__main__')
 candidate=np.load(WORK/'GPU_raw_seg.npy');candidate=pipeline.generic_prune_nparray(candidate);im=sitk.GetImageFromArray(candidate);im.CopyInformation(sitk.ReadImage(str(image)));sitk.WriteImage(im,str(WORK/'GPU_candidate.nii.gz'),True);control=sitk.GetArrayFromImage(sitk.ReadImage(str(output/'case.nii.gz')));changed=control!=candidate;(WORK/'POSTPROCESS_PARITY.json').write_text(json.dumps({'different_voxels':int(changed.sum()),'different_fraction':float(changed.mean()),'same_logits_paired':True,'GT_not_read':True,'same_models_and_probability_fusion':True,'not_whole_pipeline_or_T4_runtime':True},indent=2)+'\n')
if __name__=='__main__':main()
