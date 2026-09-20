"""Task2 socket adapter for the verified raw image pipeline.
Container build/target-hardware validation is separate from this adapter.
"""
import argparse,json,os,subprocess,sys,tempfile
from pathlib import Path
P=Path(__file__).resolve().parents[2]
def select_input(root):
 mapping={'head-mr-angiography':('MR','head-mr-angio'),'head-ct-angiography':('CT','head-ct-angio')}
 metadata=json.loads((root/'inputs.json').read_text());slugs=[r['socket']['slug'] for r in metadata]
 if len(slugs)!=1 or slugs[0] not in mapping:raise ValueError(f'unsupported Task2 input interface: {slugs}')
 modality,folder=mapping[slugs[0]];directory=root/'images'/folder
 files=[f for f in directory.iterdir() if f.is_file() and any(f.name.endswith(s) for s in ('.mha','.tif','.tiff','.nii.gz'))]
 if len(files)!=1:raise ValueError(f'expected exactly one image in {directory}, found{len(files)}')
 return modality,files[0]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input-root',type=Path,default=Path('/input'));ap.add_argument('--output-root',type=Path,default=Path('/output'));ap.add_argument('--model-root',type=Path,default=Path('/opt/ml/model'));ap.add_argument('--runtime-config',type=Path);ap.add_argument('--work-root',type=Path,default=Path('/tmp'));ap.add_argument('--gpu',default=os.environ.get('CUDA_VISIBLE_DEVICES','0'));a=ap.parse_args();modality,image=select_input(a.input_root)
 output=a.output_root/'images/aneurysm-segmentation/output.mha';output.parent.mkdir(parents=True,exist_ok=True);a.work_root.mkdir(parents=True,exist_ok=True)
 # Preserve this work directory and logs on failure for diagnosis.
 work=Path(tempfile.mkdtemp(prefix='task2-',dir=a.work_root))/'pipeline'
 cmd=[sys.executable,'-m','scripts.delivery.raw_pipeline','--modality',modality,'--image',str(image.resolve()),'--output',str(output.resolve()),'--work',str(work.resolve()),'--gpu',a.gpu,'--segmentation',str((a.model_root/'segmentation.pt').resolve())]
 if a.runtime_config:cmd+=['--runtime-config',str(a.runtime_config.resolve())]
 subprocess.run(cmd,cwd=P,check=True)
if __name__=='__main__':main()
