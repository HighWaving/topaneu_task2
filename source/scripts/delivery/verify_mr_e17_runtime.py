"""Verify completed packaged E17 raw-image benchmark without GT access."""
from pathlib import Path
import hashlib,json,shutil
import SimpleITK as sitk
import numpy as np

P=Path(__file__).resolve().parents[2]
def main():
    work=P/'artifacts/submission_bundle_MR_E17_max_20260909'
    runtime=json.loads((work/'runtime.json').read_text());monitor=json.loads((P/'artifacts/submission_bundle_MR_E17_max_20260909_resource_tree.json').read_text())
    assert monitor['complete'] and monitor['exit_code']==0
    inp=sitk.ReadImage(str(work/'case_0000.nii.gz'));out=sitk.ReadImage(str(P/'artifacts/submission_bundle_MR_E17_max_20260909_output.mha'))
    assert inp.GetSize()==out.GetSize()
    for m in ['GetOrigin','GetDirection','GetSpacing']:assert np.allclose(getattr(inp,m)(),getattr(out,m)(),atol=1e-5,rtol=0)
    arr=sitk.GetArrayViewFromImage(out);labels=np.unique(arr).tolist();assert out.GetPixelID()==sitk.sitkUInt8 and min(labels)>=0 and max(labels)<=52
    cfg=json.loads((work/'runtime_configuration.json').read_text());weight=Path(cfg['arguments']['segmentation']);sha=hashlib.sha256(weight.read_bytes()).hexdigest();assert sha=='ec1d12c8028843d9ea2cc932f4025bf1247c87135bdb0266c1d5cfbfb20fa14d'
    new=json.loads((P/'artifacts/submission_bundle_MR_E17_max_20260909_output.json').read_text());old=json.loads((P/'artifacts/delivery_MR_E14_max_20260909_output.json').read_text())
    assert [(r['index'],r['keep']) for r in new['filter_decisions']]==[(r['index'],r['keep']) for r in old['filter_decisions']]
    report={'model':'MR E02 C/E17 S/E14 F','runtime':runtime,'geometry_verified':True,'uint8_labels_0_to_52':True,'labels':labels,'segmentation_sha256':sha,'parent_filter_decisions_unchanged':True,'sampled_process_tree_peak':monitor['peak'],'memory_measurement_limitation':monitor['limitations'],'test_purpose':'Source large-case full raw-image inference, not independent generalization','T4_validated':False}
    (work/'VERIFIED.json').write_text(json.dumps(report,indent=2)+'\n');shutil.copy2(work/'VERIFIED.json',P/'submission_models_20260909/reports/PACKAGED_MR_E17_MAX_VERIFIED.json');print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
