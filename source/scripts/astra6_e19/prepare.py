"""MR anatomical F: source silver training, actual TA36 source calibration."""
from pathlib import Path
import json,shutil
import numpy as np,torch
from scripts.astra6_e01.e01_common import P,DATA,compute_feature,load_nifti_geometry,sha256_file,write_json
from scripts.astra6_e02.run_e02 import multiscale
from scripts.astra6_e14.train import combine
from scripts.astra6_e06.common import load_filter
from scripts.delivery.geometry import vessel_geometry_fast

RUN=P/'artifacts/astra6_e19_MR_anatomical_fp_filter_20260909'
F=P/'artifacts/astra6_e14_MR_filter_seed_stability_20260909'
BANK=P/'artifacts/astra6_e12_MR_expanded_fp_filter_20260909'
CACHE=P/'artifacts/source_MR_TA36_distribution_audit_20260909'

def initialize():
    for name in ['features/cases','model','evaluation','logs','predictions/mr_center2_k05']:(RUN/name).mkdir(parents=True,exist_ok=True)
    config={'experiment':'E19','hypothesis':'MR anatomical context can remove false positives remaining after E14, following the successful CT anatomical F mechanism','controls':'E17 MR D/E02 C/E17 S/E14 mean3 image F and all existing thresholds frozen','anatomical_features':'same943 geometry features plus image F probability, modality MR','source_training':'264 MR source cases with candidate rows; organizer TopBrain predicted silver vessels, not GT vessel annotations','source_calibration':'actual frozen TA36 predictions on40 source-development cases containing52 deployed candidates retained by E14 (43positive9negative); source E14 probabilities reused exactly','development_image_model':'E14 three source-development members exclude same source case groups; final model probabilities used only for final fitting','architecture':'ExtraTrees512,max_depth12,min_samples_leaf5,max_features.5,bootstrapFalse,class_weight balanced,seed20260909','training_budget':'512 development trees and512 final trees, checkpoints each64; total workflow budget6hours','source_gate':'retain all43 source positives at min-positive anatomical threshold and reject at least2 of9 remaining source negatives','official_gate':'all-six Pareto improvement overE17 and retain>=53matched and>=36correct locations; otherwise preserve frozenr2','no_MR40_or_CT5_fit':True,'no_threshold_or_seed_sweep':True,'distribution_caveat':'Source silver and TA36 are distinct predictions; calibrate on actual TA36 rather than assume equality'}
    write_json(RUN/'config.json',config)
    records=[json.loads(s) for s in (BANK/'features/records.jsonl').read_text().splitlines()];assert len(records)==3564 and len({r['case_id'] for r in records})==264 and all('_mr_' in r['case_id'] and 'center2' not in r['case_id'] for r in records)
    source=json.loads((F/'source_split.json').read_text());development=json.loads((F/'model/development_predictions.json').read_text());threshold=json.loads((F/'model/THRESHOLD.json').read_text())['threshold'];selected=[i for i,d,q in zip(development['rows'],development['deployed'],development['p']) if d and q>=threshold];cases=sorted({records[i]['case_id'] for i in selected});assert len(selected)==52 and sum(records[i]['y'] for i in selected)==43 and len(cases)==40;assert set(cases)<=set(source['development_cases'])
    write_json(RUN/'source_split.json',{**source,'anatomical_calibration_rows':selected,'anatomical_calibration_cases':cases,'parent_image_threshold':threshold});write_json(RUN/'TA36_REQUIRED_CASES.json',{'cases':cases});shutil.copy2(BANK/'features/records.jsonl',RUN/'features/records.jsonl');return records,development

def probabilities(stage,development):
    dest=RUN/f'features/{stage}_image_probability.npy'
    if dest.exists():return np.load(dest)
    if stage=='development':
        paths=[BANK/'model/development_best.pt',F/'seed20260910/model/development_last.pt',F/'seed20260911/model/development_last.pt'];model=combine(paths).cpu().eval()
    else:model=load_filter(F/'model/final_last.pt','cpu')
    images=np.load(BANK/'features/images.npy',mmap_mode='r');scores=[]
    with torch.inference_mode():
        for i in range(0,len(images),32):
            scores.extend(model(torch.from_numpy(np.asarray(images[i:i+32],np.float32))).sigmoid().numpy().tolist())
            if i%256==0:print('E19_IMAGE_PROBABILITY',stage,i,len(images),flush=True)
    scores=np.asarray(scores,np.float32)
    if stage=='development':
        ix=np.array(development['rows']);assert np.allclose(scores[ix],development['p'],atol=2e-5,rtol=2e-5);scores[ix]=development['p']
    np.save(dest,scores);return scores

def main():
    torch.set_num_threads(4);records,development=initialize()
    if (RUN/'features/SILVER_READY.json').exists():print('E19 silver features already complete',flush=True);return
    probabilities_by_stage={name:probabilities(name,development) for name in ['development','final']};groups={}
    for i,r in enumerate(records):groups.setdefault(r['case_id'],[]).append(i)
    X=np.empty((len(records),943),np.float32);hashes={}
    for n,(cid,ix) in enumerate(sorted(groups.items()),1):
        vessel=DATA/f'vessel_masks/{cid}.nii.gz';hashes[cid]=sha256_file(vessel);dest=RUN/f'features/cases/{cid}.npz'
        if dest.exists():
            cached=np.load(dest);assert np.array_equal(cached['indices'],ix);X[ix]=cached['X'];continue
        aff,shape=load_nifti_geometry(DATA/f'images/{cid}_0000.nii.gz');geom=vessel_geometry_fast(vessel,shape,aff);features=[]
        for i in ix:
            r=records[i];low,high=np.array(r['low']),np.array(r['high']);features.append(np.concatenate([compute_feature(geom,aff,low,high,'MR'),multiscale(geom,aff,low,high)]))
        xx=np.asarray(features,np.float32);assert xx.shape==(len(ix),943) and np.isfinite(xx).all();X[ix]=xx;np.savez_compressed(dest,X=xx,indices=ix);print('E19_SILVER_FEATURES',n,len(groups),cid,flush=True)
    for name,prob in probabilities_by_stage.items():np.savez_compressed(RUN/f'features/{name}_silver.npz',X=np.column_stack([X,prob]),y=np.array([r['y'] for r in records],np.uint8))
    write_json(RUN/'features/SILVER_READY.json',{'rows':len(records),'source_cases':len(groups),'feature_dim':944,'source_predicted_silver_hashes':hashes,'development_sha256':sha256_file(RUN/'features/development_silver.npz'),'final_sha256':sha256_file(RUN/'features/final_silver.npz'),'record_sha256':sha256_file(RUN/'features/records.jsonl'),'parent_source':json.loads((BANK/'features/READY.json').read_text())});print('E19_SILVER_READY',flush=True)

if __name__=='__main__':main()
