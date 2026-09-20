"""Native MR image + predicted vessels + boxes -> E19 anatomical filtering."""
from pathlib import Path
import argparse,json,pickle,resource,tempfile,time
import joblib,nibabel as nib,numpy as np,SimpleITK as sitk,torch
from scripts.astra6_e19.prepare import RUN,F,P,DATA
from scripts.astra6_e06.common import load_filter,crops
from scripts.astra6_e01.e01_common import load_boxes,select_candidates,compute_feature,sha256_file,sha256_tree,write_json,nifti_output
from scripts.astra6_e02.run_e02 import multiscale
from scripts.delivery.geometry import vessel_geometry_fast
from scripts.delivery.refine import predict
BASE=P/'artifacts/astra6_e17_MR_detector_crop_segmentation_20260909'
C=P/'artifacts/astra6_e02_multiscale_vessel_signature_20260908T173916Z'

def native(image,vessel,boxes,classifier,segmentation,image_filter,image_threshold,anatomical_filter,anatomical_threshold,output,device):
    start=time.monotonic();torch.set_num_threads(4);clf=joblib.load(anatomical_filter);assert clf.feature_version=='anatomical_fp_v1' and clf.expected_modality=='MR' and sha256_file(image_filter)==clf.required_image_filter_sha256
    im=nib.load(str(image));arr=im.get_fdata(dtype=np.float32);aff=im.affine;geom=vessel_geometry_fast(vessel,arr.shape,aff);sub=arr[::4,::4,::4];sub=sub[sub!=0];assert len(sub) and np.isfinite(arr).all();lo,hi=np.percentile(sub,[.5,99.5]);arr-=float(lo);arr/=max(float(hi-lo),1e-6);np.clip(arr,0,1,out=arr)
    net=load_filter(image_filter,device);bx,sc,obj=load_boxes(boxes);filtered=np.zeros_like(sc);decisions=[]
    for idx,score,low,high in select_candidates(bx,sc):
        with torch.inference_mode():prob=float(net(torch.from_numpy(crops(arr,aff,low,high)[None]).to(device)).sigmoid().cpu()[0])
        feature=np.concatenate([compute_feature(geom,aff,low,high,'MR'),multiscale(geom,aff,low,high),[prob]]).astype(np.float32);q=float(clf.predict_proba(feature[None])[0,list(clf.classes_).index(1)]);keep=prob>=image_threshold and q>=anatomical_threshold
        if keep:filtered[idx]=score
        decisions.append({'index':idx,'image_probability':prob,'anatomical_probability':q,'parent_keep':prob>=image_threshold,'keep':keep})
    del arr,im,geom,net,clf
    if device!='cpu':torch.cuda.empty_cache()
    obj['pred_scores']=filtered
    with tempfile.TemporaryDirectory(prefix='task2-MR-anatomy-filter-') as td:
        path=Path(td)/'boxes.pkl';path.write_bytes(pickle.dumps(obj));mask,ledger=predict(image,vessel,path,classifier,segmentation,device,'MR')
    ref=sitk.ReadImage(str(image));out=sitk.GetImageFromArray(mask.transpose(2,1,0));out.CopyInformation(ref);output.parent.mkdir(parents=True,exist_ok=True);sitk.WriteImage(out,str(output),True);write_json(output.with_suffix('.json'),{'seconds':time.monotonic()-start,'peak_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'decisions':decisions,'candidates':ledger,'no_refill':True,'native_geometry':True});return mask,decisions

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--image',type=Path);parser.add_argument('--predicted-vessel',type=Path);parser.add_argument('--boxes',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--classifier',type=Path,default=C/'model/classifier.joblib');parser.add_argument('--segmentation',type=Path,default=BASE/'model/final_last.pt');parser.add_argument('--image-filter','--fp-filter',dest='image_filter',type=Path,default=F/'model/final_last.pt');parser.add_argument('--image-threshold','--fp-threshold',dest='image_threshold',type=float);parser.add_argument('--modality',choices=['MR'],default='MR');parser.add_argument('--anatomical-filter',type=Path,default=RUN/'model/final.joblib');parser.add_argument('--anatomical-threshold',type=float);parser.add_argument('--device',default='cuda:0');a=parser.parse_args()
    threshold=json.loads((RUN/'model/THRESHOLD.json').read_text()) if a.image_threshold is None or a.anatomical_threshold is None else {};it=a.image_threshold if a.image_threshold is not None else threshold['parent_image_threshold'];at=a.anatomical_threshold if a.anatomical_threshold is not None else threshold['threshold']
    if a.image:
        assert a.predicted_vessel and a.boxes and a.output;native(a.image,a.predicted_vessel,a.boxes,a.classifier,a.segmentation,a.image_filter,it,a.anatomical_filter,at,a.output,a.device);return
    lock=json.loads((RUN/'model/LOCKED.json').read_text());assert lock['source_gate_passed'] and sha256_file(a.anatomical_filter)==lock['model_sha256'];ids=json.loads((C/'eval_case_ids.json').read_text());rows=[json.loads(s) for s in (BASE/'candidate_predictions.jsonl').read_text().splitlines()];lookup={(r['case_id'],r['original_index']):r for r in rows};result=[];start=time.monotonic()
    write_json(RUN/'INFERENCE_INPUTS_LOCKED.json',{'anatomical_filter_sha256':lock['model_sha256'],'threshold_sha256':sha256_file(RUN/'model/THRESHOLD.json'),'D_C_S_F_assignments_sha256':sha256_file(BASE/'candidate_predictions.jsonl'),'segmentation_sha256':sha256_file(a.segmentation),'classifier_sha256':sha256_file(a.classifier),'image_filter_sha256':sha256_file(a.image_filter),'GT_not_input':True})
    for cid in ids:
        image=DATA/f'images/{cid}_0000.nii.gz';dest=RUN/f'predictions/mr_center2_k05/{cid}.nii.gz';mask,decisions=native(image,P/f'artifacts/ta36_mr_center2_output/{cid}.nii.gz',P/f'artifacts/fold1_eval_center2_epoch60/{cid}_boxes.pkl',a.classifier,a.segmentation,a.image_filter,it,a.anatomical_filter,at,dest,a.device);nifti_output(mask,image,dest)
        for d in decisions:
            old=lookup[cid,d['index']];assert old['filter_keep']==d['parent_keep'];result.append({**old,'parent_filter_keep':old['filter_keep'],'filter_keep':d['keep'],'anatomical_probability':d['anatomical_probability'],'image_probability':d['image_probability']})
        print('E19_INFERENCE',cid,flush=True)
    (RUN/'candidate_predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in result)+'\n');write_json(RUN/'PREDICTIONS_LOCKED.json',{'prediction_tree_sha256':sha256_tree(RUN/'predictions'),'GT_not_read':True,'n_cases':40,'seconds':time.monotonic()-start,'anatomical_filter_sha256':lock['model_sha256']})

if __name__=='__main__':main()
