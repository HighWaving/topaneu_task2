"""CT actual-detector crop adaptation, preserving the complete E16 GT bank."""
import json
import numpy as np
from scipy.ndimage import map_coordinates
from scripts.astra6_e01.e01_common import P,DATA,load_nifti,load_nifti_geometry,load_boxes,select_candidates,write_json,sha256_file
from scripts.astra6_e04.run_e04 import SUPPORT,crop_coordinates,component_records_fast,load_image,crop_volume
from scripts.astra6_e05.run_e05 import group

RUN=P/'artifacts/astra6_e20_CT_detector_crop_segmentation_20260909'
BASE=P/'artifacts/astra6_e16_CT_expanded_segmentation_20260909'
BANK=P/'artifacts/astra6_e11_CT_expanded_fp_filter_20260909'
AUDIT=P/'artifacts/source_CT_detector_crop_dice_audit_20260909'

def main():
    pre=json.loads((AUDIT/'RESULT.json').read_text());assert pre['training_trigger_passed']
    for name in ['features/cases','model','evaluation','predictions_ct','logs']:(RUN/name).mkdir(parents=True,exist_ok=True)
    config={'experiment':'E20','hypothesis':'Actual CT detector crops address the observed paired source GT-to-D crop Dice gap, using the same successful data adaptation mechanism as MR E17','source_precondition':{'n':pre['n'],'GT_Dice':pre['scores']['GT']['mean_Dice'],'D_Dice':pre['scores']['detector']['mean_Dice'],'gap':pre['Dice_gap']},'change':'Add actual CT detector-positive crops to unchanged E16 GT-jitter image/target bank; same32cube architecture, original box support, optimizer, batch32 and fixed13development+13fullepochs','target_assignment':'Highest bounding-box IoU among GT components with>=10percent voxels in candidate, consistent with the source crop audit','source_split':'Exact E16 source case groups, CT5 andMR40 excluded','budget_hours':3,'checkpoint':'Each epoch model, optimizer and all RNG states','source_gate':'New helper detector-crop Dice on same28source lesions must not be worse than E16 helper','official_gate':'CT_E16 all-six Pareto and>=4matched/>=3correct locations; MR unchanged','no_CT5_or_MR40_fit':True};write_json(RUN/'config.json',config)
    if (RUN/'features/SOURCE_READY.json').exists():print('E20 source already complete',flush=True);return
    old=[json.loads(s) for s in (BASE/'features/train_records.jsonl').read_text().splitlines()];base_split=json.loads((BASE/'source_split.json').read_text());fixed=set(base_split['fixed_CT5']);fr=[json.loads(s) for s in (BANK/'features/records.jsonl').read_text().splitlines()];images=np.load(BANK/'features/images.npy',mmap_mode='r');groups={};pools={}
    gt_lookup={(r['case_id'],r['source_class_id'],tuple(r['low']),tuple(r['high'])):i for i,r in enumerate(old) if r['view']=='original' and r['sample_index']==0}
    for i,r in enumerate(fr):
        cid=r['case_id'];assert '_ct_' in cid and cid not in fixed
        if r['augmentation'] or not r['y']:continue
        if cid not in pools:
            bx,sc,_=load_boxes(P/f'artifacts/ct_fold2_all109_boxes/{cid}_boxes.pkl');pools[cid]={v[0] for v in select_candidates(bx,sc)}
        if r['original_index'] in pools[cid]:groups.setdefault(cid,[]).append((i,r))
    caches=[];skipped=[];geometry_checked=False
    for n,(cid,items) in enumerate(sorted(groups.items()),1):
        dest=RUN/f'features/cases/{cid}.npz';meta=dest.with_suffix('.json')
        if dest.exists() and meta.exists():caches.append((dest,meta));continue
        gt,ga,shape=load_nifti(DATA/f'location_masks/{cid}.nii.gz');aff,ish=load_nifti_geometry(DATA/f'images/{cid}_0000.nii.gz');assert shape==ish and np.allclose(ga,aff,atol=1e-4);comps=component_records_fast(gt);xx=[];yy=[];rr=[]
        for i,r in items:
            low,high=np.array(r['low']),np.array(r['high']);choices=[]
            for comp in comps:
                coords=comp['coords'];fraction=float(np.mean(np.all((coords>=low)&(coords<high),axis=1)))
                if fraction<.1:continue
                lo,hi=coords.min(0)-.5,coords.max(0)+.5;inter=np.maximum(0,np.minimum(high,hi)-np.maximum(low,lo)).prod();iou=inter/max(1e-12,(high-low).prod()+(hi-lo).prod()-inter);choices.append((iou,comp))
            assert choices,(cid,r['original_index']);comp=max(choices,key=lambda z:z[0])[1];lo=comp['coords'].min(0);hi=comp['coords'].max(0)+1;small=np.zeros(tuple(hi-lo),np.uint8);small[tuple((comp['coords']-lo).T)]=1;small=np.pad(small,1);lo-=1;coords=crop_coordinates(low,high);target=map_coordinates(small,(coords-lo[:,None,None,None]).reshape(3,-1),order=0,mode='constant',cval=0,prefilter=False).reshape(32,32,32);valid=np.all((coords>=0)&(coords<=(np.array(shape)-1)[:,None,None,None]),axis=0);target=(target*SUPPORT*valid).astype(np.uint8)
            if not target.any():skipped.append({'case_id':cid,'detector_index':r['original_index'],'reason':'no positive32cube target voxels'});continue
            if not geometry_checked:
                arr,_,_=load_image(cid);expected=crop_volume(arr,low,high,1).astype(np.float16);error=float(np.max(abs(expected.astype(np.float32)-images[i,0].astype(np.float32))));assert error<=.001;write_json(RUN/'GEOMETRY_TEST.json',{'case_id':cid,'image_cache_max_absolute_error':error,'shape':[32,32,32],'GT_image_affine_verified':True,'target_has_foreground':True,'source_only':True});geometry_checked=True;del arr
            gi=gt_lookup[cid,comp['class_id'],tuple(comp['coords'].min(0)-.5),tuple(comp['coords'].max(0)+.5)];xx.append(images[i,0]);yy.append(target);rr.append({'case_id':cid,'source_class_id':comp['class_id'],'component_id':comp['component_id'],'sample_index':1000+r['original_index'],'low':r['low'],'high':r['high'],'detector_index':r['original_index'],'score':r['score'],'source_filter_record_index':i,'matched_GT_base_row':gi,'source':'actual_CT_detector_positive'})
        np.savez_compressed(dest,x=np.asarray(xx,np.float16),y=np.asarray(yy,np.uint8));write_json(meta,rr);caches.append((dest,meta));print('E20_CT_CROPS',n,len(groups),cid,len(rr),flush=True)
    oldx=np.load(BASE/'features/images.npy',mmap_mode='r');oldy=np.load(BASE/'features/targets.npy',mmap_mode='r');count=sum(len(json.loads(meta.read_text())) for _,meta in caches);assert len(oldx)==2997 and len(old)==5994;x=np.lib.format.open_memmap(RUN/'features/images.npy',mode='w+',dtype=np.float16,shape=(len(oldx)+count,32,32,32));y=np.lib.format.open_memmap(RUN/'features/targets.npy',mode='w+',dtype=np.uint8,shape=x.shape);x[:len(oldx)]=oldx;y[:len(oldx)]=oldy;mapping=np.load(BASE/'features/rows.npz');idx=mapping['crop_index'].tolist();mir=mapping['mirrored'].tolist();records=old.copy();offset=len(oldx)
    for dest,meta in caches:
        data=np.load(dest);rr=json.loads(meta.read_text());x[offset:offset+len(rr)]=data['x'];y[offset:offset+len(rr)]=data['y']
        for j,r in enumerate(rr):
            for flip in [False,True]:idx.append(offset+j);mir.append(flip);records.append({**r,'view':'mirror' if flip else 'original','binary_shape_only_mirror':flip})
        offset+=len(rr)
    x.flush();y.flush();assert offset==len(x) and np.array_equal(x[:len(oldx)],oldx) and np.array_equal(y[:len(oldy)],oldy);np.savez_compressed(RUN/'features/rows.npz',crop_index=idx,mirrored=mir);(RUN/'features/train_records.jsonl').write_text('\n'.join(json.dumps(r) for r in records)+'\n');devgroups={group(c) for c in base_split['development_cases']};tr=[i for i,r in enumerate(records) if group(r['case_id']) not in devgroups];assert {i for i in tr if i<len(old)}==set(base_split['train_rows']);lookup={(r['case_id'],r['detector_index']):i for i,r in enumerate(records) if i>=len(old) and r['view']=='original'};pairs=pre['pairs'];dv=[lookup[r['case_id'],r['detector_index']] for r in pairs];gtrows=[r['GT_row'] for r in pairs];assert all(records[i]['matched_GT_base_row']==j for i,j in zip(dv,gtrows));assert not {group(records[i]['case_id']) for i in tr}&devgroups
    write_json(RUN/'source_split.json',{**base_split,'train_rows':tr,'development_detector_rows':dv,'development_GT_paired_rows':gtrows,'source_CT_pairs':len(dv)});write_json(RUN/'features/SOURCE_READY.json',{'new_CT_detector_crops':count,'n_crops':len(x),'n_rows':len(records),'old_E16_images_and_targets_bit_identical':True,'new_crop_source_cases':len(groups),'skipped':skipped,'images_sha256':sha256_file(RUN/'features/images.npy'),'targets_sha256':sha256_file(RUN/'features/targets.npy'),'records_sha256':sha256_file(RUN/'features/train_records.jsonl')});print('E20_SOURCE_READY',count,len(records),flush=True)

if __name__=='__main__':main()
