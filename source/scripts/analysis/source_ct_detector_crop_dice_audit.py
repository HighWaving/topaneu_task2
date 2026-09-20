"""Paired CT source GT/D crop Dice using the source-held-out E16 helper."""
import json
import numpy as np,torch
from scipy.ndimage import map_coordinates
from scripts.astra6_e01.e01_common import P,DATA,load_nifti,write_json
from scripts.astra6_e04.run_e04 import Segmenter,SegData,PRIOR,SUPPORT,crop_coordinates,component_records_fast,largest

def main():
    torch.set_num_threads(4);base=P/'artifacts/astra6_e16_CT_expanded_segmentation_20260909';bank=P/'artifacts/astra6_e11_CT_expanded_fp_filter_20260909';out=P/'artifacts/source_CT_detector_crop_dice_audit_20260909';out.mkdir(exist_ok=True)
    config={'hypothesis':'CT S may also have a source accuracy gap between GT-jitter training crops and actual detector crops, as previously demonstrated and improved for MR E17','paired_source':'28CT source-development lesions mapped in the extent audit; same lesions and source-held-out E16 development_last checkpoint for both crop types','training_trigger':'GT-box mean crop Dice exceeds detector-box mean crop Dice by>=.02','next_if_pass':'Add actual CT detector-positive crops to unchanged E16 GT bank; same architecture, support, optimizer and13development+13finalepochs; CT-only comparison to CT_E16','no_CT5_or_MR40_access':True};write_json(out/'config.json',config)
    pairs=json.loads((P/'artifacts/source_CT_segmentation_extent_audit_20260909/RESULT.json').read_text())['rows'];assert len(pairs)>=20;records=[json.loads(s) for s in (base/'features/train_records.jsonl').read_text().splitlines()];fr=[json.loads(s) for s in (bank/'features/records.jsonl').read_text().splitlines()];lookup={(r['case_id'],r['original_index']):i for i,r in enumerate(fr) if r['augmentation']==0};images=np.load(bank/'features/images.npy',mmap_mode='r');groups={}
    for j,r in enumerate(pairs):groups.setdefault(r['case_id'],[]).append((j,r))
    dx=np.empty((len(pairs),2,32,32,32),np.float32);dy=np.empty((len(pairs),32,32,32),np.uint8)
    for cid,items in sorted(groups.items()):
        gt,aff,shape=load_nifti(DATA/f'location_masks/{cid}.nii.gz');comps=component_records_fast(gt)
        for j,r in items:
            g=records[r['GT_row']];comp=next(c for c in comps if c['class_id']==g['class_id'] and np.array_equal(c['coords'].min(0)-.5,g['low']) and np.array_equal(c['coords'].max(0)+.5,g['high']));lo=comp['coords'].min(0);hi=comp['coords'].max(0)+1;small=np.zeros(tuple(hi-lo),np.uint8);small[tuple((comp['coords']-lo).T)]=1;small=np.pad(small,1);lo-=1;coords=crop_coordinates(r['low'],r['high']);target=map_coordinates(small,(coords-lo[:,None,None,None]).reshape(3,-1),order=0,mode='constant',cval=0,prefilter=False).reshape(32,32,32);valid=np.all((coords>=0)&(coords<=(np.array(shape)-1)[:,None,None,None]),axis=0);dy[j]=target*SUPPORT*valid;row=lookup[cid,r['detector_index']];assert fr[row]['y']==1;dx[j]=np.stack([images[row,0],PRIOR])
        print('CT_SOURCE_CROP_PAIR',cid,flush=True)
    model=Segmenter().eval();model.load_state_dict(torch.load(base/'model/development_last.pt',map_location='cpu',weights_only=False)['state_dict']);gtdata=SegData(base,[r['GT_row'] for r in pairs]);gx=np.stack([gtdata[i][0].numpy() for i in range(len(pairs))]);gy=np.stack([gtdata[i][1].numpy()[0] for i in range(len(pairs))]);scores={}
    with torch.inference_mode():
        for name,x,y in [('GT',gx,gy),('detector',dx,dy)]:
            ps=model(torch.from_numpy(x)).sigmoid().numpy()[:,0]*SUPPORT;ss=[]
            for pp,target in zip(ps,y):
                mask=largest(pp>=.5)
                if not mask.any():mask=PRIOR>0
                truth=target>0;ss.append(float(2*(mask&truth).sum()/max(1,mask.sum()+truth.sum())))
            scores[name]={'mean_Dice':float(np.mean(ss)),'scores':ss}
    gap=scores['GT']['mean_Dice']-scores['detector']['mean_Dice'];np.savez_compressed(out/'paired_detector_crops.npz',x=dx[:,0].astype(np.float16),y=dy);result={'n':len(pairs),'scores':scores,'Dice_gap':gap,'training_trigger_passed':gap>=.02,'pairs':pairs,'source_detector_training_exposure_disclosed':True};write_json(out/'RESULT.json',result);print('CT_SOURCE_CROP_DICE_RESULT',len(pairs),scores['GT']['mean_Dice'],scores['detector']['mean_Dice'],'gap',gap,'trigger',gap>=.02,flush=True)

if __name__=='__main__':main()
