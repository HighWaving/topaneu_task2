"""Fixed-threshold diagnostics separating ranking from operating calibration."""
import numpy as np
from scipy.stats import rankdata

def fixed_threshold_metrics(labels, probabilities, threshold):
    y=np.asarray(labels,dtype=int);p=np.asarray(probabilities,dtype=float)
    assert y.shape==p.shape and y.ndim==1
    assert np.isin(y,[0,1]).all() and np.isfinite(p).all()
    assert ((p>=0)&(p<=1)).all() and np.isfinite(threshold)
    npos=int(y.sum());nneg=len(y)-npos;keep=p>=threshold
    auc=float((rankdata(p)[y==1].sum()-npos*(npos+1)/2)/(npos*nneg)) if npos and nneg else None
    clipped=np.clip(p,1e-7,1-1e-7)
    bce=-(y*np.log(clipped)+(1-y)*np.log(1-clipped))
    return {'n':len(y),'positive':npos,'negative':nneg,'threshold':float(threshold),
            'AUROC':auc,'balanced_BCE':float(.5*(bce[y==1].mean()+bce[y==0].mean())) if npos and nneg else None,
            'TPR':float(keep[y==1].mean()) if npos else None,
            'FPR':float(keep[y==0].mean()) if nneg else None,
            'rejected_positive':int(((y==1)&~keep).sum()),'retained_negative':int(((y==0)&keep).sum()),
            'probability_minus_threshold_quantiles':{str(label):np.percentile(p[y==label]-threshold,[10,50,90]).tolist() if (y==label).any() else None for label in (0,1)}}

if __name__=='__main__':
    # Same ranking, changed operating recall: an AUC alone misses this failure.
    a=fixed_threshold_metrics([0,0,1,1],[.1,.2,.8,.9],.5)
    b=fixed_threshold_metrics([0,0,1,1],[.01,.02,.08,.09],.5)
    assert a['AUROC']==b['AUROC']==1 and a['TPR']==1 and b['TPR']==0
    assert fixed_threshold_metrics([0,1],[.5,.5],.5)['AUROC']==.5
    assert fixed_threshold_metrics([],[],.5)['AUROC'] is None
    print('Ranking versus fixed-threshold shift, ties, and empty strata checks passed; no fitting.')
