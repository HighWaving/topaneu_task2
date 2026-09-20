"""Separate official macro values from changing valid-class support; no new score."""
import argparse,json,math
from pathlib import Path
P=Path(__file__).resolve().parents[2]
def main(version,baseline='E17'):
 root=P/'artifacts/current_official_20260909';a=json.loads((root/baseline/'official.json').read_text());b=json.loads((root/version/'official.json').read_text());result={}
 for metric in ['PRECISION','RECALL','MCC','DICE','VOLSIM','HD95']:
  av={i:float(a['per_class'][f'{metric}_{i}']) for i in range(1,53) if math.isfinite(float(a['per_class'][f'{metric}_{i}']))};bv={i:float(b['per_class'][f'{metric}_{i}']) for i in range(1,53) if math.isfinite(float(b['per_class'][f'{metric}_{i}']))};common=sorted(av.keys()&bv.keys());result[metric]={'official_before':a['overall'][metric],'official_after':b['overall'][metric],'valid_classes_before':sorted(av),'valid_classes_after':sorted(bv),'removed_valid_classes':sorted(av.keys()-bv.keys()),'added_valid_classes':sorted(bv.keys()-av.keys()),'diagnostic_common_class_mean_before':sum(av[i] for i in common)/len(common) if common else None,'diagnostic_common_class_mean_after':sum(bv[i] for i in common)/len(common) if common else None,'common_class_changes':[{'class_id':i,'before':av[i],'after':bv[i],'delta':bv[i]-av[i]} for i in common if bv[i]!=av[i]]}
 out=root/version/'VALID_CLASS_SUPPORT_ATTRIBUTION.json';out.write_text(json.dumps({'baseline':baseline,'version':version,'metrics':result,'interpretation':'Official means remain authoritative. Common-class means are diagnostic of denominator changes, not a replacement official score; never average raw six measures.'},indent=2)+'\n');print(out)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--version',required=True);p.add_argument('--baseline',default='E17');a=p.parse_args();main(a.version,a.baseline)
