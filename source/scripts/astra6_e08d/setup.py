from pathlib import Path
import json,shutil
import numpy as np
from scripts.astra6_e01.e01_common import P,sha256_file,write_json
PARENT=P/'artifacts/astra6_e08_mr_only_fp_filter_20260909';RUN=P/'artifacts/astra6_e08_deployed_pool_calibration_20260909'
def main():
 for s in ['model','evaluation','logs','predictions/mr_center2_k05']:(RUN/s).mkdir(parents=True,exist_ok=True)
 audit=json.loads((P/'reports/SOURCE_FILTER_CALIBRATION_AUDIT_20260909.json').read_text());src=audit['models']['E08']['MR'];threshold=float(np.nextafter(src['minimum_deployed_pool_positive']['p'],0.));assert src['deployed_pool_positive']==24;lock=json.loads((PARENT/'model/LOCKED.json').read_text());assert sha256_file(PARENT/'model/final_last.pt')==lock['model_sha256'];shutil.copy2(PARENT/'model/final_last.pt',RUN/'model/final_last.pt')
 write_json(RUN/'config.json',{'experiment':'E08D','new_training':False,'parent_trained_model':str(PARENT),'hypothesis':'calibrate only on candidates eligible for actual deployment, rather than low-score proposals never sent to the filter','fixed_policy':'minimum source-development positive probability among score>=.3, original top5 candidates; preserve all24 observed positives','threshold':threshold,'source_audit_sha256':sha256_file(P/'reports/SOURCE_FILTER_CALIBRATION_AUDIT_20260909.json'),'no_MR40_labels_or_probabilities_used_to_choose_threshold':True,'one_source_defined_operating_point_only':True,'baseline':'E08 and E06; E04 fallback','gate':'all-six Pareto improvement plus no lost matched lesions; no further threshold tuning on MR40'})
 write_json(RUN/'model/THRESHOLD.json',{'threshold':threshold,'source_positive_count':24,'source_only':True,'selection':'minimum positive probability in actual source deployment pool; fixed before this branch predictions','parent_all_pool_threshold':json.loads((PARENT/'model/THRESHOLD.json').read_text())['threshold']});lock.update(threshold_sha256=sha256_file(RUN/'model/THRESHOLD.json'),parent_model=str(PARENT),config_sha256=sha256_file(RUN/'config.json'),new_training=False);write_json(RUN/'model/LOCKED.json',lock)
if __name__=='__main__':main()
