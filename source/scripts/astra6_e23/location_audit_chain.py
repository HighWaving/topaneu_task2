"""Wait for OOF proposals, then run a bounded CPU-only source location audit."""
import os,time,subprocess
from pathlib import Path
P=Path(__file__).resolve().parents[2];R=P/'artifacts/astra6_e23_MR_oof_candidates_20260909'
if __name__=='__main__':
 while not all((R/f'checkpoints/fold{i}_OOF_COMPLETE.json').exists() for i in [0,1]):time.sleep(15)
 env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4');subprocess.run([str(P.parent/'conda_envs/nnunet_v100/bin/python'),'-m','scripts.astra6_e23.audit_oof_location'],cwd=P,env=env,check=True)
