"""GPU dependency -> full training -> locked evaluation -> adoption evidence."""
import os,sys,json,subprocess
from scripts.astra6_e20.prepare import RUN,P
from scripts.astra6_e01.e01_common import write_json

def main():
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='GPU-a643ded3-193b-58e8-b362-be93dc8eac14',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    subprocess.run([sys.executable,'-m','scripts.astra6_e20.train'],cwd=P,env=env,check=True)
    if not json.loads((RUN/'model/LOCKED.json').read_text())['source_gate_passed']:
        write_json(RUN/'ADOPTION.json',{'adopt':False,'stage':'source_only','full_training_completed':True,'reason':'Same-lesion source detector crop Dice did not improve; CT5 not evaluated'});return
    official=str(P/'.venv_official_eval_20260909/bin/python')
    for cmd in [[sys.executable,'-m','scripts.astra6_e20.infer'],[official,'-m','scripts.analysis.current_official_evaluation','--version','CT_E20'],[official,'-m','scripts.analysis.compare_final_versions','--baseline','CT_E16','--version','CT_E20','--output',str(RUN/'evaluation/paired_official_comparison.json')],[sys.executable,'-m','scripts.astra6_e20.diagnose']]:subprocess.run(cmd,cwd=P,env=env,check=True)
if __name__=='__main__':main()
