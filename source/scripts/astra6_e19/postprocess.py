"""Automatically continue the locked source model into inference and evaluation."""
import json,os,subprocess,sys,time
from scripts.astra6_e19.prepare import RUN,P
from scripts.astra6_e01.e01_common import write_json

def main():
    start=time.monotonic();lock=RUN/'model/LOCKED.json'
    while not lock.exists():
        assert time.monotonic()-start<21600,'E19 calibration exceeded six-hour workflow budget';time.sleep(10)
    if not json.loads(lock.read_text())['source_gate_passed']:
        write_json(RUN/'ADOPTION.json',{'adopt':False,'stage':'source_only','reason':'Predeclared source negative-rejection gate failed; no MR40 evaluation performed','full_training_completed':True});print('E19_SOURCE_GATE_FAILED_RETAIN_R2',flush=True);return
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='GPU-f7491bbb-3972-1264-755a-63dec96b4a7d',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    commands=[[sys.executable,'-m','scripts.astra6_e19.infer'],[str(P/'.venv_official_eval_20260909/bin/python'),'-m','scripts.analysis.current_official_evaluation','--version','E19'],[str(P/'.venv_official_eval_20260909/bin/python'),'-m','scripts.analysis.compare_final_versions','--baseline','E17','--version','E19','--output',str(RUN/'evaluation/paired_official_comparison.json')],[sys.executable,'-m','scripts.astra6_e19.diagnose']]
    for cmd in commands:subprocess.run(cmd,cwd=P,env=env,check=True)

if __name__=='__main__':main()
