"""Wait for our existing source GPU work, then populate MR F calibration cases."""
import json,subprocess,sys,time
from scripts.astra6_e19.prepare import RUN,CACHE,P
from scripts.astra6_e01.e01_common import write_json

def ready(path):
    try:return json.loads(path.read_text()).get('complete',False)
    except (OSError,json.JSONDecodeError):return False

def main():
    start=time.monotonic()
    while not ((CACHE/'RESULT.json').exists() and ready(CACHE/'PREFETCH_PROGRESS.json')):
        assert time.monotonic()-start<14400,'Existing source audit exceeded its four-hour budget';time.sleep(10)
    commands=[];jobs=[];logs=[]
    for part,gpu in enumerate(['GPU-a643ded3-193b-58e8-b362-be93dc8eac14','GPU-f7491bbb-3972-1264-755a-63dec96b4a7d']):
        cmd=[sys.executable,'-m','scripts.analysis.prefetch_source_mr_ta36','--gpu',gpu,'--case-list',str(RUN/'TA36_REQUIRED_CASES.json'),'--part',str(part),'--parts','2','--progress-name',f'E19_PREFETCH_PART{part}.json'];commands.append(cmd);log=(RUN/f'logs/TA36_part{part}.log').open('w');logs.append(log);jobs.append(subprocess.Popen(cmd,cwd=P,stdout=log,stderr=subprocess.STDOUT))
    write_json(RUN/'TA36_JOBS.json',{'commands':commands,'pids':[p.pid for p in jobs],'existing_source_predictions_reused':True})
    codes=[p.wait() for p in jobs]
    for log in logs:log.close()
    assert codes==[0,0],codes
    cases=json.loads((RUN/'TA36_REQUIRED_CASES.json').read_text())['cases'];assert all((CACHE/c/'predicted_vessel.nii.gz').exists() and (CACHE/c/'PROVENANCE.json').exists() for c in cases)
    write_json(RUN/'TA36_READY.json',{'cases':cases,'n_cases':len(cases),'complete':True,'seconds_including_wait':time.monotonic()-start})
    while not (RUN/'model/TRAINING_COMPLETE.json').exists():
        assert time.monotonic()-start<21600,'E19 training readiness exceeded its six-hour workflow budget';time.sleep(10)
    subprocess.run([sys.executable,'-m','scripts.astra6_e19.calibrate'],cwd=P,check=True)

if __name__=='__main__':main()
