"""Measure a complete inference subprocess tree without changing its execution."""
import argparse,subprocess,time,json,os
from pathlib import Path

def tree(root):
 parents={}
 for item in Path('/proc').iterdir():
  if not item.name.isdigit():continue
  try:
   stat=(item/'stat').read_text();parents[int(item.name)]=int(stat[stat.rfind(')')+2:].split()[1])
  except (OSError,ValueError,IndexError):continue
 found={root};changed=True
 while changed:
  extra={pid for pid,parent in parents.items() if parent in found}-found;changed=bool(extra);found.update(extra)
 return sorted(found)

def memory(pids):
 rss=pss=private=0;missing=[]
 for pid in pids:
  try:
   rows=Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines();fields={r.split(':')[0]:int(r.split()[1]) for r in rows if ':' in r and len(r.split())>=2 and r.split()[1].isdigit()};rss+=fields.get('Rss',0);pss+=fields.get('Pss',0);private+=fields.get('Private_Clean',0)+fields.get('Private_Dirty',0)
  except (OSError,ValueError):missing.append(pid)
 return {'RSS_sum_kb':rss,'PSS_sum_kb':pss,'private_sum_kb':private,'unreadable_or_exited_pids':missing}

def main():
 p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--gpu',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();cmd=a.command[1:] if a.command and a.command[0]=='--' else a.command;assert cmd;start=time.monotonic();proc=subprocess.Popen(cmd);samples=[];a.report.parent.mkdir(parents=True,exist_ok=True)
 while True:
  pids=tree(proc.pid);row={'elapsed_seconds':time.monotonic()-start,'pids':pids,**memory(pids)}
  try:
   result=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True,timeout=5)
   row['GPU_used_MiB']=next(int(line.split(',')[1].strip()) for line in result.splitlines() if line.split(',')[0].strip()==a.gpu)
  except (OSError,subprocess.SubprocessError,StopIteration,ValueError):row['GPU_used_MiB']=None
  samples.append(row);state={'command':cmd,'root_pid':proc.pid,'gpu_uuid':a.gpu,'sample_interval_seconds':2,'samples':samples,'complete':proc.poll() is not None,'elapsed_seconds':time.monotonic()-start};state['peak']={k:max(r[k] for r in samples if r.get(k) is not None) for k in ['RSS_sum_kb','PSS_sum_kb','private_sum_kb']};state['peak']['GPU_used_MiB']=max((r['GPU_used_MiB'] for r in samples if r['GPU_used_MiB'] is not None),default=None);state['limitations']='2-second process-tree polling can miss shorter peaks. PSS apportions shared pages; RSS sum double-counts them. GPU memory is whole selected GPU. This is a host subprocess measurement, not an isolated container or T4 test.';a.report.write_text(json.dumps(state,indent=2)+'\n')
  if proc.poll() is not None:break
  time.sleep(2)
 state['exit_code']=proc.returncode;a.report.write_text(json.dumps(state,indent=2)+'\n');raise SystemExit(proc.returncode)
if __name__=='__main__':main()
