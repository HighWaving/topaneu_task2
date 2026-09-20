"""Release only our sequential-copy page cache, preserving all file contents."""
from pathlib import Path
import os,json,time
P=Path(__file__).resolve().parents[2];R=P/'artifacts/astra6_e23_MR_oof_candidates_20260909';M=R/'evidence/LOCAL_CACHE_MANIFEST.json';C=Path('/tmp/topaneu_e23_verified_cache_20260909');seen=set()
while True:
 a=json.loads(M.read_text())
 for name,v in a['files'].items():
  if name in seen:continue
  f=Path(v['local']);assert f.parent==C and f.name==name and f.is_file()
  # This private path is used only by E23. No global cache flush or original-file advice.
  with f.open('rb') as stream:os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
  seen.add(name)
 (R/'evidence/LOCAL_COPY_CACHE_RELEASE.json').write_text(json.dumps({'n_files':len(seen),'scope':str(C),'operation':'POSIX_FADV_DONTNEED once per verified private copied file; bytes unchanged, durable originals and other task files untouched','purpose':'Avoid sequential verification copy filling cgroup file cache and delaying active training/imports','memory_current':Path('/sys/fs/cgroup/memory.current').read_text().strip()},indent=2)+'\n')
 if a.get('complete'):break
 time.sleep(15)
