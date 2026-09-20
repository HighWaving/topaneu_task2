"""Byte-verified ephemeral IO cache; durable originals are always preserved."""
from pathlib import Path
import json,hashlib,os,shutil,time,argparse
P=Path(__file__).resolve().parents[2];R=P/'artifacts/astra6_e23_MR_oof_candidates_20260909';D=R/'data/Task130FG_TopAneuMR_OOF/preprocessed/D3V001_3d/imagesTr';C=Path('/tmp/topaneu_e23_verified_cache_20260909');M=R/'evidence/LOCAL_CACHE_MANIFEST.json'
def write(obj):
 tmp=M.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,M)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def relink(link,target):
 assert link.parent==D and link.is_symlink();tmp=link.with_suffix('.cache_link_tmp');tmp.symlink_to(target);os.replace(tmp,link)
def restore_missing():
 if not M.exists():return
 a=json.loads(M.read_text())
 for item in a['files'].values():
  link=Path(item['link']);original=Path(item['durable_original']);assert original.exists()
  if not link.exists():relink(link,original)
def main():
 restore_missing();C.mkdir(exist_ok=True);a=json.loads(M.read_text()) if M.exists() else {'purpose':'IO optimization only; all files copied from durable verified preprocessing; loss of /tmp cache automatically restores original targets before training restart','cache':str(C),'files':{}}
 files=sorted(D.glob('*.npy'));assert len(files)==534
 needed=sum(f.stat().st_size for f in files if f.name not in a['files']);assert shutil.disk_usage(C).free>needed+30*2**30
 start=time.time()
 for i,f in enumerate(files,1):
  if f.name in a['files'] and Path(a['files'][f.name]['local']).exists():continue
  original=Path(a['files'][f.name]['durable_original']) if f.name in a['files'] else f.resolve();assert str(original).startswith(str(P.parent))
  dest=C/f.name;temp=dest.with_suffix('.copy_tmp');h=hashlib.sha256()
  with original.open('rb') as src,temp.open('wb') as dst:
   for chunk in iter(lambda:src.read(8*1024*1024),b''):dst.write(chunk);h.update(chunk)
   dst.flush();os.fsync(dst.fileno())
  assert digest(temp)==h.hexdigest();os.replace(temp,dest);a['files'][f.name]={'link':str(f),'durable_original':str(original),'local':str(dest),'bytes':dest.stat().st_size,'sha256':h.hexdigest()};write(a);relink(f,dest)
  print('E23_VERIFIED_LOCAL_CACHE',i,len(files),f.name,round(time.time()-start,1),flush=True)
 a['complete']=True;write(a)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--restore-missing',action='store_true');args=ap.parse_args()
 if args.restore_missing:restore_missing()
 else:main()
