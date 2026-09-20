"""Verified local training cache, durable arrays remain authoritative."""
import json,os,shutil,time
from pathlib import Path
from scripts.astra6_e33.prepare import RUN,PREP,sha,write
CACHE=Path('/tmp/topaneu_e33_source_only_cache_20260910')
def restore_missing():
 manifest=RUN/'LOCAL_CACHE_MANIFEST.json'
 if not manifest.exists():return
 for name,r in json.loads(manifest.read_text())['files'].items():
  link=PREP/'D3V001_3d/imagesTr'/name
  if not link.exists():
   original=RUN/'persistent_preprocessed'/name;assert original.exists();temp=link.with_suffix(link.suffix+'.restore_tmp');temp.symlink_to(original);os.replace(temp,link)
def main():
 assert (RUN/'SOURCE_READY.json').exists();restore_missing();CACHE.mkdir(exist_ok=True);records=[json.loads(f.read_text()) for f in (RUN/'preprocessing_cases').glob('*.json')];files={n:d for r in records for n,d in r['files'].items()};assert len(files)==1068;manifest=RUN/'LOCAL_CACHE_MANIFEST.json';m=json.loads(manifest.read_text()) if manifest.exists() else {'files':{},'durable_originals_preserved':True,'complete':False};needed=sum(d['bytes'] for n,d in files.items() if not (CACHE/n).exists());assert shutil.disk_usage(CACHE).free>needed+30*2**30
 for i,(name,d) in enumerate(sorted(files.items()),1):
  local=CACHE/name
  if name in m['files'] and local.exists():assert local.stat().st_size==d['bytes']
  else:
   original=RUN/'persistent_preprocessed'/name;temp=local.with_suffix(local.suffix+'.copy_tmp');shutil.copyfile(original,temp);assert sha(temp)==d['sha256'];os.replace(temp,local);m['files'][name]={**d,'local':str(local),'durable_original':str(original)};write(manifest,m)
  link=PREP/'D3V001_3d/imagesTr'/name;assert link.is_symlink();temp=link.with_suffix(link.suffix+'.cache_link_tmp');temp.symlink_to(local);os.replace(temp,link)
  if i%50==0:print('E33 LOCAL CACHE',i,len(files),flush=True)
 m.update(complete=True,total_bytes=sum(d['bytes'] for d in files.values()),timestamp=time.time());write(manifest,m)
if __name__=='__main__':main()
