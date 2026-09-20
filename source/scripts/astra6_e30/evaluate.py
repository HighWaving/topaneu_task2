import json
from scripts.astra6_e30.train import RUN
from scripts.astra6_e01.e01_common import write_json

def main():
 output=RUN/'evaluation/lesion_diagnostics.json'
 if not output.exists():
  import scripts.astra6_e28.evaluate as d
  d.RUN=RUN;d.main()
 r=json.loads(output.read_text())
 if 'E28' in r:r['E30']=r.pop('E28')
 r['strata']={}
 for key in ['size_bin','class']:
  groups={}
  for row in r['paired_rows']:
   value=str(row.get(key,row['after'].get(key)))
   q=groups.setdefault(value,{'n':0,'rescued':0,'lost':0});q['n']+=1;q['rescued']+=row['location_rescued'];q['lost']+=row['location_lost']
  r['strata'][key]=groups
 write_json(output,r)
if __name__=='__main__':main()
