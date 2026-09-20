import argparse,json
from scripts.astra6_e32.common import RUN
from scripts.astra6_e01.e01_common import write_json

def main(arm):
 out=RUN/arm;dest=out/'evaluation/lesion_diagnostics.json'
 if not dest.exists():
  import scripts.astra6_e28.evaluate as d
  d.RUN=out;d.main()
 r=json.loads(dest.read_text());name='E32_'+arm
 if 'E28' in r:r[name]=r.pop('E28')
 r['strata']={}
 for key in ['size_bin','class']:
  groups={}
  for row in r['paired_rows']:
   q=groups.setdefault(str(row[key]),{'n':0,'matched_before':0,'matched_after':0,'location_rescued':0,'location_lost':0,'paired_Dice_deltas':[]});q['n']+=1;q['matched_before']+=row['before']['matched'];q['matched_after']+=row['after']['matched'];q['location_rescued']+=row['location_rescued'];q['location_lost']+=row['location_lost']
   if row['before']['matched'] and row['after']['matched']:q['paired_Dice_deltas'].append(row['after']['binary_dice']-row['before']['binary_dice'])
  r['strata'][key]=groups
 write_json(dest,r)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['normalized'],required=True);main(p.parse_args().arm)
