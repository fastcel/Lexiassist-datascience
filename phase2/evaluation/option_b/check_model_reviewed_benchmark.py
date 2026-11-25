"""Check Option B model-screened 100Q benchmark without claiming human validation."""
import csv,hashlib,json
from pathlib import Path
from collections import Counter
HERE=Path(__file__).resolve().parent
B=HERE/'benchmark'/'benchmark_100_DRAFT.csv'
F=HERE/'frozen'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
 a=load(B);m=json.loads((F/'freeze_manifest.json').read_text(encoding='utf-8'))
 assert len(a)==100 and Counter(x['split'] for x in a)=={'dev':40,'test':60}
 assert len(set(x['query_id'] for x in a))==100
 assert all(x['question_status']=='MODEL_REVIEWED' and x['review_date']=='2026-11-25' for x in a)
 assigned = {}
 for name,reviewer,count in [('A','Meerab',34),('B','Shehryar',33),('C','Momna',33)]:
  batch = load(HERE/'benchmark'/'review_batches'/f'Reviewer_{name}.csv')
  assert len(batch)==count
  for r in batch:
   assert r['reviewer']==reviewer and r['query_id'] not in assigned
   assigned[r['query_id']]=r
 assert len(assigned)==100
 assert all(x==assigned[x['query_id']] for x in a)
 assert m['status']=='FROZEN_MODEL_PASSAGE_REVIEWED'
 assert sha(B)==m['source_benchmark_sha256']
 for split in ['dev','test']:
  p=F/f'{split}_questions.csv'
  assert sha(p)==m[split+'_sha256'] and len(load(p))==m[split]
 print('CHECK PASSED: 100 model-reviewed source-linked questions, dev=40 and test=60.')
 print('NOTE: The benchmark was not independently human or legal-expert verified.')
if __name__=='__main__':main()
