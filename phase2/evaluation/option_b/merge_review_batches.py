"""Merge completed teammate review batches without hiding missing/duplicate rows."""
import argparse,csv
from pathlib import Path
HERE=Path(__file__).resolve().parent

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--master',type=Path,default=HERE/'benchmark'/'benchmark_100_DRAFT.csv')
    ap.add_argument('--batches',nargs='+',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=HERE/'benchmark'/'benchmark_100_REVIEWED.csv')
    args=ap.parse_args();master=read(args.master)
    byid={r['query_id']:r for r in master};assigned={}
    for file in args.batches:
        for row in read(file):
            qid=row.get('query_id','')
            if qid not in byid:raise SystemExit(f'Unknown question {qid} in {file}')
            if qid in assigned:raise SystemExit(f'Question {qid} appears in two batches')
            if row.get('split')!=byid[qid].get('split'):
                raise SystemExit(f'Question {qid} split was modified; refusing to merge')
            assigned[qid]=row
    missing=sorted(set(byid)-set(assigned))
    if missing:raise SystemExit(f'Missing assignments: {missing}')
    if args.output.exists():raise SystemExit('Output exists; refusing to overwrite')
    with args.output.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(master[0]));w.writeheader()
        for r in master:w.writerow(assigned[r['query_id']])
    print('Merged all',len(master),'questions into',args.output)
    print('Next: run validate_freeze_benchmark.py --benchmark',args.output)
if __name__=='__main__':main()
