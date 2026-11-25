"""Union result pools so every method is scored against identical judgments.

Use for development (seven baseline methods + tuning methods) and held-out test.
No labels invented or inherited: output relevance_grade stays BLANK.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def save(p,rows,headers):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=headers);w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--inputs',nargs='+',type=Path,required=True,help='Result folders, e.g. option_b_dev_baselines and option_b_dev_grid')
    ap.add_argument('--output',type=Path,required=True,help='NEW combined result folder')
    args=ap.parse_args()
    if len(args.inputs)<2:raise SystemExit('Need at least two result folders')
    if args.output.exists():raise SystemExit('Combined output exists; refusing to overwrite')
    ranking=[];candidate={};question_sets=[];method_sets=[]
    for folder in args.inputs:
        ranks=read(folder/'rankings.csv');pool=read(folder/'judgments_TO_FILL.csv')
        if not ranks or not pool:raise SystemExit(f'Empty result file in {folder}')
        ranking.extend(ranks)
        question_sets.append({r['query_id'] for r in ranks});method_sets.append({r['method'] for r in ranks})
        for r in pool:
            k=(r['query_id'],r['chunk_key'])
            if k in candidate and candidate[k]['passage_text']!=r['passage_text']:
                raise SystemExit(f'Conflicting source passage at {k}')
            candidate[k]=dict(r,relevance_grade='')
    if not all(q==question_sets[0] for q in question_sets):raise SystemExit('Input runs did not use the exact same question IDs')
    for i in range(len(method_sets)):
        for j in range(i+1,len(method_sets)):
            if method_sets[i]&method_sets[j]:raise SystemExit(f'Method names duplicated across runs: {method_sets[i]&method_sets[j]}')
    seen=set()
    for r in ranking:
        k=(r['query_id'],r['method'],r['rank'])
        if k in seen:raise SystemExit(f'Duplicate ranking {k}')
        seen.add(k)
        if (r['query_id'],r['chunk_key']) not in candidate:raise SystemExit(f'Missing judged candidate for {k}')
    args.output.mkdir(parents=True)
    save(args.output/'rankings.csv',ranking,list(ranking[0]))
    pool=sorted(candidate.values(),key=lambda r:(r['query_id'],r['doc_id'],int(r['chunk_index'])))
    save(args.output/'judgments_TO_FILL.csv',pool,list(pool[0]))
    (args.output/'run_metadata.json').write_text(json.dumps({'question_count':len(question_sets[0]),
      'methods':sorted(set().union(*method_sets)),'candidate_pool_size':len(pool),
      'warning':'All relevance grades intentionally blank. Do NOT score before human review. All methods now share same judged pool.'},indent=2),encoding='utf-8')
    print('Shared pool has',len(pool),'unique question-passage pairs for',len(set().union(*method_sets)),'methods.')
    print('Review all candidates and save as judgments_reviewed.csv before scoring.')
if __name__=='__main__':main()
