"""Run dev-only RRF/query-expansion grid using EXISTING shared indexes.

Do not touch working retrievers, existing indexes, website, or previous results.
Save raw rankings and a candidate-pool worksheet. No relevance metrics are
calculated until every new question-passage pair is independently reviewed.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'phase2'))
sys.path.insert(0,str(ROOT/'phase2'/'shared'))
from test_retrieval import check_same_corpus,import_retriever
from query_expander import expand_query

def parse_values(s):
    try:v=[int(x) for x in s.split(',')]
    except ValueError:raise argparse.ArgumentTypeError('Use comma-separated integers')
    if not v or any(x<0 for x in v):raise argparse.ArgumentTypeError('Values must be nonnegative')
    return sorted(set(v))

def csvwrite(p,headers,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=headers);wr.writeheader();wr.writerows(rows)

def load_questions(path):
    with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if not rows or not {'query_id','query','domain'}<=set(rows[0]):raise ValueError('Wrong question CSV format')
    return rows

def frozen_guard(questions,allow_dev_only=True):
    folder=questions.parent
    mf=folder/'freeze_manifest.json'
    if not mf.is_file():raise ValueError('Run validate_freeze_benchmark.py after reviewing questions first')
    manifest=json.loads(mf.read_text(encoding='utf-8'))
    sha=hashlib.sha256(questions.read_bytes()).hexdigest()
    if sha!=manifest['dev_sha256'] or questions.name!='dev_questions.csv':
        raise ValueError('Development-only script: supplied questions are not the frozen dev split')
    return manifest

def retrieve(retriever,variant,question,subqueries,rrf_k,top5=5):
    queries=[question]+subqueries
    if variant=='lightweight':
        dense=retriever._tfidf_search(queries);sparse=retriever._bm25_search(queries)
    else:
        dense=retriever._dense(queries);sparse=retriever._sparse(queries)
    scores={}
    for lst in dense+sparse:
        for rank,i in enumerate(lst):scores[i]=scores.get(i,0.0)+1/(rrf_k+rank+1)
    fused=sorted(scores,key=lambda i:scores[i],reverse=True)
    ranked=retriever._rerank(question,fused[:10]);return retriever._dedupe(ranked)[:top5]

def collect(questions,variants,rrfs,expands,output,condition_prefix=''):
    check_same_corpus()
    models={v:import_retriever(v)() for v in variants}
    pool={};ranks=[];times=defaultdict(list)
    for n,q in enumerate(questions,1):
        print(f"[{n}/{len(questions)}] {q['query_id']}",flush=True)
        for v in variants:
            re=models[v]
            for k in rrfs:
                for e in expands:
                    name=f'{condition_prefix}{v}_rrf{k}_expand{e}'
                    sub=expand_query(q['query'],n=e) if e else []
                    t=time.perf_counter();found=retrieve(re,v,q['query'],sub,k);dt=(time.perf_counter()-t)*1000
                    times[name].append(dt)
                    for ix,row in enumerate(found,1):
                        c=row['chunk'];key=f'{c.doc_id}::chunk_{c.chunk_index}'
                        ranks.append({'query_id':q['query_id'],'method':name,'rank':ix,'doc_id':c.doc_id,
                                      'chunk_index':c.chunk_index,'chunk_key':key,'citation':c.citation,
                                      'jurisdiction':c.jurisdiction,'url':c.url,'latency_ms':round(dt,2)})
                        pool[(q['query_id'],key)]={'query_id':q['query_id'],'query':q['query'],'domain':q['domain'],
                           'chunk_key':key,'doc_id':c.doc_id,'chunk_index':c.chunk_index,'citation':c.citation,
                           'url':c.url,'jurisdiction':c.jurisdiction,'passage_text':c.text,'relevance_grade':''}
    output.mkdir(parents=True,exist_ok=False)
    csvwrite(output/'rankings.csv',['query_id','method','rank','doc_id','chunk_index','chunk_key','citation','jurisdiction','url','latency_ms'],ranks)
    csvwrite(output/'judgments_TO_FILL.csv',['query_id','query','domain','chunk_key','doc_id','chunk_index','citation','url','jurisdiction','passage_text','relevance_grade'],
        sorted(pool.values(),key=lambda r:(r['query_id'],r['doc_id'],int(r['chunk_index']))))
    (output/'run_metadata.json').write_text(json.dumps({'questions':len(questions),'variants':variants,'rrf_constants':rrfs,
       'subqueries':expands,'top_k':5,'total_unique_judgments':len(pool),'mean_retrieval_ms':{k:round(sum(v)/len(v),2) for k,v in times.items()},
       'warning':'All relevance labels must be reviewed before scoring. Development results only.'},indent=2),encoding='utf-8')
    print('Saved',output,':',len(pool),'unique question-passage pairs to review.')
    return len(pool)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--questions',type=Path,default=HERE/'frozen'/'dev_questions.csv')
    ap.add_argument('--rrf',type=parse_values,default=[30,60,90])
    ap.add_argument('--expand',type=parse_values,default=[0,1])
    ap.add_argument('--variants',nargs='+',choices=['lightweight','pretrained'],default=['lightweight','pretrained'])
    ap.add_argument('--limit',type=int,default=None,help='Development smoke test only')
    ap.add_argument('--output',type=Path,default=ROOT/'phase2/evaluation/results/option_b_dev_grid')
    args=ap.parse_args();frozen_guard(args.questions)
    qs=load_questions(args.questions)
    if args.limit is not None:
        if args.limit<1:ap.error('limit must be >=1')
        qs=qs[:args.limit]
    if args.output.exists():raise SystemExit(f'Output already exists: {args.output}. Choose a fresh --output directory.')
    collect(qs,args.variants,args.rrf,args.expand,args.output)
if __name__=='__main__':main()
