"""Evaluate one preselected method per retriever on the 60 held-out questions.

Freezes the method selection based on DEV metrics before reading test scores.
Test relevance judgments must still be independently reviewed before scoring.
"""
from __future__ import annotations
import argparse,csv,json,hashlib,sys,time
from pathlib import Path
from collections import defaultdict
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'phase2'))
sys.path.insert(0,str(ROOT/'phase2'/'shared'))
sys.path.insert(0,str(ROOT/'phase2'/'evaluation'))
from test_retrieval import check_same_corpus,import_retriever
from query_expander import expand_query
from tune_rrf_expansion import retrieve,csvwrite
from run_experiments import execute

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--frozen',type=Path,default=HERE/'frozen')
    ap.add_argument('--output',type=Path,default=ROOT/'phase2/evaluation/results/option_b_test_final')
    args=ap.parse_args()
    if args.output.exists():raise SystemExit('Test output exists; refusing to overwrite to discourage test iteration')
    cfgp=args.frozen/'selected_retrieval_config.json';qsp=args.frozen/'test_questions.csv';mp=args.frozen/'freeze_manifest.json'
    if not all(p.is_file() for p in (cfgp,qsp,mp)):raise SystemExit('Benchmark and dev configuration must be frozen first')
    cfg=json.loads(cfgp.read_text(encoding='utf-8'));mf=json.loads(mp.read_text(encoding='utf-8'))
    if mf['test_sha256']!=digest(qsp) or cfg['benchmark_freeze_sha256']!=digest(mp):
        raise SystemExit('Test set or freeze manifest changed; abort')
    with qsp.open(encoding='utf-8-sig',newline='') as f:qs=list(csv.DictReader(f))
    if len(qs)!=60:raise SystemExit('Expected exactly 60 sealed test questions')
    check_same_corpus()
    rets={v:import_retriever(v)() for v in ('lightweight','pretrained')}
    pool={};rankings=[];latencies=defaultdict(list)
    for q in qs:
        print(q['query_id'],flush=True)
        for v,setting in cfg['selected'].items():
            t=time.perf_counter()
            if setting['kind']=='baseline':
                rows=execute(setting['method'],rets[v],q['query'])
            else:
                sub=expand_query(q['query'],n=setting['expand_n']) if setting['expand_n'] else []
                rows=retrieve(rets[v],v,q['query'],sub,setting['rrf_k'])
            dt=round((time.perf_counter()-t)*1000,2)
            latencies[f'selected_{v}'].append(dt)
            for i,r in enumerate(rows,1):
                c=r['chunk'];key=f'{c.doc_id}::chunk_{c.chunk_index}'
                rankings.append({'query_id':q['query_id'],'method':f'selected_{v}','rank':i,'doc_id':c.doc_id,'chunk_index':c.chunk_index,
                 'chunk_key':key,'citation':c.citation,'jurisdiction':c.jurisdiction,'url':c.url,'latency_ms':dt})
                pool[(q['query_id'],key)]={'query_id':q['query_id'],'query':q['query'],'domain':q['domain'],'chunk_key':key,
                 'doc_id':c.doc_id,'chunk_index':c.chunk_index,'citation':c.citation,'url':c.url,'jurisdiction':c.jurisdiction,
                 'passage_text':c.text,'relevance_grade':''}
    args.output.mkdir(parents=True)
    csvwrite(args.output/'rankings.csv',['query_id','method','rank','doc_id','chunk_index','chunk_key','citation','jurisdiction','url','latency_ms'],rankings)
    csvwrite(args.output/'judgments_TO_FILL.csv',['query_id','query','domain','chunk_key','doc_id','chunk_index','citation','url','jurisdiction','passage_text','relevance_grade'],sorted(pool.values(),key=lambda x:(x['query_id'],x['doc_id'],int(x['chunk_index']))))
    (args.output/'run_metadata.json').write_text(json.dumps({'test_questions':60,'selected_config':cfg['selected'],
      'mean_latency_ms':{k:round(sum(v)/len(v),2) for k,v in latencies.items()},'ungraded_unique_candidates':len(pool),
      'warning':'No evaluation scores until every passage is manually judged; do not tune using test results.'},indent=2),encoding='utf-8')
    print('Saved held-out test data; review',len(pool),'unique passages before scoring.')
if __name__=='__main__':main()
