"""Freeze best DEV-only retrieval config; never inspect test metrics here."""
import argparse,csv,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dev-results',type=Path,required=True)
    ap.add_argument('--frozen',type=Path,default=HERE/'frozen')
    ap.add_argument('--out',type=Path,default=HERE/'frozen'/'selected_retrieval_config.json')
    args=ap.parse_args()
    manifest=args.frozen/'freeze_manifest.json'
    if not manifest.is_file():raise SystemExit('Missing verified benchmark freeze')
    metrics=args.dev_results/'comparison_metrics.csv'
    if not metrics.is_file():raise SystemExit('Score dev judgments first using existing score_results.py')
    with metrics.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    expected=40
    if not rows or any(int(r['questions'])!=expected for r in rows):
        raise SystemExit('Only full 40-question DEV scores can be used to freeze settings')
    if not {'bm25','tfidf','minilm','lightweight_full','pretrained_full'} <= {r['method'] for r in rows}:
        raise SystemExit('Use the SCORED COMBINED dev pool; the seven baseline methods must be included.')
    for r in rows:
        if r['ndcg_at_5']=='':raise SystemExit('Missing nDCG; review judgments')
    # Select the best per-variant settings by nDCG, latency for a tie; no test data.
    cfg={}
    for var in ('lightweight','pretrained'):
        xs=[r for r in rows if r['method'].startswith(var+'_rrf') or r['method'] in ({'bm25','tfidf','lightweight_full'} if var=='lightweight' else {'minilm','pretrained_full'})]
        if not xs:raise SystemExit(f'No {var} candidates in dev results')
        best=max(xs,key=lambda r:(float(r['ndcg_at_5']),-float(r['mean_latency_ms'])))
        import re
        m=re.fullmatch(rf'{var}_rrf(\d+)_expand(\d+)',best['method'])
        if m:
            cfg[var]={'method':best['method'],'kind':'tuned','rrf_k':int(m[1]),'expand_n':int(m[2]),'dev_ndcg_at_5':best['ndcg_at_5'],'dev_mean_latency_ms':best['mean_latency_ms']}
        else:
            valid=({'bm25','tfidf','lightweight_full','lightweight_rrf'} if var=='lightweight' else {'minilm','pretrained_full','pretrained_rrf'})
            if best['method'] not in valid:raise SystemExit('Unexpected method label; cannot freeze')
            cfg[var]={'method':best['method'],'kind':'baseline','dev_ndcg_at_5':best['ndcg_at_5'],'dev_mean_latency_ms':best['mean_latency_ms']}
    locked={'dev_metrics_sha256':hashlib.sha256(metrics.read_bytes()).hexdigest(),
            'benchmark_freeze_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
            'source':'development set only, provisional until relevance judgments independently verified',
            'selected':cfg}
    if args.out.exists():raise SystemExit(f'Config already frozen: {args.out}; do not tune again on test')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(locked,indent=2),encoding='utf-8')
    print('LOCKED dev-only configs:',cfg)
    print('Now you may run the test script exactly once. Never choose based on test metrics.')
if __name__=='__main__':main()
