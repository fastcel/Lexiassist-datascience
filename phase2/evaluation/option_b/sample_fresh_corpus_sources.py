"""Optional, corpus-wide evidence suggestions to replace retrieval-pool-biased drafts.

Selects fresh documents from the SAME first 1,000 Phase 1 CSV rows used by
both saved indexes, using one sentence-aware passage per document.
Outputs NO generated questions and NO relevance labels. Reviewers write questions.
"""
from __future__ import annotations
import argparse,csv,random,re,sys
from pathlib import Path
from collections import defaultdict
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'phase2'/'shared'))
from corpus import load_phase1_corpus,DEFAULT_DATASET
from chunker import chunk_document

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,default=DEFAULT_DATASET)
    ap.add_argument('--out',type=Path,default=HERE/'benchmark'/'fresh_corpus_80_SOURCE_ONLY.csv')
    ap.add_argument('--count',type=int,default=80)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    if args.out.exists():raise SystemExit(f'Output exists: {args.out}; use another --out')
    docs=load_phase1_corpus(args.data,limit=1000)
    if len(docs)!=1000:raise SystemExit(f'Expected the indexed 1,000 documents, got {len(docs)}')
    # Avoid reusing doc IDs already present in the draft if possible.
    used=set()
    f=HERE/'benchmark'/'benchmark_100_DRAFT.csv'
    if f.exists():
        with f.open(encoding='utf-8-sig',newline='') as stream:
            used={r['source_doc_id'] for r in csv.DictReader(stream)}-{''}
    rng=random.Random(args.seed)
    groups=defaultdict(list)
    for d in docs:
        if d['id'] in used:continue
        if not d['url'].startswith('http'):continue
        groups[d['jurisdiction'] or 'unspecified'].append(d)
    for g in groups.values():rng.shuffle(g)
    picked=[]
    while len(picked)<args.count and any(groups.values()):
        for jurisdiction in sorted(groups):
            if groups[jurisdiction] and len(picked)<args.count:picked.append(groups[jurisdiction].pop())
    if len(picked)!=args.count:raise SystemExit(f'Only {len(picked)} viable unseen documents')
    rows=[]
    priority=re.compile(r'\b(means|must|shall|is defined|is entitled|may apply|the court held|the court concluded|the purpose|the object|the duty|the requirement)\b',re.I)
    for i,d in enumerate(picked,21):
        chunks=chunk_document(d['text'],d['id'],d['citation'],d['url'],d['jurisdiction'],chunk_size=250,overlap_sentences=2)
        if not chunks:continue
        candidates=[c for c in chunks[:min(35,len(chunks))] if priority.search(c.text)]
        chosen=rng.choice(candidates or chunks[:min(20,len(chunks))])
        text=' '.join(chosen.text.split())
        m=priority.search(text);pos=m.start() if m else 0
        begin=max(0,pos-250);excerpt=text[begin:begin+900]
        rows.append({'candidate_id':f'C{i:03d}', 'source_doc_id':d['id'],'source_chunk_index':chosen.chunk_index,
          'source_chunk_key':f"{d['id']}::chunk_{chosen.chunk_index}",'source_citation':d['citation'],'source_url':d['url'],
          'source_jurisdiction':d['jurisdiction'],'source_excerpt':excerpt,'draft_question':'',
          'review_status':'SOURCE_ONLY_NEEDS_QUESTION','notes':'Draft an answerable natural-language question based on this exact excerpt; check original source.'})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Saved',len(rows),'fresh source candidates:',args.out)
    print('These are NOT questions, source verifications, or relevance judgments.')
if __name__=='__main__':main()
