"""Describe chunking/overlap/subset effects without altering existing indexes.

The output measures preprocessing complexity, NOT retrieval relevance. To compare
quality across chunk settings, each candidate would require separately rebuilt
indexes, independently graded evidence, and DEV-only selection.
"""
import argparse,csv,sys,time
from pathlib import Path
from collections import defaultdict
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'phase2'/'shared'))
from corpus import load_phase1_corpus,DEFAULT_DATASET
from chunker import chunk_document

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,default=DEFAULT_DATASET)
    ap.add_argument('--out',type=Path,default=HERE/'preprocessing_sensitivity.csv')
    ap.add_argument('--max-docs',type=int,default=1000)
    args=ap.parse_args();docs=load_phase1_corpus(args.data,limit=args.max_docs)
    sizes=[150,250,350];overlaps=[1,2,3];subsets=[min(250,len(docs)),min(500,len(docs)),len(docs)]
    out=[]
    for n in sorted(set(subsets)):
        for size in sizes:
            for overlap in overlaps:
                tm=time.perf_counter();chunks=[]
                for d in docs[:n]:
                    chunks.extend(chunk_document(d['text'],d['id'],d['citation'],d['url'],d['jurisdiction'],chunk_size=size,overlap_sentences=overlap))
                out.append({'documents':n,'chunk_size_words':size,'overlap_sentences':overlap,
                   'chunks_indexed':len(chunks),'mean_words_per_chunk':round(sum(len(c.text.split()) for c in chunks)/max(1,len(chunks)),1),
                   'chunking_seconds':round(time.perf_counter()-tm,2),'retrieval_quality_evaluated':'NO'})
                print(out[-1])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=out[0]);w.writeheader();w.writerows(out)
    print('Saved descriptive analysis to',args.out)
if __name__=='__main__':main()
