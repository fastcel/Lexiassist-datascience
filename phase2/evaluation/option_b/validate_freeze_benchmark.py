"""Validate a source-grounded 100-question benchmark and freeze 40 dev/60 test.

NOT a relevance assessor. Freeze is denied until all questions have human-reviewed
source anchors, reviewers, and REVIEWED status. Never invent missing evidence.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path

HERE=Path(__file__).resolve().parent
DEFAULT=HERE/'benchmark'/'benchmark_100_DRAFT.csv'
REQ=['query_id','query','domain','split','question_status','source_doc_id',
     'source_chunk_index','source_chunk_key','source_citation','source_url',
     'source_jurisdiction','source_excerpt','reviewer','review_date']

def read_csv(p):
    with p.open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write_csv(p,headers,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=headers);w.writeheader();w.writerows(rows)

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def validate(rows):
    problems=[]
    if len(rows)!=100:problems.append(f'Expected exactly 100 rows; got {len(rows)}')
    ids=[r.get('query_id','').strip() for r in rows]
    if len(set(ids))!=len(ids):problems.append('Duplicate query IDs')
    expected={f'Q{i:03d}' for i in range(1,101)}
    if set(ids)!=expected:problems.append('Question IDs must be Q001 through Q100 exactly')
    splits={'dev':0,'test':0}
    for r in rows:
        qid=r.get('query_id','?')
        for k in REQ:
            if not r.get(k,'').strip():problems.append(f'{qid}: missing {k}')
        s=r.get('split','')
        if s not in splits:problems.append(f'{qid}: invalid split {s!r}')
        else:splits[s]+=1
        if qid in {f'Q{i:03d}' for i in range(1,21)} and s!='dev':
            problems.append(f'{qid}: legacy question must be dev')
        if r.get('question_status','').strip().upper()!='VERIFIED':
            problems.append(f'{qid}: source/question not verified by reviewer')
        if not r.get('source_url','').strip().startswith('http'):
            problems.append(f'{qid}: missing source URL')
        if '::chunk_' not in r.get('source_chunk_key',''):
            problems.append(f'{qid}: missing correct chunk key')
    if splits!={'dev':40,'test':60}:problems.append(f'Splits must be dev=40,test=60: {splits}')
    questions=[r.get('query','').strip().casefold() for r in rows]
    if len(set(questions))!=len(questions):problems.append('Questions must not be identical after casing')
    # New anchored sources should not appear in both development and test.
    devdocs={r.get('source_doc_id') for r in rows if r.get('split')=='dev'}-{''}
    testdocs={r.get('source_doc_id') for r in rows if r.get('split')=='test'}-{''}
    clash=devdocs & testdocs
    if clash:problems.append(f'{len(clash)} source document(s) shared across dev and test')
    return problems,splits

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--benchmark',type=Path,default=DEFAULT)
    a.add_argument('--out',type=Path,default=HERE/'frozen')
    a.add_argument('--check-only',action='store_true')
    args=a.parse_args()
    rows=read_csv(args.benchmark)
    if not rows:raise SystemExit('Empty benchmark')
    if any(k not in rows[0] for k in REQ):raise SystemExit('Missing required columns')
    problems,splits=validate(rows)
    print('Benchmark rows:',len(rows),'| dev/test:',splits)
    print('Review statuses:',{s:sum(r.get('question_status')==s for r in rows) for s in sorted(set(r.get('question_status') for r in rows))})
    if problems:
        print('\nCannot freeze: ',len(problems),'check(s) remain. First 15:')
        for p in problems[:15]:print(' -',p)
        print('Edit benchmark_100_DRAFT.csv. Mark VERIFIED only after human source review.')
        raise SystemExit(1)
    if args.check_only:
        print('Passed all checks; --check-only did not write anything.')
        return
    args.out.mkdir(parents=True,exist_ok=True)
    fields=['query_id','query','domain']
    for split in ('dev','test'):
        write_csv(args.out/f'{split}_questions.csv',fields,[{k:r[k] for k in fields} for r in rows if r['split']==split])
    # Copy complete audited source-anchored benchmark alongside the freeze.
    write_csv(args.out/'benchmark_100_VERIFIED.csv',list(rows[0]),rows)
    manifest={'status':'FROZEN_HUMAN_VERIFIED','questions':100,'dev':40,'test':60,
     'source_benchmark_sha256':digest(args.benchmark),'verified_copy_sha256':digest(args.out/'benchmark_100_VERIFIED.csv'),
     'dev_sha256':digest(args.out/'dev_questions.csv'),'test_sha256':digest(args.out/'test_questions.csv')}
    (args.out/'freeze_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('FROZEN. Development set:',args.out/'dev_questions.csv')
    print('The test set is sealed. Do not run test retrieval until configuration is locked.')
if __name__=='__main__':main()
