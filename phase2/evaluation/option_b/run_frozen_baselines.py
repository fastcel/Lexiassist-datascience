"""Safely run the original seven methods on the 40-dev or 60-test benchmark.

Tests are not allowed until development settings are locked.
"""
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--split',choices=['dev','test'],required=True)
    ap.add_argument('--frozen',type=Path,default=HERE/'frozen')
    ap.add_argument('--output',type=Path,default=None)
    args=ap.parse_args()
    p=args.frozen/f'{args.split}_questions.csv'
    m=args.frozen/'freeze_manifest.json'
    if not p.is_file() or not m.is_file():raise SystemExit('Missing frozen benchmark')
    mf=json.loads(m.read_text(encoding='utf-8'))
    if digest(p)!=mf[f'{args.split}_sha256']:raise SystemExit('Question list changed after freeze')
    if args.split=='test':
        cfg=args.frozen/'selected_retrieval_config.json'
        if not cfg.is_file():raise SystemExit('Freeze DEV-selected configuration before running TEST')
        locked=json.loads(cfg.read_text(encoding='utf-8'))
        if locked.get('benchmark_freeze_sha256')!=digest(m):raise SystemExit('Config refers to a different benchmark')
    out=args.output or ROOT/f'phase2/evaluation/results/option_b_{args.split}_baselines'
    if out.exists():raise SystemExit(f'Output exists; refusing to overwrite: {out}')
    # Call the original experiment code; it is the canonical seven-method implementation.
    subprocess.run([sys.executable,str(ROOT/'phase2/evaluation/run_experiments.py'),
                    '--questions',str(p),'--output',str(out)],check=True,cwd=ROOT)
    print('Original baselines saved to',out)
    print('Grade all candidate passages independently before scoring.')
if __name__=='__main__':main()
