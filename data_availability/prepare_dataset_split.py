import argparse
from pathlib import Path
import csv
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--raw-root', required=True)
parser.add_argument('--split-dir', required=True)
parser.add_argument('--out-root', required=True)
parser.add_argument('--mode', choices=['symlink', 'copy'], default='symlink')
args = parser.parse_args()
raw_root = Path(args.raw_root)
split_dir = Path(args.split_dir)
out_root = Path(args.out_root)
for subset in ['train', 'test']:
    tsv = split_dir / f'{subset}.tsv'
    with tsv.open(newline='') as f:
        reader = csv.DictReader(f, delimiter='	')
        for row in reader:
            src = raw_root / row['class_relative_path']
            dst = out_root / subset / row['class_relative_path']
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                continue
            if args.mode == 'copy':
                shutil.copy2(src, dst)
            else:
                os.symlink(src, dst)
