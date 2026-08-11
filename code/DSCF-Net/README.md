# DSCF-Net

This directory contains the source code for DSCF-Net. The release root contains dataset split records, result materials, environment files, and reproducibility instructions.

## Quick Checks

```bash
python train.py --help
python -m compileall -q data models utils paper_analysis train.py run_exp1_ablation.py
python ../verify_reproducibility.py
```

## Data Root

Set `DSCF_DATA_ROOT` or pass `--data-root` to training and analysis commands.

```bash
export DSCF_DATA_ROOT=/path/to/Dataset
```
