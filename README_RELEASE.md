# DSCF-Net Public Final Reproducibility Package

This package contains the paper-final code, dataset split records, result materials, and reproducibility instructions for DSCF-Net.

## Directory Map

- `code/DSCF-Net`: model implementation, training entry points, evaluation utilities, and analysis scripts.
- `data_availability/splits`: train/test file lists for UCM, AID, and NWPU45 settings used in the paper.
- `materials/main_results`: JSON/CSV records corresponding to Table 4.
- `materials/tables_5_8`: result records corresponding to Tables 5-8.
- `materials/confusion_matrices_fig6_8`: outputs corresponding to Figures 6-8.
- `materials/latency`: latency and throughput measurement records.
- `environment`: environment versions and dependency files.

## First Checks

```bash
cd code/DSCF-Net
python ../verify_reproducibility.py
python train.py --help
```

## Dataset Root

Set `DSCF_DATA_ROOT` or pass `--data-root` to training and analysis commands.

```bash
export DSCF_DATA_ROOT=/path/to/Dataset
```

See `REPRODUCE.md` for the full workflow.
