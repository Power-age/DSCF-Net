# DSCF-Net

This repository contains the public final reproducibility package for the paper "DSCF-Net: A pretrained dual-scale cross-fusion network with attentional spatial-channel enhancement for remote sensing scene classification".

## Scope

This release includes the paper-final code, dataset split records, environment files, result materials, and scripts required to reproduce or interpret the reported findings.

Original UCM, AID, and NWPU-RESISC45 images are not redistributed. Users should obtain the original datasets from their providers and use the released split files to rebuild the exact train/test partitions used in the paper.

## Contents

- `code/DSCF-Net`: model implementation, training entry points, evaluation utilities, and analysis scripts.
- `data_availability`: train/test split records and per-class split statistics.
- `environment`: dependency versions and environment files.
- `materials`: result JSON/CSV files, confusion matrices, latency records, and table mappings.
- `paper_tables`: regenerated CSV tables corresponding to Tables 4-8.
- `REPRODUCE.md`: end-to-end reproduction workflow.

## Quick Verification

```bash
python check_release_integrity.py
cd code/DSCF-Net
python ../verify_reproducibility.py
python train.py --help
```

The integrity script checks required files, Python syntax, split records, material paths, checksums, and common release hygiene issues. The reproducibility script checks model parameter counts, FLOPs, split records, and required result materials.

## Dataset Preparation

After downloading the original datasets, run split preparation from the release root. For example:

```bash
cd /path/to/DSCF-Net
python data_availability/prepare_dataset_split.py --raw-root /path/to/Data/AID --split-dir data_availability/splits/AID_50 --out-root /path/to/Dataset/AID_50 --mode symlink
```

Set the dataset root before training:

```bash
export DSCF_DATA_ROOT=/path/to/Dataset
```

## Reproduction

See `REPRODUCE.md` for commands corresponding to:

- Table 4 main experiments.
- Table 5 channel-reduced prototype analysis.
- Tables 6-8 fixed structural analyses.
- Figures 6-8 confusion matrices.

## Checkpoints

Main-experiment checkpoints are provided separately as an optional archive for evaluating the final DSCF-Net and regenerating the confusion matrices. They are not stored in this GitHub repository.

## License

Code is released under the MIT License. Reproducibility materials in this repository are released under CC BY 4.0 unless otherwise stated. Original datasets remain subject to their providers' terms.
