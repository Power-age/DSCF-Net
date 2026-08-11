# Dataset Availability Records

This directory contains split records for the public UCM, AID, and NWPU45 datasets used in the paper. Original images are not included.

`dataset_split_summary.csv` gives the sample counts for each pre-split setting. `splits/<setting>/train.tsv` and `splits/<setting>/test.tsv` list the exact class-relative image names used by the experiments.

AID records contain 9,927 readable images in this experimental environment. This matches the sample-count statement in the paper and should be kept as the reproducibility scope for AID results.

To rebuild a pre-split directory from an already downloaded raw dataset, use `prepare_dataset_split.py` with a raw dataset root whose class folders match the file names in the split TSV files.
