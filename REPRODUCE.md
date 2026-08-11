# Reproduce DSCF-Net Results

## 1. Environment

```bash
cd /path/to/DSCF-Net
cd code/DSCF-Net
conda env create -f ../../environment/environment.yml
conda activate dscf-net
```

If CUDA-specific PyTorch wheels are installed manually, use `../../environment/requirements.txt` as the version reference.

## 2. Datasets

Download UCM, AID, and NWPU45 from their original providers. Keep raw class-folder layouts as:

```text
Data/UCM/<class>/<image>
Data/AID/<class>/<image>
Data/NWPU45/<class>/<image>
```

Rebuild the pre-split directories from the released split records:

```bash
cd /path/to/DSCF-Net
python data_availability/prepare_dataset_split.py --raw-root /path/to/Data/UCM --split-dir data_availability/splits/UCM_20 --out-root /path/to/Dataset/UCM_20 --mode symlink
python data_availability/prepare_dataset_split.py --raw-root /path/to/Data/UCM --split-dir data_availability/splits/UCM_50 --out-root /path/to/Dataset/UCM_50 --mode symlink
python data_availability/prepare_dataset_split.py --raw-root /path/to/Data/UCM --split-dir data_availability/splits/UCM_80 --out-root /path/to/Dataset/UCM_80 --mode symlink
python data_availability/prepare_dataset_split.py --raw-root /path/to/Data/AID --split-dir data_availability/splits/AID_20 --out-root /path/to/Dataset/AID_20 --mode symlink
python data_availability/prepare_dataset_split.py --raw-root /path/to/Data/AID --split-dir data_availability/splits/AID_50 --out-root /path/to/Dataset/AID_50 --mode symlink
python data_availability/prepare_dataset_split.py --raw-root /path/to/Data/NWPU45 --split-dir data_availability/splits/NWPU45_10 --out-root /path/to/Dataset/NWPU45_10 --mode symlink
python data_availability/prepare_dataset_split.py --raw-root /path/to/Data/NWPU45 --split-dir data_availability/splits/NWPU45_20 --out-root /path/to/Dataset/NWPU45_20 --mode symlink
```

Use `--mode copy` instead of `--mode symlink` if symbolic links are not available.

## 3. Verification Before Training

```bash
cd /path/to/DSCF-Net
python check_release_integrity.py
cd code/DSCF-Net
export DSCF_DATA_ROOT=/path/to/Dataset
python ../verify_reproducibility.py
python train.py --help
```

## 4. Table 4 Main Experiments

```bash
cd /path/to/DSCF-Net/code/DSCF-Net
python train.py --dataset UCM --train-ratio 0.5 --pretrained-backbone --neck-channels 128 --epochs 80 --batch-size 64 --num-runs 10 --data-root $DSCF_DATA_ROOT --output-dir results/main_training
python train.py --dataset UCM --train-ratio 0.8 --pretrained-backbone --neck-channels 128 --epochs 80 --batch-size 64 --num-runs 10 --data-root $DSCF_DATA_ROOT --output-dir results/main_training
python train.py --dataset AID --train-ratio 0.2 --pretrained-backbone --neck-channels 128 --epochs 100 --batch-size 64 --num-runs 10 --data-root $DSCF_DATA_ROOT --output-dir results/main_training
python train.py --dataset AID --train-ratio 0.5 --pretrained-backbone --neck-channels 128 --epochs 100 --batch-size 64 --num-runs 10 --data-root $DSCF_DATA_ROOT --output-dir results/main_training
python train.py --dataset NWPU45 --train-ratio 0.1 --pretrained-backbone --neck-channels 128 --epochs 80 --batch-size 64 --num-runs 10 --data-root $DSCF_DATA_ROOT --output-dir results/main_training
python train.py --dataset NWPU45 --train-ratio 0.2 --pretrained-backbone --neck-channels 128 --epochs 80 --batch-size 64 --num-runs 10 --data-root $DSCF_DATA_ROOT --output-dir results/main_training
PYTHON=python bash run_paper_analysis.sh tables
```

## 5. Table 5 Channel-Reduced Prototype

```bash
cd /path/to/DSCF-Net/code/DSCF-Net
python run_exp1_ablation.py --variant baseline --dataset UCM --train-ratio 0.2 --epochs 150 --batch-size 64 --num-runs 10 --data-root $DSCF_DATA_ROOT
python run_exp1_ablation.py --variant pdc2k_only --dataset UCM --train-ratio 0.2 --epochs 150 --batch-size 64 --num-runs 10 --data-root $DSCF_DATA_ROOT
python run_exp1_ablation.py --variant acs_only --dataset UCM --train-ratio 0.2 --epochs 150 --batch-size 64 --num-runs 10 --data-root $DSCF_DATA_ROOT
python run_exp1_ablation.py --variant full --dataset UCM --train-ratio 0.2 --epochs 150 --batch-size 64 --num-runs 10 --data-root $DSCF_DATA_ROOT
```

## 6. Tables 6-8 Fixed Structural Analyses

```bash
cd /path/to/DSCF-Net/code/DSCF-Net
python train.py --dataset AID --train-ratio 0.5 --ablation-variant conv_standard --epochs 100 --batch-size 64 --num-runs 5 --data-root $DSCF_DATA_ROOT
python train.py --dataset AID --train-ratio 0.5 --ablation-variant conv_depthwise --epochs 100 --batch-size 64 --num-runs 5 --data-root $DSCF_DATA_ROOT
python train.py --dataset AID --train-ratio 0.5 --ablation-variant conv_dilated --epochs 100 --batch-size 64 --num-runs 5 --data-root $DSCF_DATA_ROOT
python train.py --dataset AID --train-ratio 0.5 --ablation-variant acs_nosccfa --epochs 100 --batch-size 64 --num-runs 5 --data-root $DSCF_DATA_ROOT
python train.py --dataset AID --train-ratio 0.5 --ablation-variant acs_spatial_only --epochs 100 --batch-size 64 --num-runs 5 --data-root $DSCF_DATA_ROOT
python train.py --dataset AID --train-ratio 0.5 --ablation-variant acs_channel_only --epochs 100 --batch-size 64 --num-runs 5 --data-root $DSCF_DATA_ROOT
python train.py --dataset AID --train-ratio 0.5 --ablation-variant acs_full --epochs 100 --batch-size 64 --num-runs 5 --data-root $DSCF_DATA_ROOT
python train.py --dataset AID --train-ratio 0.5 --ablation-variant dil_123 --epochs 100 --batch-size 64 --num-runs 3 --data-root $DSCF_DATA_ROOT
python train.py --dataset AID --train-ratio 0.5 --ablation-variant dil_246 --epochs 100 --batch-size 64 --num-runs 3 --data-root $DSCF_DATA_ROOT
python train.py --dataset AID --train-ratio 0.5 --ablation-variant dil_357 --epochs 100 --batch-size 64 --num-runs 3 --data-root $DSCF_DATA_ROOT
```

## 7. Figures 6-8 Confusion Matrices

To regenerate confusion matrices without retraining, extract the optional checkpoint archive into the release root so that files appear under:

```text
checkpoints/main_training/AID_0.5/best.pth
checkpoints/main_training/NWPU45_0.2/best.pth
checkpoints/main_training/UCM_0.8/best.pth
```

Then copy or link them into `code/DSCF-Net/results/main_training/checkpoints/`, or set `DSCF_CKPT_ROOT` when using `utils/confusion_matrix.py`.

```bash
cd /path/to/DSCF-Net/code/DSCF-Net
export DSCF_DATA_ROOT=/path/to/Dataset
PYTHON=python DSCF_DATA_ROOT=$DSCF_DATA_ROOT bash run_paper_analysis.sh confusion --task main:AID:0.5 --task main:NWPU45:0.2 --task main:UCM:0.8
```

The generated matrix materials used in the paper are in `materials/confusion_matrices_fig6_8`.
