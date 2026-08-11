# Paper Result Tables

## Main Results

| Dataset | Train ratio | OA (%) | Params (M) | FLOPs (G) | Source |
| --- | --- | --- | --- | --- | --- |
| UCM | 0.5 | 98.47 +/- 0.13 | 7.38 | 2.0617 | results/main_training/results_UCM_0.5.json |
| UCM | 0.8 | 99.45 +/- 0.20 | 7.38 | 2.0617 | results/main_training/results_UCM_0.8.json |
| AID | 0.2 | 95.39 +/- 0.12 | 7.38 | 2.0617 | results/main_training/results_AID_0.2.json |
| AID | 0.5 | 97.59 +/- 0.08 | 7.38 | 2.0617 | results/main_training/results_AID_0.5.json |
| NWPU45 | 0.1 | 92.76 +/- 0.10 | 7.39 | 2.0617 | results/main_training/results_NWPU45_0.1.json |
| NWPU45 | 0.2 | 94.91 +/- 0.09 | 7.39 | 2.0617 | results/main_training/results_NWPU45_0.2.json |

## AID 50% Ablation

| Variant | Meaning | OA (%) | Params (M) | FLOPs (G) | Source |
| --- | --- | --- | --- | --- | --- |
| pretrained_no_neck | EfficientNet-B0 pretrained + GAP + FC | 97.60 +/- 0.07 | 3.59 | 0.5138 | results/ablation/runs/exp1_core_modules/pretrained_no_neck/results_AID_0.5.json |
| pretrained_baseline | EfficientNet-B0 pretrained + plain FPN/PAN + DSCFHead | 97.25 +/- 0.01 | 6.16 | 1.4525 | results/ablation/runs/exp1_core_modules/pretrained_baseline/results_AID_0.5.json |
| pretrained_pdc2k_only | EfficientNet-B0 pretrained + PDC-2k only | 97.39 +/- 0.23 | 6.07 | 1.4719 | results/ablation/runs/exp1_core_modules/pretrained_pdc2k_only/results_AID_0.5.json |
| pretrained_acs_only | EfficientNet-B0 pretrained + ACS Fusion only | 97.35 +/- 0.06 | 7.32 | 2.0328 | results/ablation/runs/exp1_core_modules/pretrained_acs_only/results_AID_0.5.json |
| pretrained_full | EfficientNet-B0 pretrained + PDC-2k + ACS Fusion | 97.33 +/- 0.13 | 7.38 | 2.0617 | results/ablation/runs/exp1_core_modules/pretrained_full/results_AID_0.5.json |
