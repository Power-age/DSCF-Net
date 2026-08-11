#!/usr/bin/env python3
"""
Exp1 ablation: PDC-2k + ACS Fusion interaction (single variant, single GPU).

Usage:
    python run_exp1_ablation.py --variant baseline --gpu 0
    python run_exp1_ablation.py --variant full --gpu 1 --epochs 150 --num-runs 10
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.dscf_net import DSCFBackbone, DSCFNeck, DSCFHead
from data.dataloader import create_dataloaders, set_seed
from utils.model_profile import count_parameters, count_flops


# ---------------------------------------------------------------------------
# Model building blocks (from ablation_study.py)
# ---------------------------------------------------------------------------

class BaselineConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return x + self.conv(x)


class PlainBackbone(nn.Module):
    def __init__(self, embed_dims=(64, 128, 192, 256)):
        super().__init__()
        c1 = embed_dims[0]
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()
        for i in range(len(embed_dims)):
            in_ch = embed_dims[i - 1] if i > 0 else c1
            out_ch = embed_dims[i]
            if i > 0:
                self.transitions.append(nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.SiLU(inplace=True),
                ))
            self.stages.append(BaselineConvBlock(out_ch))

    def forward(self, x):
        x = self.stem(x)
        features = []
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = self.transitions[i - 1](x)
            x = stage(x)
            features.append(x)
        return features


class PlainNeck(nn.Module):
    def __init__(self, in_channels, neck_channels=76):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, neck_channels, 1, bias=False) for c in in_channels
        ])
        # Highest level (i == len-1) gets single input (no concat); lower levels get double
        n_levels = len(in_channels)
        self.fpn_refines = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(neck_channels * 2 if i < n_levels - 1 else neck_channels,
                          neck_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(neck_channels),
                nn.SiLU(inplace=True),
            )
            for i in range(n_levels)
        ])
        self.pan_refines = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(neck_channels * 2 if i < len(in_channels) - 1 else neck_channels,
                          neck_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(neck_channels),
                nn.SiLU(inplace=True),
            )
            for i in range(len(in_channels))
        ])
        self.pan_downsamples = nn.ModuleList([
            nn.Conv2d(neck_channels, neck_channels, 3, stride=2, padding=1, bias=False)
            for _ in range(len(in_channels) - 1)
        ])

    def forward(self, features):
        feats = features[-3:]
        lateral = [conv(f) for conv, f in zip(self.lateral_convs, feats)]
        fpn_feats = []
        prev = None
        for i in range(len(lateral) - 1, -1, -1):
            if prev is None:
                fpn_feats.insert(0, self.fpn_refines[i](lateral[i]))
            else:
                up = nn.functional.interpolate(prev, size=lateral[i].shape[2:],
                                               mode='bilinear', align_corners=False)
                fused = torch.cat([lateral[i], up], dim=1)
                fpn_feats.insert(0, self.fpn_refines[i](fused))
            prev = fpn_feats[0] if i == len(lateral) - 1 else fpn_feats[0]
        pan_feats = []
        prev = None
        for i in range(len(fpn_feats)):
            if prev is None:
                pan_feats.append(fpn_feats[i])
            else:
                down = self.pan_downsamples[i - 1](prev)
                fused = torch.cat([fpn_feats[i], down], dim=1)
                pan_feats.append(self.pan_refines[i - 1](fused))
            prev = pan_feats[-1]
        return pan_feats


class Exp1FullModel(nn.Module):
    def __init__(self, num_classes, embed_dims, neck_channels, use_pdc2k=True, use_acs=True):
        super().__init__()
        if use_pdc2k:
            self.backbone = DSCFBackbone(embed_dims=embed_dims)
        else:
            self.backbone = PlainBackbone(embed_dims=embed_dims)
        neck_in = embed_dims[-3:]
        if use_acs:
            self.neck = DSCFNeck(in_channels=neck_in, neck_channels=neck_channels)
        else:
            self.neck = PlainNeck(in_channels=neck_in, neck_channels=neck_channels)
        self.head = DSCFHead(in_channels=neck_channels, num_classes=num_classes, head_channels=neck_channels)

    def forward(self, x):
        feats = self.backbone(x)
        neck_feats = self.neck(feats)
        return self.head(neck_feats)


# ---------------------------------------------------------------------------
# Training loop (adapted from train.py)
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def run_experiment(args, num_classes, model_builder, out_dir):
    """Run multiple training repeats for one ablation variant."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Profile model
    model = model_builder(num_classes).to(device)
    params_m = count_parameters(model)
    flops_g = count_flops(model)
    del model

    oa_list = []

    for run_id in range(args.num_runs):
        set_seed(args.seed + run_id)

        train_loader, val_loader, _, _ = create_dataloaders(
            args.dataset, args.train_ratio, args.batch_size,
            args.num_workers, args.data_root, args.seed + run_id,
        )

        model = model_builder(num_classes).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.lr * 1e-4,
        )
        scaler = GradScaler() if args.amp else None

        best_oa = 0.0
        best_epoch = 0

        for epoch in range(1, args.epochs + 1):
            if epoch <= args.warmup_epochs:
                warmup_lr = args.lr * epoch / args.warmup_epochs
                for pg in optimizer.param_groups:
                    pg["lr"] = warmup_lr

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device, args.amp,
            )
            val_loss, val_oa = validate(model, val_loader, criterion, device)

            if epoch > args.warmup_epochs:
                cosine_scheduler.step()

            if val_oa > best_oa:
                best_oa = val_oa
                best_epoch = epoch
                ckpt_dir = out_dir / "checkpoints" / f"{args.dataset}_{args.train_ratio}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_oa": val_oa,
                    "params_M": params_m,
                    "flops_G": flops_g,
                    "run": run_id + 1,
                }, ckpt_dir / f"run_{run_id+1}_best.pth")

            if epoch % 15 == 0 or epoch == 1:
                print(f"  [GPU {args.gpu}] {args.variant} Run {run_id+1}/{args.num_runs} "
                      f"Epoch {epoch:3d}/{args.epochs} | "
                      f"Train Acc: {train_acc*100:.2f}% | Val OA: {val_oa*100:.2f}%")

        oa_list.append(best_oa)
        print(f"  [GPU {args.gpu}] {args.variant} Run {run_id+1} best OA: {best_oa*100:.2f}% @ epoch {best_epoch}")

        del model, train_loader, val_loader
        torch.cuda.empty_cache()

    # Select best run checkpoint
    _select_best_checkpoint(out_dir, args.dataset, args.train_ratio)

    # Statistics
    oa_tensor = torch.tensor(oa_list)
    mean_oa = oa_tensor.mean().item()
    std_oa = oa_tensor.std().item()

    print(f"\n  [GPU {args.gpu}] {args.variant} Final: OA = {mean_oa*100:.2f} ± {std_oa*100:.2f} %")
    print(f"  Params: {params_m:.2f}M | FLOPs: {flops_g:.4f}G")

    # Save results
    results = {
        "variant": args.variant,
        "dataset": args.dataset,
        "train_ratio": args.train_ratio,
        "oa_mean": mean_oa,
        "oa_std": std_oa,
        "oa_list": oa_list,
        "params_M": params_m,
        "flops_G": flops_g,
    }
    result_path = out_dir / f"results_{args.dataset}_{args.train_ratio}.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"  Results saved to: {result_path}")


def _select_best_checkpoint(out_dir, dataset, train_ratio):
    """Keep only the single best checkpoint across all runs."""
    ckpt_dir = out_dir / "checkpoints" / f"{dataset}_{train_ratio}"
    if not ckpt_dir.is_dir():
        return
    best_oa = -1.0
    best_path = None
    for ckpt_path in sorted(ckpt_dir.glob("run_*_best.pth")):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        oa = ckpt.get("val_oa", 0)
        if oa > best_oa:
            best_oa = oa
            best_path = ckpt_path
    if best_path:
        final_path = ckpt_dir / "best.pth"
        best_path.rename(final_path)
        for ckpt_path in ckpt_dir.glob("run_*_best.pth"):
            ckpt_path.unlink()
        print(f"  Best checkpoint: {final_path} (OA={best_oa*100:.2f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Exp1 ablation - single variant")
    parser.add_argument("--variant", type=str, required=True,
                        choices=["baseline", "pdc2k_only", "acs_only", "full"])
    parser.add_argument("--gpu", type=int, default=0)
    # Dataset
    parser.add_argument("--dataset", type=str, default="UCM")
    parser.add_argument("--train-ratio", type=float, default=0.2)
    parser.add_argument("--data-root", type=str, default=os.environ.get("DSCF_DATA_ROOT", "Dataset"))
    # Training
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--output-dir", type=str,
                        default="results/ablation/table5_ucm20_channel_reduced_prototype/exp1_core_modules")
    return parser.parse_args()


def main():
    args = parse_args()

    embed_dims = (28, 60, 100, 140)
    neck_channels = 60
    num_pdckk = 2

    out_dir = Path(args.output_dir) / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    # Map variant to model builder
    variant_configs = {
        "baseline":    (False, False),
        "pdc2k_only":  (True,  False),
        "acs_only":    (False, True),
        "full":        (True,  True),
    }
    use_pdc2k, use_acs = variant_configs[args.variant]

    def model_builder(num_classes):
        return Exp1FullModel(
            num_classes=num_classes,
            embed_dims=embed_dims,
            neck_channels=neck_channels,
            use_pdc2k=use_pdc2k,
            use_acs=use_acs,
        )

    print(f"\n{'='*60}")
    print(f"Exp1 Ablation: {args.variant} (GPU {args.gpu})")
    print(f"{'='*60}")
    print(f"  Dataset: {args.dataset} ({int(args.train_ratio*100)}%)")
    print(f"  Epochs: {args.epochs} | Runs: {args.num_runs}")
    print(f"  LR: {args.lr} | Warmup: {args.warmup_epochs} epochs")
    print(f"  embed_dims: {embed_dims} | neck: {neck_channels}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}\n")

    # Get num_classes
    train_loader, _, num_classes, _ = create_dataloaders(
        args.dataset, args.train_ratio, args.batch_size,
        args.num_workers, args.data_root, args.seed,
    )
    del train_loader

    run_experiment(args, num_classes, model_builder, out_dir)


if __name__ == "__main__":
    main()
