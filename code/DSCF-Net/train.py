#!/usr/bin/env python3
"""
Training script for DSCF-Net on UCM / AID / NWPU45 datasets.

Usage:
    python train.py --dataset UCM --train-ratio 0.5
    python train.py --dataset AID --train-ratio 0.2 --lr 0.01 --epochs 100
    python train.py --dataset NWPU45 --train-ratio 0.1 --batch-size 32
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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.dscf_net import DSCFNet
from data.dataloader import create_dataloaders, set_seed
from utils.model_profile import count_parameters, count_flops


def parse_args():
    parser = argparse.ArgumentParser(description="Train DSCF-Net")
    # Dataset
    parser.add_argument("--dataset", type=str, default="UCM",
                        choices=["UCM", "AID", "NWPU45"])
    parser.add_argument("--train-ratio", type=float, default=0.5,
                        help="Training set proportion")
    parser.add_argument("--data-root", type=str, default=os.environ.get("DSCF_DATA_ROOT", "Dataset"))
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05,
                        help="Initial learning rate")
    parser.add_argument("--lr-multiplier", type=float, default=1.0,
                        help="Per-epoch LR decay multiplier (1.0 = no decay)")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing factor")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Linear warmup epochs")
    parser.add_argument("--use-cosine", action="store_true", default=True,
                        help="Use cosine annealing scheduler")
    # Model
    parser.add_argument("--ablation-variant", type=str, default=None,
                        help="Ablation variant name (uses custom backbone, ignores --pretrained-backbone)")
    parser.add_argument("--pretrained-backbone", action="store_true",
                        help="Use pretrained EfficientNet backbone")
    parser.add_argument("--backbone-name", type=str, default="efficientnet_b0",
                        help="Pretrained backbone model name")
    parser.add_argument("--embed-dims", type=int, nargs=4,
                        default=[48, 96, 160, 224],
                        help="Backbone stage channels (custom backbone only)")
    parser.add_argument("--neck-channels", type=int, default=96)
    parser.add_argument("--num-pdckk", type=int, default=3)
    parser.add_argument("--bottleneck-count", type=int, default=1)
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-runs", type=int, default=10,
                        help="Number of repeated experiments")
    parser.add_argument("--start-run", type=int, default=0,
                        help="Starting run index (for multi-GPU splitting)")
    # System
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use automatic mixed precision")
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


def build_model(args, num_classes):
    if args.ablation_variant:
        from models.ablation_variants import VARIANT_REGISTRY
        if args.ablation_variant not in VARIANT_REGISTRY:
            raise ValueError(f"Unknown ablation variant: {args.ablation_variant}. "
                           f"Available: {list(VARIANT_REGISTRY.keys())}")
        return VARIANT_REGISTRY[args.ablation_variant](num_classes)
    if args.pretrained_backbone:
        from models.dscf_net_pretrained import DSCFNetPretrained
        return DSCFNetPretrained(
            num_classes=num_classes,
            neck_channels=args.neck_channels,
        )
    return DSCFNet(
        num_classes=num_classes,
        embed_dims=tuple(args.embed_dims),
        neck_channels=args.neck_channels,
        num_pdckk=args.num_pdckk,
        bottleneck_count=args.bottleneck_count,
    )


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


def run_single_experiment(args, num_classes, run_id):
    """Run one training experiment and return best OA."""
    set_seed(args.seed + run_id)

    train_loader, val_loader, _, class_names = create_dataloaders(
        args.dataset, args.train_ratio, args.batch_size,
        args.num_workers, args.data_root, args.seed + run_id,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(args, num_classes).to(device)

    # Profile model once per run
    params_m = count_parameters(model)
    flops_g = count_flops(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay,
    )
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.lr * 1e-4)
    scaler = GradScaler() if args.amp else None  # pylint: disable=deprecated

    best_oa = 0.0
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        # Linear warmup
        if epoch <= args.warmup_epochs:
            warmup_lr = args.lr * epoch / args.warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, args.amp,
        )
        val_loss, val_oa = validate(model, val_loader, criterion, device)

        # LR schedule (after warmup)
        if epoch > args.warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] *= args.lr_multiplier
            if args.use_cosine:
                cosine_scheduler.step()

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_oa": val_oa,
        })

        if val_oa > best_oa:
            best_oa = val_oa
            best_epoch = epoch
            # Save best model with metrics
            ckpt_dir = Path(args.output_dir) / "checkpoints" / f"{args.dataset}_{args.train_ratio}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_oa": val_oa,
                "params_M": params_m,
                "flops_G": flops_g,
                "run": run_id + 1,
            }, ckpt_dir / f"run_{run_id+1}_best.pth")

        if epoch % 10 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Run {run_id+1} Epoch {epoch:3d}/{args.epochs} | "
                  f"LR: {lr:.6f} | Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc*100:.2f}% | Val OA: {val_oa*100:.2f}%")

    print(f"  Run {run_id+1} best OA: {best_oa*100:.2f}% @ epoch {best_epoch}")
    return best_oa, history


def select_best_checkpoint(args, out_dir):
    """Select the single best checkpoint across all runs and delete others."""
    ckpt_dir = Path(out_dir) / "checkpoints" / f"{args.dataset}_{args.train_ratio}"
    if not ckpt_dir.is_dir():
        return

    best_oa = -1.0
    best_path = None
    best_info = {}

    for ckpt_path in sorted(ckpt_dir.glob("run_*_best.pth")):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        oa = ckpt.get("val_oa", 0)
        if oa > best_oa:
            best_oa = oa
            best_path = ckpt_path
            best_info = {
                "run": ckpt.get("run", "?"),
                "epoch": ckpt.get("epoch", "?"),
                "val_oa": oa,
                "params_M": ckpt.get("params_M", 0),
                "flops_G": ckpt.get("flops_G", 0),
            }

    if best_path:
        # Rename best to a clean name
        final_path = ckpt_dir / "best.pth"
        best_path.rename(final_path)
        # Delete other checkpoints
        for ckpt_path in ckpt_dir.glob("run_*_best.pth"):
            ckpt_path.unlink()
        print(f"\n  Best model: Run {best_info['run']} @ epoch {best_info['epoch']}")
        print(f"  OA: {best_info['val_oa']*100:.2f}% | Params: {best_info['params_M']:.2f}M | FLOPs: {best_info['flops_G']:.4f}G")
        print(f"  Saved: {final_path}")


def main():
    args = parse_args()
    set_seed(args.seed)

    # Output directory - use ablation path if running an ablation variant
    if args.ablation_variant:
        exp_map = {
            "baseline":"exp1_core_modules", "pdc2k_only":"exp1_core_modules",
            "acs_only":"exp1_core_modules", "full":"exp1_core_modules",
            "pretrained_no_neck":"exp1_core_modules",
            "pretrained_baseline":"exp1_core_modules", "pretrained_pdc2k_only":"exp1_core_modules",
            "pretrained_acs_only":"exp1_core_modules", "pretrained_full":"exp1_core_modules",
            "conv_standard":"exp2_conv_type", "conv_depthwise":"exp2_conv_type",
            "conv_dilated":"exp2_conv_type",
            "acs_nosccfa":"exp3_sccfa_components", "acs_spatial_only":"exp3_sccfa_components",
            "acs_channel_only":"exp3_sccfa_components", "acs_full":"exp3_sccfa_components",
            "dil_123":"exp4_dilations", "dil_246":"exp4_dilations", "dil_357":"exp4_dilations",
        }
        exp_dir = exp_map.get(args.ablation_variant, "unknown")
        out_dir = Path("results/ablation/runs") / exp_dir / args.ablation_variant
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(out_dir)  # Update for downstream use

    # Get num_classes from a temporary dataloader
    train_loader, _, num_classes, class_names = create_dataloaders(
        args.dataset, args.train_ratio, args.batch_size,
        args.num_workers, args.data_root, args.seed,
    )
    # Clean up to avoid CUDA OOM
    del train_loader

    print(f"\n{'='*60}")
    print(f"DSCF-Net Training")
    print(f"{'='*60}")
    print(f"  Dataset     : {args.dataset} ({num_classes} classes)")
    print(f"  Train ratio : {args.train_ratio}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Init LR     : {args.lr}")
    print(f"  LR decay    : {args.lr_multiplier}/epoch")
    print(f"  Optimizer   : SGD (momentum={args.momentum}, wd={args.weight_decay})")
    print(f"  Runs        : {args.num_runs}")
    print(f"{'='*60}\n")

    # Profile model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(args, num_classes).to(device)
    params_m = count_parameters(model)
    flops_g = count_flops(model)
    print(f"  Parameters  : {params_m:.2f} M")
    print(f"  FLOPs       : {flops_g:.4f} G\n")
    del model

    # Run experiments
    oa_list = []
    all_histories = []

    start_time = time.time()
    for run_id in range(args.start_run, args.start_run + args.num_runs):
        best_oa, history = run_single_experiment(args, num_classes, run_id)
        oa_list.append(best_oa)
        all_histories.append(history)

    total_time = time.time() - start_time

    # Compute statistics
    oa_array = torch.tensor(oa_list)
    mean_oa = oa_array.mean().item()
    std_oa = oa_array.std().item()

    # Print results
    print(f"\n{'='*60}")
    print(f"Final Results")
    print(f"{'='*60}")
    print(f"  Dataset      : {args.dataset} ({args.train_ratio*100:.0f}%)")
    print(f"  OA           : {mean_oa*100:.2f} +/- {std_oa*100:.2f} %")
    print(f"  Best OA      : {max(oa_list)*100:.2f} %")
    print(f"  Worst OA     : {min(oa_list)*100:.2f} %")
    print(f"  Parameters   : {params_m:.2f} M")
    print(f"  FLOPs        : {flops_g:.4f} G")
    print(f"  Total time   : {total_time/60:.1f} min")
    print(f"  Per run      : {total_time/args.num_runs/60:.1f} min")
    print(f"{'='*60}")

    # Target check
    targets = {
        ("UCM", 0.5): 0.985, ("UCM", 0.8): 0.991,
        ("AID", 0.2): 0.960, ("AID", 0.5): 0.977,
        ("NWPU45", 0.1): 0.940, ("NWPU45", 0.2): 0.960,
    }
    target = targets.get((args.dataset, args.train_ratio))
    passed = mean_oa >= target if target else None
    param_ok = params_m < 4.0

    print(f"\n  Target OA    : {target*100:.2f}%" if target else "\n  Target: N/A")
    print(f"  OA achieved  : {'YES' if passed else 'NO'}" if target is not None else "")
    print(f"  Params < 4M  : {'YES' if param_ok else 'NO'}")
    print(f"  FLOPs < 0.5G : {'YES' if flops_g < 0.5 else 'NO'}")

    if passed and param_ok:
        print(f"\n  *** All targets met. Ready for ablation study. ***")

    # Save results
    results = {
        "dataset": args.dataset,
        "train_ratio": args.train_ratio,
        "oa_mean": mean_oa,
        "oa_std": std_oa,
        "oa_list": oa_list,
        "params_M": params_m,
        "flops_G": flops_g,
        "target_met": bool(passed) if target is not None else None,
        "total_time_min": total_time / 60,
    }

    result_path = out_dir / f"results_{args.dataset}_{args.train_ratio}.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Results saved to: {result_path}")

    # Write summary markdown
    summary_path = out_dir / "performance_summary.md"
    with open(summary_path, "a") as f:
        f.write(f"## {args.dataset} ({args.train_ratio*100:.0f}%)\n\n")
        f.write(f"| Metric | Value | Target | Status |\n")
        f.write(f"|--------|-------|--------|--------|\n")
        oa_target = f"{target*100:.2f}" if target is not None else "N/A"
        oa_status = f"{'OK' if passed else 'MISS'}" if target is not None else "N/A"
        f.write(f"| OA (%) | {mean_oa*100:.2f} +/- {std_oa*100:.2f} | {oa_target} | {oa_status} |\n")
        f.write(f"| Params (M) | {params_m:.2f} | < 4.0 | {'OK' if param_ok else 'MISS'} |\n")
        f.write(f"| FLOPs (G) | {flops_g:.4f} | < 0.5 | {'OK' if flops_g < 0.5 else 'MISS'} |\n\n")

    # Select and keep only the single best checkpoint across all runs
    select_best_checkpoint(args, out_dir)

    return results


if __name__ == "__main__":
    main()
