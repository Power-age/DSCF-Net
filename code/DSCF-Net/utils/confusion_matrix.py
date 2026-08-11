"""
Confusion matrix generation for DSCF-Net (paper Section 4.3.4).

Generates row-normalized heatmaps (PNG) for:
    - AID 50% (Fig.11)
    - NWPU45 20% (Fig.12)
    - UCM 80% (Fig.13)

Usage:
    python utils/confusion_matrix.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataloader import create_dataloaders


@torch.no_grad()
def compute_confusion_matrix(model, dataloader, num_classes, device="cuda"):
    """Compute confusion matrix on a dataset."""
    model.eval()
    model = model.to(device)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            cm[t, p] += 1

    return cm


def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    """Plot and save row-normalized confusion matrix heatmap.

    Adaptive layout: small matrices use larger cells, large matrices compact.
    """
    num_classes = len(class_names)

    # Row-normalize
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    # Font sizes matched to cell size so text fills each square
    if num_classes <= 21:
        figsize, font_size, dpi = (20, 12), 11, 200
    elif num_classes <= 30:
        figsize, font_size, dpi = (26, 16), 9.5, 200
    else:
        figsize, font_size, dpi = (34, 22), 8, 200

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.65)
    cbar.ax.set_ylabel("Proportion", rotation=-90, va="bottom", fontsize=font_size, fontweight="bold")

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=font_size, fontweight="bold")
    ax.set_yticklabels(class_names, fontsize=font_size, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=font_size + 1, labelpad=2, fontweight="bold")
    ax.set_ylabel("True", fontsize=font_size + 1, labelpad=2, fontweight="bold")
    ax.tick_params(axis="both", which="major", pad=1)
    ax.set_title(title, fontsize=font_size + 2, fontweight="bold")

    # Cell annotations - same size as axis labels, fills each square
    threshold = cm_norm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            v = cm_norm[i, j]
            if v > 0.003:
                color = "white" if v > threshold else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=font_size, color=color, fontweight="bold")

    plt.tight_layout(pad=1.0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def per_class_accuracy(cm, class_names):
    """Compute and print per-class accuracy."""
    row_sums = cm.sum(axis=1)
    accuracies = np.divide(
        np.diag(cm), row_sums,
        out=np.zeros_like(np.diag(cm).astype(np.float64)),
        where=row_sums != 0,
    )

    print(f"\n  Per-Class Accuracy (sorted):")
    sorted_idx = np.argsort(accuracies)
    for i in sorted_idx:
        status = "!!" if accuracies[i] < 0.7 else ("!" if accuracies[i] < 0.85 else "  ")
        print(f"    {status} {class_names[i]:<30s}: {accuracies[i]*100:5.1f}%")

    mean_acc = accuracies.mean()
    print(f"  Mean per-class accuracy: {mean_acc*100:.2f}%")
    return mean_acc


def generate_from_checkpoint(ckpt_path, data_root, dataset_name, train_ratio, out_dir, device="cuda"):
    """Load best model, run inference on test set, plot and save confusion matrix.

    Args:
        ckpt_path: path to best.pth checkpoint.
        data_root: parent directory of pre-split datasets.
        dataset_name: "UCM" | "AID" | "NWPU45".
        train_ratio: training proportion (e.g. 0.5).
        out_dir: output directory for PNG.
        device: compute device.
    """
    print(f"\n{'='*60}")
    print(f"Confusion Matrix: {dataset_name} ({int(train_ratio*100)}%)")
    print(f"{'='*60}")

    # Build test dataloader
    _, test_loader, num_classes, class_names = create_dataloaders(
        dataset_name, train_ratio, batch_size=128,
        num_workers=4, data_root=data_root, seed=42,
    )

    # Build model from checkpoint
    from models.dscf_net_pretrained import DSCFNetPretrained
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = DSCFNetPretrained(num_classes=num_classes, neck_channels=128)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    oa = ckpt["val_oa"] * 100

    # Compute confusion matrix
    cm = compute_confusion_matrix(model, test_loader, num_classes, device)

    # Plot
    title = f"Confusion Matrix - {dataset_name} ({int(train_ratio*100)}% Training)"
    save_path = os.path.join(out_dir, f"confusion_matrix_{dataset_name}_{int(train_ratio*100)}.png")
    plot_confusion_matrix(cm, class_names, save_path, title=title)

    # Per-class report
    per_class_accuracy(cm, class_names)

    # Overall OA from test set
    test_oa = np.diag(cm).sum() / cm.sum() * 100
    print(f"  Checkpoint OA: {oa:.2f}% | Test-set OA: {test_oa:.2f}%")

    return cm, test_oa


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = os.environ.get("DSCF_DATA_ROOT", "Dataset")
    ckpt_root = os.environ.get("DSCF_CKPT_ROOT", "results/main_training/checkpoints")
    out_dir = os.environ.get("DSCF_CONFUSION_OUT", "results/confusion_matrices")

    # Paper Fig.11: AID 50%
    # Paper Fig.12: NWPU45 20%
    # Paper Fig.13: UCM 80%
    tasks = [
        ("UCM", 0.5),
        ("UCM", 0.8),
        ("AID", 0.2),
        ("AID", 0.5),
        ("NWPU45", 0.1),
        ("NWPU45", 0.2),
    ]

    for dataset_name, train_ratio in tasks:
        ratio_str = str(train_ratio).replace(".0", "")
        ckpt_path = os.path.join(ckpt_root, f"{dataset_name}_{ratio_str}", "best.pth")
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
            continue

        print(f"\n  [{dataset_name} {int(train_ratio*100)}%] Generating...")
        generate_from_checkpoint(ckpt_path, data_root, dataset_name, train_ratio, out_dir, device)

    print(f"\n{'='*60}")
    print(f"All confusion matrices saved to: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
