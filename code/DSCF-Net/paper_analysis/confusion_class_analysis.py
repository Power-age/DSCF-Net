import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import DATA_ROOT, PAPER_DIR, PAPER_ANALYSIS_DIR, checkpoint_path, ensure_dir, load_checkpoint_model, write_json, write_text
from data.dataloader import create_dataloaders

DEFAULT_TASKS = [
    "ablation:pretrained_no_neck:AID:0.5",
    "ablation:pretrained_full:AID:0.5",
    "main:UCM:0.8",
    "main:AID:0.5",
    "main:NWPU45:0.2",
]


def parse_task(text):
    parts = text.split(":")
    if parts[0] == "main" and len(parts) == 3:
        return {"source": "main", "variant": None, "dataset": parts[1], "ratio": float(parts[2])}
    if parts[0] == "ablation" and len(parts) == 4:
        return {"source": "ablation", "variant": parts[1], "dataset": parts[2], "ratio": float(parts[3])}
    raise ValueError(f"Invalid task: {text}")


def task_id(task):
    variant = task["variant"] or "main"
    ratio = str(task["ratio"]).replace(".", "p")
    return f"{task['source']}_{variant}_{task['dataset']}_{ratio}"


def compute(model, loader, num_classes, device):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            labels = labels.numpy()
            for true_label, pred_label in zip(labels, preds):
                cm[int(true_label), int(pred_label)] += 1
    return cm


def class_rows(cm, class_names):
    totals = cm.sum(axis=1)
    values = np.divide(np.diag(cm), totals, out=np.zeros(cm.shape[0], dtype=np.float64), where=totals != 0)
    rows = []
    for idx, name in enumerate(class_names):
        rows.append({
            "class_index": idx,
            "class_name": name,
            "correct": int(cm[idx, idx]),
            "total": int(totals[idx]),
            "accuracy": float(values[idx]),
            "accuracy_percent": float(values[idx] * 100.0),
        })
    return rows


def top_confusions(cm, class_names, limit):
    rows = []
    for i in range(cm.shape[0]):
        total = cm[i].sum()
        if total == 0:
            continue
        for j in range(cm.shape[1]):
            if i == j or cm[i, j] == 0:
                continue
            rows.append({
                "true_class": class_names[i],
                "pred_class": class_names[j],
                "count": int(cm[i, j]),
                "rate": float(cm[i, j] / total),
                "rate_percent": float(cm[i, j] * 100.0 / total),
            })
    rows.sort(key=lambda item: item["rate"], reverse=True)
    return rows[:limit]


def save_csv(path, rows, fieldnames):
    ensure_dir(Path(path).parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(cm, class_names, path, title):
    cm_norm = cm.astype(np.float64)
    totals = cm_norm.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    cm_norm = cm_norm / totals
    n = len(class_names)
    if n <= 21:
        figsize = (16, 12)
        fontsize = 8
    elif n <= 30:
        figsize = (20, 15)
        fontsize = 7
    else:
        figsize = (24, 18)
        fontsize = 5
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=fontsize)
    ax.set_yticklabels(class_names, fontsize=fontsize)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    ensure_dir(Path(path).parent)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def markdown(summary, class_data, confusions):
    lines = [
        f"# {summary['task_id']}",
        "",
        f"Dataset: {summary['dataset']}",
        f"Train ratio: {summary['train_ratio']}",
        f"Source: {summary['source']}",
        f"Variant: {summary['variant']}",
        f"OA: {summary['oa_percent']:.2f}%",
        f"Mean class accuracy: {summary['mean_class_accuracy_percent']:.2f}%",
        "",
        "## Lowest Class Accuracy",
        "",
        "| Class | Accuracy (%) | Correct | Total |",
        "|---|---:|---:|---:|",
    ]
    for row in sorted(class_data, key=lambda item: item["accuracy"])[:10]:
        lines.append(f"| {row['class_name']} | {row['accuracy_percent']:.2f} | {row['correct']} | {row['total']} |")
    lines.extend(["", "## Top Confusions", "", "| True | Predicted | Rate (%) | Count |", "|---|---|---:|---:|"])
    for row in confusions:
        lines.append(f"| {row['true_class']} | {row['pred_class']} | {row['rate_percent']:.2f} | {row['count']} |")
    return "\n".join(lines) + "\n"


def run_task(task, args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    _, loader, num_classes, class_names = create_dataloaders(task["dataset"], task["ratio"], args.batch_size, args.num_workers, args.data_root, args.seed)
    ckpt = checkpoint_path(task["source"], task["dataset"], task["ratio"], task["variant"], args.allow_partial)
    if not ckpt.exists():
        return {"task": task, "status": "missing_checkpoint", "checkpoint": str(ckpt)}
    model, ckpt_data = load_checkpoint_model(task["source"], task["dataset"], task["ratio"], num_classes, task["variant"], device, args.allow_partial)
    cm = compute(model, loader, num_classes, device)
    ident = task_id(task)
    out_dir = ensure_dir(Path(args.output_dir) / ident)
    np.save(out_dir / "confusion_matrix.npy", cm)
    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    class_data = class_rows(cm, class_names)
    confusions = top_confusions(cm, class_names, args.top_k)
    save_csv(out_dir / "per_class_accuracy.csv", class_data, ["class_index", "class_name", "correct", "total", "accuracy", "accuracy_percent"])
    save_csv(out_dir / "top_confusions.csv", confusions, ["true_class", "pred_class", "count", "rate", "rate_percent"])
    oa = float(np.diag(cm).sum() / cm.sum())
    mean_class_acc = float(np.mean([row["accuracy"] for row in class_data]))
    summary = {
        "task_id": ident,
        "source": task["source"],
        "variant": task["variant"] or "main",
        "dataset": task["dataset"],
        "train_ratio": task["ratio"],
        "checkpoint": str(ckpt),
        "checkpoint_val_oa": float(ckpt_data.get("val_oa", 0.0)),
        "oa": oa,
        "oa_percent": oa * 100.0,
        "mean_class_accuracy": mean_class_acc,
        "mean_class_accuracy_percent": mean_class_acc * 100.0,
        "num_samples": int(cm.sum()),
        "num_classes": num_classes,
    }
    write_json(out_dir / "summary.json", {"summary": summary, "class_accuracy": class_data, "top_confusions": confusions})
    plot(cm, class_names, out_dir / "confusion_matrix.png", ident)
    report = markdown(summary, class_data, confusions)
    write_text(out_dir / "analysis.md", report)
    return {"task": task, "status": "done", "summary": summary, "report": str(out_dir / "analysis.md")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", action="append")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output-dir", default=str(PAPER_ANALYSIS_DIR / "confusion_class"))
    parser.add_argument("--paper-dir", default=str(PAPER_DIR))
    args = parser.parse_args()

    tasks = [parse_task(item) for item in (args.task or DEFAULT_TASKS)]
    results = [run_task(task, args) for task in tasks]
    output_dir = ensure_dir(args.output_dir)
    write_json(output_dir / "run_summary.json", results)
    done_reports = [item["report"] for item in results if item.get("status") == "done"]
    text = "# Paper Confusion and Class Analysis\n\n" + "\n".join(f"- {path}" for path in done_reports) + "\n"
    write_text(output_dir / "paper_confusion_class_index.md", text)
    paper_dir = ensure_dir(args.paper_dir)
    write_text(paper_dir / "paper_confusion_class_index.md", text)
    print(output_dir / "paper_confusion_class_index.md")


if __name__ == "__main__":
    main()
