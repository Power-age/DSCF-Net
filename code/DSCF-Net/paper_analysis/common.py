import csv
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = Path(os.environ.get("DSCF_DATA_ROOT", PROJECT_ROOT / "Dataset"))
PAPER_DIR = Path(os.environ.get("DSCF_PAPER_DIR", PROJECT_ROOT / "paper_outputs"))
PAPER_ANALYSIS_DIR = PROJECT_ROOT / "results" / "paper_analysis"


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ratio_label(ratio):
    value = float(ratio)
    if value.is_integer():
        return str(int(value))
    return str(value)


def ratio_percent(ratio):
    return int(round(float(ratio) * 100))


def result_json_path(source, dataset, ratio, variant=None):
    if source == "main":
        return PROJECT_ROOT / "results" / "main_training" / f"results_{dataset}_{ratio_label(ratio)}.json"
    return PROJECT_ROOT / "results" / "ablation" / "runs" / "exp1_core_modules" / variant / f"results_{dataset}_{ratio_label(ratio)}.json"


def checkpoint_path(source, dataset, ratio, variant=None, allow_partial=False):
    if source == "main":
        return PROJECT_ROOT / "results" / "main_training" / "checkpoints" / f"{dataset}_{ratio_label(ratio)}" / "best.pth"
    ckpt_dir = PROJECT_ROOT / "results" / "ablation" / "runs" / "exp1_core_modules" / variant / "checkpoints" / f"{dataset}_{ratio_label(ratio)}"
    best = ckpt_dir / "best.pth"
    if best.exists():
        return best
    if allow_partial:
        matching_paths = sorted(ckpt_dir.glob("run_*_best.pth"))
        if matching_paths:
            return matching_paths[-1]
    return best


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_csv(path, rows, fieldnames):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, text):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def percent(value):
    return float(value) * 100.0


def format_mean_std(mean_value, std_value):
    return f"{percent(mean_value):.2f} +/- {percent(std_value):.2f}"


def build_model(source, num_classes, variant=None):
    if source == "main":
        from models.dscf_net_pretrained import DSCFNetPretrained
        return DSCFNetPretrained(num_classes=num_classes, neck_channels=128)
    from models.ablation_variants import VARIANT_REGISTRY
    return VARIANT_REGISTRY[variant](num_classes)


def load_checkpoint_model(source, dataset, ratio, num_classes, variant=None, device="cuda", allow_partial=False):
    import torch
    model = build_model(source, num_classes, variant)
    ckpt = torch.load(checkpoint_path(source, dataset, ratio, variant, allow_partial), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model, ckpt
