import argparse
import statistics
import time

import torch

from common import PAPER_DIR, PAPER_ANALYSIS_DIR, build_model, ensure_dir, write_csv, write_json, write_text
from utils.model_profile import count_flops, count_parameters

VARIANTS = [
    ("main", None, "DSCFNetPretrained"),
    ("ablation", "pretrained_no_neck", "pretrained_no_neck"),
    ("ablation", "pretrained_baseline", "pretrained_baseline"),
    ("ablation", "pretrained_pdc2k_only", "pretrained_pdc2k_only"),
    ("ablation", "pretrained_acs_only", "pretrained_acs_only"),
    ("ablation", "pretrained_full", "pretrained_full"),
]

DATASET_CLASSES = {"UCM": 21, "AID": 30, "NWPU45": 45}


def sync(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_latency(model, device, batch_size, warmup, iters):
    x = torch.randn(batch_size, 3, 256, 256, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        sync(device)
        values = []
        for _ in range(iters):
            start = time.perf_counter()
            model(x)
            sync(device)
            values.append((time.perf_counter() - start) * 1000.0)
    return {
        "latency_ms_mean": statistics.mean(values),
        "latency_ms_std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "latency_ms_min": min(values),
        "latency_ms_max": max(values),
        "throughput_img_s": batch_size * 1000.0 / statistics.mean(values),
    }


def markdown(rows):
    lines = [
        "# Paper Profile and Latency",
        "",
        "| Model | Params (M) | FLOPs (G) | Latency mean (ms) | Latency std (ms) | Throughput (img/s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['name']} | {row['params_M']:.4f} | {row['flops_G']:.4f} | {row['latency_ms_mean']:.3f} | {row['latency_ms_std']:.3f} | {row['throughput_img_s']:.2f} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="AID", choices=sorted(DATASET_CLASSES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output-dir", default=str(PAPER_ANALYSIS_DIR / "profile_latency"))
    parser.add_argument("--paper-dir", default=str(PAPER_DIR))
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = ensure_dir(args.output_dir)
    rows = []

    for source, variant, name in VARIANTS:
        model = build_model(source, DATASET_CLASSES[args.dataset], variant).to(device)
        params_m = count_parameters(model)
        flops_g = count_flops(model)
        metrics = measure_latency(model, device, args.batch_size, args.warmup, args.iters)
        row = {
            "name": name,
            "source": source,
            "variant": variant or "",
            "dataset": args.dataset,
            "batch_size": args.batch_size,
            "device": str(device),
            "params_M": params_m,
            "flops_G": flops_g,
            **metrics,
        }
        rows.append(row)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_json(output_dir / f"profile_latency_{args.dataset}_b{args.batch_size}.json", rows)
    write_csv(output_dir / f"profile_latency_{args.dataset}_b{args.batch_size}.csv", rows, list(rows[0].keys()))
    report = markdown(rows)
    write_text(output_dir / f"profile_latency_{args.dataset}_b{args.batch_size}.md", report)
    paper_dir = ensure_dir(args.paper_dir)
    write_text(paper_dir / f"paper_profile_latency_{args.dataset}_b{args.batch_size}.md", report)
    print(output_dir / f"profile_latency_{args.dataset}_b{args.batch_size}.md")


if __name__ == "__main__":
    main()
