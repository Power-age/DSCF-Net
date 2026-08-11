import argparse
from pathlib import Path

from common import PAPER_DIR, PAPER_ANALYSIS_DIR, ensure_dir, format_mean_std, load_json, result_json_path, write_csv, write_json, write_text

MAIN_TASKS = [
    ("UCM", 0.5),
    ("UCM", 0.8),
    ("AID", 0.2),
    ("AID", 0.5),
    ("NWPU45", 0.1),
    ("NWPU45", 0.2),
]

ABLATION_VARIANTS = [
    ("pretrained_no_neck", "EfficientNet-B0 pretrained + GAP + FC"),
    ("pretrained_baseline", "EfficientNet-B0 pretrained + plain FPN/PAN + DSCFHead"),
    ("pretrained_pdc2k_only", "EfficientNet-B0 pretrained + PDC-2k only"),
    ("pretrained_acs_only", "EfficientNet-B0 pretrained + ACS Fusion only"),
    ("pretrained_full", "EfficientNet-B0 pretrained + PDC-2k + ACS Fusion"),
]


def read_result(source, dataset, ratio, variant=None, label=None):
    path = result_json_path(source, dataset, ratio, variant)
    if not path.exists():
        return None
    data = load_json(path)
    return {
        "source": source,
        "dataset": dataset,
        "train_ratio": ratio,
        "variant": variant or "DSCFNetPretrained",
        "label": label or variant or "DSCFNetPretrained",
        "oa_mean": data.get("oa_mean"),
        "oa_std": data.get("oa_std"),
        "oa_percent": data.get("oa_mean", 0.0) * 100.0,
        "std_percent": data.get("oa_std", 0.0) * 100.0,
        "oa_text": format_mean_std(data.get("oa_mean", 0.0), data.get("oa_std", 0.0)),
        "params_M": data.get("params_M"),
        "flops_G": data.get("flops_G"),
        "target_met": data.get("target_met"),
        "total_time_min": data.get("total_time_min"),
        "path": str(path.relative_to(Path.cwd())),
    }


def markdown_table(rows, columns):
    lines = []
    lines.append("| " + " | ".join(title for title, _ in columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        cells = []
        for _, key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                if "percent" in key or key.startswith("std"):
                    value = f"{value:.2f}"
                elif key in {"params_M", "flops_G", "total_time_min"}:
                    value = f"{value:.4f}" if key == "flops_G" else f"{value:.2f}"
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(PAPER_ANALYSIS_DIR / "tables"))
    parser.add_argument("--paper-dir", default=str(PAPER_DIR))
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    paper_dir = Path(args.paper_dir)

    main_rows = []
    for dataset, ratio in MAIN_TASKS:
        item = read_result("main", dataset, ratio)
        if item:
            main_rows.append(item)

    ablation_rows = []
    for variant, label in ABLATION_VARIANTS:
        item = read_result("ablation", "AID", 0.5, variant, label)
        if item:
            ablation_rows.append(item)

    write_json(output_dir / "main_results.json", main_rows)
    write_json(output_dir / "aid50_ablation_results.json", ablation_rows)
    write_csv(output_dir / "main_results.csv", main_rows, list(main_rows[0].keys()) if main_rows else [])
    write_csv(output_dir / "aid50_ablation_results.csv", ablation_rows, list(ablation_rows[0].keys()) if ablation_rows else [])

    text = []
    text.append("# Paper Result Tables\n")
    text.append("## Main Results\n")
    text.append(markdown_table(main_rows, [("Dataset", "dataset"), ("Train ratio", "train_ratio"), ("OA (%)", "oa_text"), ("Params (M)", "params_M"), ("FLOPs (G)", "flops_G"), ("Source", "path")]))
    text.append("\n## AID 50% Ablation\n")
    text.append(markdown_table(ablation_rows, [("Variant", "variant"), ("Meaning", "label"), ("OA (%)", "oa_text"), ("Params (M)", "params_M"), ("FLOPs (G)", "flops_G"), ("Source", "path")]))
    report = "\n".join(text) + "\n"

    write_text(output_dir / "paper_result_tables.md", report)
    if paper_dir.exists():
        write_text(paper_dir / "paper_result_tables.md", report)

    print(output_dir / "paper_result_tables.md")


if __name__ == "__main__":
    main()
