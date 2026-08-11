import argparse
from pathlib import Path

from common import PAPER_DIR, PAPER_ANALYSIS_DIR, ensure_dir, load_json, write_text


def read_optional(path):
    path = Path(path)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-analysis-dir", default=str(PAPER_ANALYSIS_DIR))
    parser.add_argument("--paper-dir", default=str(PAPER_DIR))
    args = parser.parse_args()

    paper_analysis_dir = Path(args.paper_analysis_dir)
    paper_dir = ensure_dir(args.paper_dir)
    sections = ["# Paper Analysis Report", ""]

    table_report = paper_analysis_dir / "tables" / "paper_result_tables.md"
    profile_reports = sorted((paper_analysis_dir / "profile_latency").glob("*.md")) if (paper_analysis_dir / "profile_latency").exists() else []
    confusion_index = paper_analysis_dir / "confusion_class" / "paper_confusion_class_index.md"

    if table_report.exists():
        sections.append(read_optional(table_report))
    else:
        sections.append("Result tables: pending.\n")

    if profile_reports:
        for path in profile_reports:
            sections.append(read_optional(path))
    else:
        sections.append("Profile and latency: pending.\n")

    if confusion_index.exists():
        sections.append(read_optional(confusion_index))
        summary_path = paper_analysis_dir / "confusion_class" / "run_summary.json"
        if summary_path.exists():
            data = load_json(summary_path)
            sections.append("## Confusion Summary\n")
            for item in data:
                if item.get("status") == "done":
                    summary = item["summary"]
                    gap = abs(summary.get("checkpoint_val_oa", 0.0) - summary.get("oa", 0.0)) * 100.0
                    flag = "OK" if gap <= 0.05 else "CHECK"
                    sections.append(f"- {summary['task_id']}: OA {summary['oa_percent']:.2f}%, mean class accuracy {summary['mean_class_accuracy_percent']:.2f}%, checkpoint gap {gap:.2f} pp [{flag}]")
                else:
                    sections.append(f"- {item['task']}: {item['status']}")
            sections.append("")
            risky = []
            for item in data:
                if item.get("status") == "done":
                    summary = item["summary"]
                    gap = abs(summary.get("checkpoint_val_oa", 0.0) - summary.get("oa", 0.0)) * 100.0
                    if gap > 0.05:
                        risky.append((summary["task_id"], gap))
            if risky:
                sections.append("## Consistency Warning\n")
                sections.append("The following confusion-matrix evaluations differ from checkpoint val_oa by more than 0.05 percentage points and should not be used as final paper evidence until the checkpoint/code/data mapping is rechecked.\n")
                for name, gap in risky:
                    sections.append(f"- {name}: {gap:.2f} pp")
                sections.append("")
    else:
        sections.append("Confusion and class analysis: pending.\n")

    report = "\n".join(sections).strip() + "\n"
    write_text(paper_analysis_dir / "paper_analysis_report.md", report)
    write_text(paper_dir / "paper_analysis_report.md", report)
    print(paper_dir / "paper_analysis_report.md")


if __name__ == "__main__":
    main()
