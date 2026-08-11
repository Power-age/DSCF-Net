from pathlib import Path
import compileall
import csv
import hashlib
import sys

ROOT = Path(__file__).resolve().parent

REQUIRED_FILES = [
    "README.md",
    "REPRODUCE.md",
    "LICENSE",
    "CITATION.cff",
    "MATERIAL_TERMS.md",
    "environment/requirements.txt",
    "environment/environment.yml",
    "code/DSCF-Net/data/__init__.py",
    "code/DSCF-Net/data/dataloader.py",
    "code/DSCF-Net/train.py",
    "code/DSCF-Net/run_exp1_ablation.py",
    "code/DSCF-Net/models/dscf_net.py",
    "code/DSCF-Net/models/dscf_net_pretrained.py",
    "code/DSCF-Net/models/ablation_variants.py",
    "code/DSCF-Net/utils/model_profile.py",
    "code/DSCF-Net/utils/confusion_matrix.py",
    "code/verify_reproducibility.py",
    "data_availability/dataset_split_summary.csv",
    "data_availability/all_split_files.tsv",
    "materials/main_results/table4_main_results_index.csv",
    "materials/tables_5_8/tables_5_8_source_mapping.csv",
    "materials/confusion_matrices_fig6_8/confusion_matrix_AID_50.png",
    "materials/confusion_matrices_fig6_8/confusion_matrix_NWPU45_20.png",
    "materials/confusion_matrices_fig6_8/confusion_matrix_UCM_80.png",
]

FORBIDDEN_PATTERNS = [
    "91D1",
    "connect.nmb2",
    "seetacloud",
    "/root/",
    "root@",
    "gitee",
    "remote_sensing_scene_classification",
    "stage12",
    "stage3",
    "stage4",
    "no-residual",
    "no_residual",
    "probe",
    "source project",
    "Matlab",
    "\"source_mat\""
]

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".json",
    ".csv",
    ".txt",
    ".tsv",
    ".yml",
    ".yaml",
    ".cff",
    ".sh",
    ".gitignore",
    ".gitattributes",
    "license",
}


def fail(message):
    print("FAIL " + message)
    raise SystemExit(1)


def ok(message):
    print("OK " + message)


def rel_path(path):
    return path.as_posix()


def check_required_files():
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    if missing:
        fail("missing required files: " + ", ".join(missing))
    ok("required files")


def check_python_syntax():
    targets = [ROOT / "code" / "DSCF-Net", ROOT / "data_availability"]
    for target in targets:
        if not compileall.compile_dir(str(target), quiet=1):
            fail("python syntax: " + rel_path(target.relative_to(ROOT)))
    ok("python syntax")


def check_release_paths():
    csv_paths = [
        ROOT / "materials" / "main_results" / "table4_main_results_index.csv",
        ROOT / "materials" / "tables_5_8" / "tables_5_8_source_mapping.csv",
    ]
    missing = []
    for csv_path in csv_paths:
        with csv_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                release_file = row.get("release_file")
                if release_file and not (ROOT / release_file).is_file():
                    missing.append(release_file)
    if missing:
        fail("missing release files: " + ", ".join(missing))
    ok("release file paths")


def check_split_summary():
    summary_path = ROOT / "data_availability" / "dataset_split_summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    total = sum(int(row["total_images"]) for row in rows)
    if len(rows) != 7 or total != 89154:
        fail(f"split summary rows={len(rows)} total={total}")
    ok("split summary")


def check_forbidden_text():
    hits = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.name == "check_release_integrity.py":
            continue
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        suffix = path.suffix.lower() if path.suffix else path.name.lower()
        if suffix not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        lowered = text.lower()
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.lower() in lowered:
                hits.append(f"{rel_path(path.relative_to(ROOT))}: {pattern}")
    if hits:
        fail("forbidden text: " + "; ".join(hits[:20]))
    ok("forbidden text scan")


def check_tracked_like_artifacts():
    bad_suffixes = {".pyc", ".pth", ".pt", ".ckpt"}
    hits = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.suffix.lower() in bad_suffixes:
            hits.append(rel_path(path.relative_to(ROOT)))
    if hits:
        fail("unexpected artifacts: " + ", ".join(hits[:20]))
    ok("artifact scan")


def check_sha256sums():
    sums_path = ROOT / "sha256sums.txt"
    if not sums_path.is_file():
        fail("missing sha256sums.txt")
    with sums_path.open(encoding="utf-8") as fh:
        entries = [line.strip().split("  ", 1) for line in fh if line.strip()]
    for expected, raw_path in entries:
        rel = raw_path[2:] if raw_path.startswith("./") else raw_path
        path = ROOT / rel
        if not path.is_file():
            fail("sha256 missing file: " + rel)
        data = path.read_bytes()
        suffix = path.suffix.lower() if path.suffix else path.name.lower()
        if suffix in TEXT_SUFFIXES:
            data = data.replace(b"\r\n", b"\n")
        actual = hashlib.sha256(data).hexdigest()
        if actual.lower() != expected.lower():
            fail("sha256 mismatch: " + rel)
    ok("sha256sums")


def main():
    check_required_files()
    check_python_syntax()
    check_release_paths()
    check_split_summary()
    check_forbidden_text()
    check_tracked_like_artifacts()
    check_sha256sums()
    print("OK public release integrity")


if __name__ == "__main__":
    main()
