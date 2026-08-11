set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON_BIN="${PYTHON:-python}"

MODE=${1:-tables}
shift || true

mkdir -p results/paper_analysis/logs

if [ "$MODE" = "tables" ]; then
  "$PYTHON_BIN" paper_analysis/collect_results.py "$@" | tee results/paper_analysis/logs/tables.log
elif [ "$MODE" = "profile" ]; then
  "$PYTHON_BIN" paper_analysis/profile_latency.py "$@" | tee results/paper_analysis/logs/profile.log
elif [ "$MODE" = "confusion" ]; then
  "$PYTHON_BIN" paper_analysis/confusion_class_analysis.py "$@" | tee results/paper_analysis/logs/confusion.log
elif [ "$MODE" = "report" ]; then
  "$PYTHON_BIN" paper_analysis/make_paper_report.py "$@" | tee results/paper_analysis/logs/report.log
elif [ "$MODE" = "all" ]; then
  "$PYTHON_BIN" paper_analysis/collect_results.py | tee results/paper_analysis/logs/tables.log
  "$PYTHON_BIN" paper_analysis/profile_latency.py --dataset AID --batch-size 1 | tee results/paper_analysis/logs/profile.log
  "$PYTHON_BIN" paper_analysis/confusion_class_analysis.py | tee results/paper_analysis/logs/confusion.log
  "$PYTHON_BIN" paper_analysis/make_paper_report.py | tee results/paper_analysis/logs/report.log
else
  printf 'unknown mode: %s
' "$MODE" >&2
  exit 2
fi
