from pathlib import Path
import csv
root = Path(__file__).resolve().parent
out = root / 'paper_tables'
out.mkdir(exist_ok=True)
with (root / 'materials' / 'main_results' / 'table4_main_results_index.csv').open(newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
fields = ['dataset', 'train_ratio', 'oa_percent', 'params_M', 'flops_G', 'release_file']
with (out / 'table4_main_results.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for row in rows:
        w.writerow({k: row.get(k, '') for k in fields})
with (root / 'materials' / 'tables_5_8' / 'tables_5_8_source_mapping.csv').open(newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
for table in sorted(set(row['table'] for row in rows)):
    table_rows = [row for row in rows if row['table'] == table]
    fields = ['variant', 'oa_percent', 'params_M', 'flops_G', 'release_file', 'purpose']
    with (out / f'{table}.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in table_rows:
            w.writerow({k: row.get(k, '') for k in fields})
print(out)
