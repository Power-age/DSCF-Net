from pathlib import Path
import csv
import sys
root = Path(__file__).resolve().parents[1]
code = root / 'code' / 'DSCF-Net'
sys.path.insert(0, str(code))
from models.dscf_net_pretrained import DSCFNetPretrained
from models.ablation_variants import VARIANT_REGISTRY
from run_exp1_ablation import Exp1FullModel
from utils.model_profile import count_parameters, count_flops
checks = [
    ('main_AID', lambda: DSCFNetPretrained(30, neck_channels=128), 7.380423, 2.061661056),
    ('table5_baseline', lambda: Exp1FullModel(21, (28, 60, 100, 140), 60, False, False), 1.423154, 0.485940856),
    ('table5_pdc2k_only', lambda: Exp1FullModel(21, (28, 60, 100, 140), 60, True, False), 0.971174, 0.354542968),
    ('table5_acs_only', lambda: Exp1FullModel(21, (28, 60, 100, 140), 60, False, True), 1.690334, 0.634514296),
    ('table5_full', lambda: Exp1FullModel(21, (28, 60, 100, 140), 60, True, True), 1.238354, 0.503116408),
    ('table6_conv_standard', lambda: VARIANT_REGISTRY['conv_standard'](30), 3.782173, 1.492028128),
    ('table6_conv_depthwise', lambda: VARIANT_REGISTRY['conv_depthwise'](30), 3.095413, 1.23803824),
    ('table6_conv_dilated', lambda: VARIANT_REGISTRY['conv_dilated'](30), 3.089029, 1.246596832),
    ('table7_acs_nosccfa', lambda: VARIANT_REGISTRY['acs_nosccfa'](30), 2.712901, 1.077575392),
    ('table7_acs_spatial_only', lambda: VARIANT_REGISTRY['acs_spatial_only'](30), 2.629957, 1.04041648),
    ('table7_acs_channel_only', lambda: VARIANT_REGISTRY['acs_channel_only'](30), 2.922565, 1.171762912),
    ('table7_acs_full', lambda: VARIANT_REGISTRY['acs_full'](30), 3.089029, 1.246596832),
    ('table8_dil_123', lambda: VARIANT_REGISTRY['dil_123'](30), 3.042373, 1.222930144),
    ('table8_dil_246', lambda: VARIANT_REGISTRY['dil_246'](30), 3.042373, 1.222930144),
    ('table8_dil_357', lambda: VARIANT_REGISTRY['dil_357'](30), 3.042373, 1.222930144),
]
ok = True
for name, build, exp_p, exp_f in checks:
    model = build()
    p = count_parameters(model)
    f = count_flops(model)
    status = abs(p - exp_p) < 1e-5 and abs(f - exp_f) < 1e-5
    ok = ok and status
    print(f'{name}: {"OK" if status else "MISMATCH"} params={p:.6f} flops={f:.9f}')
with (root / 'data_availability' / 'dataset_split_summary.csv').open(newline='') as fh:
    rows = list(csv.DictReader(fh))
print(f'split_settings: {len(rows)}')
print(f'total_split_images: {sum(int(r["total_images"]) for r in rows)}')
required = [
    root / 'materials' / 'main_results' / 'table4_main_results_index.csv',
    root / 'materials' / 'tables_5_8' / 'tables_5_8_source_mapping.csv',
    root / 'data_availability' / 'all_split_files.tsv',
    root / 'environment' / 'env_versions.txt',
]
for path in required:
    exists = path.exists()
    ok = ok and exists
    print(f'{path.relative_to(root)}: {"OK" if exists else "MISSING"}')
raise SystemExit(0 if ok else 1)
