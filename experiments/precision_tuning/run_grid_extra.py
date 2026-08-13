"""
Extension puntual del grid (misma metodologia que run_grid.py), explorando
la zona alrededor de candidate_min_support=3-4 con candidate_window=4-5,
que en el primer grid mostro la mejor relacion precision/TP. Se corre
tambien SOLO sobre el subset de ajuste. Los resultados se agregan a
grid_results.csv (concatenados) y se recalcula el top10.
"""
import itertools
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prototype_memory"))
from common import DATA_DIR
from evaluate import evaluate

from precompute import precompute_trips
from run_grid import build_detail_df, TUNING_DIR

GRID_EXTRA = {
    'candidate_window': [4, 5],
    'candidate_min_support': [3, 4],
    'coherence_threshold': [0.25, 0.30, 0.35],
    'min_avg_dist_confirm': [0.85, 0.90],
    'min_phys_confirm': [0.6, 0.7, 0.8],
}


def run_configs(cache, gt_df, configs):
    rows = []
    for cfg in configs:
        detail_df = build_detail_df(cache, cfg)
        m, tp, fp = evaluate(detail_df, gt_df, window=3, label="grid_extra")
        tp_confirm = m['cambio_confirmado_correctos']
        fp_confirm = m['cambio_confirmado_incorrectos_fp_estrictos']
        gt_clean = m['gt_clean']
        recall_any = m['recall_sin_suspect_cualquier_alerta']
        recall_confirmado = (tp_confirm / gt_clean) if gt_clean else float('nan')
        misses = gt_clean - round(recall_any * gt_clean) if gt_clean else 0
        row = dict(cfg)
        row.update({
            'TP_confirmado': tp_confirm,
            'FP_confirmado': fp_confirm,
            'precision_confirmado': m['precision_estricta_cambio_confirmado'],
            'recall_confirmado': recall_confirmado,
            'recall_ANY': recall_any,
            'misses': misses,
            'cambio_confirmado_total': m['cambio_confirmado_total'],
            'posible_cambio_total': m['posible_cambio_total'],
        })
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    subset = pd.read_csv(os.path.join(TUNING_DIR, "subset_trips.csv"))
    trip_ids_by_dataset = {
        name: subset.loc[subset['dataset'] == name, 'trip_id'].tolist()
        for name in subset['dataset'].unique()
    }
    print("Precomputando (subset, reutilizado)...")
    cache = precompute_trips(trip_ids_by_dataset)

    gt_parts = []
    for name in subset['dataset'].unique():
        gt = pd.read_csv(os.path.join(DATA_DIR, f"gt_events_{name}.csv"))
        gt_parts.append(gt[gt['trip_id'].isin(set(cache.keys()))])
    gt_df = pd.concat(gt_parts, ignore_index=True)
    gt_df['gt_suspect'] = gt_df['gt_suspect'].astype(bool)

    keys = list(GRID_EXTRA.keys())
    configs = []
    for combo in itertools.product(*GRID_EXTRA.values()):
        cfg = dict(zip(keys, combo))
        if cfg['candidate_min_support'] > cfg['candidate_window']:
            continue
        configs.append(cfg)
    print(f"Configuraciones extra validas: {len(configs)}")

    results_extra = run_configs(cache, gt_df, configs)

    prev = pd.read_csv(os.path.join(TUNING_DIR, "grid_results.csv"))
    combined = pd.concat([prev, results_extra], ignore_index=True)
    combined = combined.drop_duplicates(subset=list(GRID_EXTRA.keys()))
    combined.to_csv(os.path.join(TUNING_DIR, "grid_results.csv"), index=False)
    print(f"Grid combinado total: {len(combined)} configuraciones")

    valid = combined[combined['recall_ANY'] >= 0.98].copy()
    valid = valid.sort_values(
        by=['precision_confirmado', 'FP_confirmado', 'TP_confirmado'],
        ascending=[False, True, False],
    )
    top10 = valid.head(10)
    top10.to_csv(os.path.join(TUNING_DIR, "grid_top10.csv"), index=False)

    cols = ['candidate_window', 'candidate_min_support', 'coherence_threshold',
            'min_avg_dist_confirm', 'min_phys_confirm', 'precision_confirmado',
            'TP_confirmado', 'FP_confirmado', 'recall_ANY', 'recall_confirmado',
            'cambio_confirmado_total']
    print("\nTop 10 combinado (subset de ajuste):")
    print(top10[cols].to_string(index=False))


if __name__ == "__main__":
    main()
