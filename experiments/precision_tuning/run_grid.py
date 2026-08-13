"""
PASO 2 y 3: grid search RAPIDO de umbrales de confirmacion de memory v1
(algo_memory.py), corrido SOLO sobre el subset de ajuste (precision_tuning/
subset_trips.csv, 295 viajes de DEV), para subir la precision de
CAMBIO_CONFIRMADO sin romper recall_ANY.

NO se toca VISUAL_MATCH (umbral de match contra memoria): se deja fijo en
el valor del baseline (0.5) para no tocar la sensibilidad de POSIBLE_CAMBIO.
NO se usa identity_id como feature del algoritmo (solo para el GT/evaluacion).
NO se corre nada sobre TEST.

Guarda:
  - grid_results.csv (todas las configuraciones probadas)
  - grid_top10.csv (las mejores 5-10 segun el criterio pedido)
"""
import itertools
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prototype_memory"))
from common import EXP_DIR, DATA_DIR
from evaluate import evaluate

from precompute import precompute_trips
from algo_v1_param import run_v1_param, DEFAULT_PARAMS

TUNING_DIR = os.path.dirname(os.path.abspath(__file__))

# Rangos razonables alrededor de los valores actuales de algo_memory.py
GRID = {
    'candidate_window': [3, 4],
    'candidate_min_support': [2, 3],
    'coherence_threshold': [0.25, 0.30, 0.35],
    'min_avg_dist_confirm': [0.85, 0.90, 0.95],
    'min_phys_confirm': [0.6, 0.7, 0.8],
}


def build_detail_df(cache, params):
    all_records = []
    for trip_id, frames in cache.items():
        recs = run_v1_param(frames, params, trip_id=trip_id)
        all_records.extend(recs)
    return pd.DataFrame(all_records)


def run_configs(cache, gt_df, configs):
    rows = []
    for cfg in configs:
        detail_df = build_detail_df(cache, cfg)
        m, tp, fp = evaluate(detail_df, gt_df, window=3, label="grid")

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

    print("Precomputando embeddings del subset (una sola vez)...")
    cache = precompute_trips(trip_ids_by_dataset)
    print(f"  {len(cache)} viajes precomputados, "
          f"{sum(len(v) for v in cache.values())} frames totales")

    gt_parts = []
    for name in subset['dataset'].unique():
        gt = pd.read_csv(os.path.join(DATA_DIR, f"gt_events_{name}.csv"))
        gt_parts.append(gt[gt['trip_id'].isin(set(cache.keys()))])
    gt_df = pd.concat(gt_parts, ignore_index=True)
    gt_df['gt_suspect'] = gt_df['gt_suspect'].astype(bool)

    keys = list(GRID.keys())
    configs = []
    for combo in itertools.product(*GRID.values()):
        cfg = dict(zip(keys, combo))
        if cfg['candidate_min_support'] > cfg['candidate_window']:
            continue
        configs.append(cfg)
    print(f"Configuraciones validas en el grid: {len(configs)}")

    results = run_configs(cache, gt_df, configs)
    results_path = os.path.join(TUNING_DIR, "grid_results.csv")
    results.to_csv(results_path, index=False)
    print(f"Guardado grid completo en {results_path}")

    # PASO 3: ranking. Filtro duro: recall_ANY >= 0.98. Entre esos, ordenar
    # por precision_confirmado desc, luego FP_confirmado asc, luego
    # TP_confirmado desc (para no elegir "no confirma nunca").
    valid = results[results['recall_ANY'] >= 0.98].copy()
    print(f"\nConfiguraciones con recall_ANY >= 0.98: {len(valid)} / {len(results)}")

    valid = valid.sort_values(
        by=['precision_confirmado', 'FP_confirmado', 'TP_confirmado'],
        ascending=[False, True, False],
    )
    top10 = valid.head(10)
    top10_path = os.path.join(TUNING_DIR, "grid_top10.csv")
    top10.to_csv(top10_path, index=False)

    print(f"\nTop 10 (subset de ajuste, {len(cache)} viajes):")
    cols = list(GRID.keys()) + ['precision_confirmado', 'TP_confirmado', 'FP_confirmado',
                                 'recall_ANY', 'recall_confirmado', 'cambio_confirmado_total']
    print(top10[cols].to_string(index=False))
    print(f"\nGuardado en {top10_path}")

    # Referencia: v1 default sobre el mismo subset
    default_row = run_configs(cache, gt_df, [DEFAULT_PARAMS])
    print("\nmemory v1 (valores actuales, mismo subset) como referencia:")
    print(default_row[cols].to_string(index=False))
    default_row.to_csv(os.path.join(TUNING_DIR, "v1_default_on_subset.csv"), index=False)


if __name__ == "__main__":
    main()
