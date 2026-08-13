"""
PASO 4: correr SOLO las 5-10 configuraciones ganadoras del grid (elegidas
sobre el subset de ajuste) sobre TODO DEV (1416 viajes, NO el subset).
Se agrega tambien memory v1 (valores actuales) como referencia.

NO se toca TEST en este script.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prototype_memory"))
from common import EXP_DIR, DATA_DIR
from evaluate import evaluate

from precompute import precompute_trips
from algo_v1_param import run_v1_param, DEFAULT_PARAMS

TUNING_DIR = os.path.dirname(os.path.abspath(__file__))

# Shortlist (top de precision sobre el subset, con piso de TP para no elegir
# configuraciones que casi nunca confirman). Ver grid_results.csv / analisis.
SHORTLIST = {
    'A_w4_s2_c035_d085_p08': dict(candidate_window=4, candidate_min_support=2, coherence_threshold=0.35, min_avg_dist_confirm=0.85, min_phys_confirm=0.8),
    'B_w3_s2_c035_d085_p08': dict(candidate_window=3, candidate_min_support=2, coherence_threshold=0.35, min_avg_dist_confirm=0.85, min_phys_confirm=0.8),
    'C_w4_s2_c035_d085_p07': dict(candidate_window=4, candidate_min_support=2, coherence_threshold=0.35, min_avg_dist_confirm=0.85, min_phys_confirm=0.7),
    'D_w3_s2_c035_d085_p07': dict(candidate_window=3, candidate_min_support=2, coherence_threshold=0.35, min_avg_dist_confirm=0.85, min_phys_confirm=0.7),
    'E_w4_s3_c035_d085_p06': dict(candidate_window=4, candidate_min_support=3, coherence_threshold=0.35, min_avg_dist_confirm=0.85, min_phys_confirm=0.6),
    'F_w5_s3_c035_d085_p06': dict(candidate_window=5, candidate_min_support=3, coherence_threshold=0.35, min_avg_dist_confirm=0.85, min_phys_confirm=0.6),
    'G_w4_s2_c030_d085_p08': dict(candidate_window=4, candidate_min_support=2, coherence_threshold=0.30, min_avg_dist_confirm=0.85, min_phys_confirm=0.8),
    'H_w3_s2_c030_d085_p08': dict(candidate_window=3, candidate_min_support=2, coherence_threshold=0.30, min_avg_dist_confirm=0.85, min_phys_confirm=0.8),
    'I_w4_s2_c030_d085_p07': dict(candidate_window=4, candidate_min_support=2, coherence_threshold=0.30, min_avg_dist_confirm=0.85, min_phys_confirm=0.7),
    'J_w4_s2_c035_d085_p06': dict(candidate_window=4, candidate_min_support=2, coherence_threshold=0.35, min_avg_dist_confirm=0.85, min_phys_confirm=0.6),
    'V1_DEFAULT': dict(DEFAULT_PARAMS),
}


def build_detail_df(cache, params):
    all_records = []
    for trip_id, frames in cache.items():
        recs = run_v1_param(frames, params, trip_id=trip_id)
        all_records.extend(recs)
    return pd.DataFrame(all_records)


def main():
    splits = pd.read_csv(os.path.join(EXP_DIR, "splits", "splits.csv"))
    dev = splits[splits["split"] == "DEV"].copy()
    trip_ids_by_dataset = {
        name: dev.loc[dev['dataset'] == name, 'trip_id'].tolist()
        for name in dev['dataset'].unique()
    }
    n_dev = len(dev)
    print(f"Precomputando TODO DEV ({n_dev} viajes) -- puede tardar varios minutos...")
    cache = precompute_trips(trip_ids_by_dataset)
    print(f"  {len(cache)} viajes precomputados, {sum(len(v) for v in cache.values())} frames totales")

    gt_parts = []
    for name in dev['dataset'].unique():
        gt = pd.read_csv(os.path.join(DATA_DIR, f"gt_events_{name}.csv"))
        gt_parts.append(gt[gt['trip_id'].isin(set(cache.keys()))])
    gt_df = pd.concat(gt_parts, ignore_index=True)
    gt_df['gt_suspect'] = gt_df['gt_suspect'].astype(bool)

    rows = []
    for config_name, cfg in SHORTLIST.items():
        print(f"  corriendo {config_name} ...")
        detail_df = build_detail_df(cache, cfg)
        m, tp_df, fp_df = evaluate(detail_df, gt_df, window=3, label=config_name)

        tp_confirm = m['cambio_confirmado_correctos']
        fp_confirm = m['cambio_confirmado_incorrectos_fp_estrictos']
        gt_clean = m['gt_clean']
        recall_any = m['recall_sin_suspect_cualquier_alerta']
        recall_confirmado = (tp_confirm / gt_clean) if gt_clean else float('nan')
        misses = gt_clean - round(recall_any * gt_clean) if gt_clean else 0

        row = dict(cfg)
        row['config_name'] = config_name
        row.update({
            'TP_confirmado': tp_confirm,
            'FP_confirmado': fp_confirm,
            'precision_confirmado': m['precision_estricta_cambio_confirmado'],
            'recall_confirmado': recall_confirmado,
            'recall_ANY': recall_any,
            'misses': misses,
            'cambio_confirmado_total': m['cambio_confirmado_total'],
            'posible_cambio_total': m['posible_cambio_total'],
            'FP_por_1000_viajes': 1000 * fp_confirm / n_dev,
            'viajes_con_1mas_fp_por_1000': 1000 * fp_df['trip_id'].nunique() / n_dev if len(fp_df) else 0.0,
            'misses_por_1000_cambios_reales': 1000 * misses / gt_clean if gt_clean else 0.0,
        })
        rows.append(row)

    results = pd.DataFrame(rows)
    out_path = os.path.join(TUNING_DIR, "full_dev_results.csv")
    results.to_csv(out_path, index=False)

    cols = ['config_name', 'candidate_window', 'candidate_min_support', 'coherence_threshold',
            'min_avg_dist_confirm', 'min_phys_confirm', 'precision_confirmado', 'TP_confirmado',
            'FP_confirmado', 'recall_ANY', 'recall_confirmado', 'FP_por_1000_viajes']
    results_sorted = results.sort_values(
        by=['precision_confirmado', 'FP_confirmado', 'TP_confirmado'],
        ascending=[False, True, False],
    )
    print(f"\nResultados sobre DEV completo ({n_dev} viajes):")
    print(results_sorted[cols].to_string(index=False))
    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()
