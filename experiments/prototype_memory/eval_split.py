"""
Evaluacion por split (DEV/TEST), con las metricas "por cada 1000" pedidas.

Uso:
    uv run python eval_split.py DEV
    uv run python eval_split.py TEST

Reusa evaluate.py (matching por ventana de 3 frames) restringiendo detail_df
y gt_df al conjunto de trip_id del split pedido (splits.csv), y agrega las
metricas de negocio: por cada 1000 viajes / por cada 1000 cambios reales.
"""
import os
import sys

import pandas as pd

from common import EXP_DIR, DATA_DIR
from evaluate import evaluate, _add_frame_idx
from common import match_events_window

OUT_DIR = os.path.join(EXP_DIR, "splits")
os.makedirs(OUT_DIR, exist_ok=True)


def load_combined(pattern, base_dir=DATA_DIR):
    dfs = []
    for name in ["random_04", "reviewed_07"]:
        p = os.path.join(base_dir, pattern.format(name=name))
        d = pd.read_csv(p)
        d['dataset'] = name
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def restrict(df, trip_ids, col='ASSET_ID'):
    return df[df[col].isin(trip_ids)].reset_index(drop=True)


def business_metrics(detail_df, gt_df, trip_universe, label, window=3):
    """Metricas 'para el jefe': por cada 1000 viajes / por cada 1000 cambios
    reales, sobre el universo de viajes del split (trip_universe), no solo
    los que aparecen en detail_df (un viaje sin ninguna fila de detalle
    tambien cuenta como viaje sin FP)."""
    m, tp, fp = evaluate(detail_df, gt_df, window=window, label=label)

    n_trips = len(trip_universe)
    n_fp = len(fp)
    n_trips_con_fp = fp['trip_id'].nunique() if len(fp) else 0
    n_posible = int(m['posible_cambio_total'])

    n_gt_clean = int(m['gt_clean'])
    n_gt_clean_matched = round(m['recall_sin_suspect_cualquier_alerta'] * n_gt_clean) if n_gt_clean else 0
    n_gt_clean_perdidos = n_gt_clean - n_gt_clean_matched

    out = {
        'label': label,
        'n_viajes': n_trips,
        'n_cambios_reales_gt_clean': n_gt_clean,
        'n_cambios_reales_gt_suspect': int(m['gt_suspect']),
        'fp_estrictos_total': n_fp,
        'fp_estrictos_por_1000_viajes': round(1000 * n_fp / n_trips, 2) if n_trips else float('nan'),
        'viajes_con_1mas_fp_total': n_trips_con_fp,
        'viajes_con_1mas_fp_por_1000_viajes': round(1000 * n_trips_con_fp / n_trips, 2) if n_trips else float('nan'),
        'posible_cambio_revision_total': n_posible,
        'posible_cambio_revision_por_1000_viajes': round(1000 * n_posible / n_trips, 2) if n_trips else float('nan'),
        'cambios_detectados_por_1000_cambios_reales': round(1000 * n_gt_clean_matched / n_gt_clean, 2) if n_gt_clean else float('nan'),
        'cambios_perdidos_por_1000_cambios_reales': round(1000 * n_gt_clean_perdidos / n_gt_clean, 2) if n_gt_clean else float('nan'),
        'recall_sin_suspect': round(m['recall_sin_suspect_cualquier_alerta'], 4),
        'recall_con_suspect': round(m['recall_con_suspect_cualquier_alerta'], 4),
        'cambio_confirmado_total': int(m['cambio_confirmado_total']),
        'cambio_confirmado_correctos': int(m['cambio_confirmado_correctos']),
        'cambio_confirmado_incorrectos_fp': n_fp,
        'precision_estricta': round(m['precision_estricta_cambio_confirmado'], 4) if pd.notna(m['precision_estricta_cambio_confirmado']) else None,
        'delay_medio_frames': round(m['delay_medio_frames_confirmado'], 2) if pd.notna(m['delay_medio_frames_confirmado']) else None,
        'delay_max_frames': m['delay_max_frames_confirmado'],
        'window_frames': window,
    }
    return out, tp, fp


def main():
    split_name = sys.argv[1] if len(sys.argv) > 1 else "DEV"
    assert split_name in ("DEV", "TEST")

    splits = pd.read_csv(os.path.join(OUT_DIR, "splits.csv"))
    trip_ids = set(splits.loc[splits['split'] == split_name, 'trip_id'].tolist())

    gt_all = load_combined("gt_events_{name}.csv")
    gt_all['gt_suspect'] = gt_all['gt_suspect'].astype(bool)
    gt_split = restrict(gt_all, trip_ids, col='trip_id')

    print(f"\n{'=' * 70}\nSPLIT: {split_name}  ({len(trip_ids)} viajes)\n{'=' * 70}")

    for algo_label, pattern, base_dir in [
        ("baseline", "baseline_detail_{name}.csv", DATA_DIR),
        ("memory_v1", "memory_detail_{name}.csv", DATA_DIR),
        ("memory_v2", "memory_detail_v2_{name}.csv", OUT_DIR),
    ]:
        detail_all = load_combined(pattern, base_dir=base_dir)
        detail_split = restrict(detail_all, trip_ids)
        out, tp, fp = business_metrics(detail_split, gt_split, trip_ids, label=f"{algo_label}_{split_name}")
        print(f"\n--- {algo_label} ---")
        for k, v in out.items():
            print(f"  {k}: {v}")

        tp.to_csv(os.path.join(OUT_DIR, f"{algo_label}_tp_{split_name}.csv"), index=False)
        fp.to_csv(os.path.join(OUT_DIR, f"{algo_label}_fp_{split_name}.csv"), index=False)

        summary_path = os.path.join(OUT_DIR, f"metrics_{algo_label}_{split_name}.csv")
        pd.DataFrame([out]).to_csv(summary_path, index=False)


if __name__ == "__main__":
    main()
