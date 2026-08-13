"""
Corre la variante candidate_fix (algo_candidate_fix.py) sobre TODO DEV
(1416 viajes) con los MISMOS umbrales que la config congelada "3 de 4"
(E_w4_s3_c035_d085_p06, ver winner_config.json) y la compara contra esa
config congelada corrida con la logica ORIGINAL (algo_v1_param.py), sin
tocarla. Objetivo: ver si corregir la logica de ventana/estado del
candidato (sin ajustar ningun umbral todavia) reduce los FP.

NO se toca TEST. NO se ajusta ningun umbral aqui.
"""
import json
import os
import sys

import pandas as pd

CANDIDATE_FIX_DIR = os.path.dirname(os.path.abspath(__file__))
TUNING_DIR = os.path.dirname(CANDIDATE_FIX_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(TUNING_DIR))

sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "prototype_memory"))
from common import EXP_DIR, DATA_DIR  # noqa: E402
from evaluate import evaluate  # noqa: E402

sys.path.insert(0, TUNING_DIR)
from precompute import precompute_trips  # noqa: E402
from algo_v1_param import run_v1_param  # noqa: E402

sys.path.insert(0, CANDIDATE_FIX_DIR)
from algo_candidate_fix import run_candidate_fix, DEFAULT_PARAMS  # noqa: E402

WINNER_CONFIG_PATH = os.path.join(TUNING_DIR, "winner_config.json")


def load_frozen_params():
    with open(WINNER_CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    params = {
        "candidate_window": cfg["candidate_window"],
        "candidate_min_support": cfg["candidate_min_support"],
        "coherence_threshold": cfg["coherence_threshold"],
        "min_avg_dist_confirm": cfg["min_avg_dist_confirm"],
        "min_phys_confirm": cfg["min_phys_confirm"],
        "memory_size": cfg["memory_size"],
    }
    return params, cfg["config_name"]


def build_detail_df(cache, run_fn, params):
    all_records = []
    for trip_id, frames in cache.items():
        recs = run_fn(frames, params, trip_id=trip_id)
        all_records.extend(recs)
    return pd.DataFrame(all_records)


def metrics_row(config_name, params, m, n_dev, fp_df):
    tp_confirm = m["cambio_confirmado_correctos"]
    fp_confirm = m["cambio_confirmado_incorrectos_fp_estrictos"]
    row = dict(params)
    row["config_name"] = config_name
    row.update({
        "TP_confirmado": tp_confirm,
        "FP_confirmado": fp_confirm,
        "precision_confirmado": m["precision_estricta_cambio_confirmado"],
        "recall_ANY": m["recall_sin_suspect_cualquier_alerta"],
        "cambio_confirmado_total": m["cambio_confirmado_total"],
        "posible_cambio_total": m["posible_cambio_total"],
        "FP_por_1000_viajes": 1000 * fp_confirm / n_dev,
        "posible_cambio_por_1000_viajes": 1000 * m["posible_cambio_total"] / n_dev,
    })
    return row


def main():
    frozen_params, frozen_name = load_frozen_params()
    print(f"Config congelada: {frozen_name} -> {frozen_params}")
    print(f"Config candidate_fix (mismos umbrales): {DEFAULT_PARAMS}")

    splits = pd.read_csv(os.path.join(EXP_DIR, "splits", "splits.csv"))
    dev = splits[splits["split"] == "DEV"].copy()
    trip_ids_by_dataset = {
        name: dev.loc[dev["dataset"] == name, "trip_id"].tolist()
        for name in dev["dataset"].unique()
    }
    n_dev = len(dev)
    print(f"Precomputando TODO DEV ({n_dev} viajes) -- puede tardar varios minutos...")
    cache = precompute_trips(trip_ids_by_dataset)
    print(f"  {len(cache)} viajes precomputados, {sum(len(v) for v in cache.values())} frames totales")

    gt_parts = []
    for name in dev["dataset"].unique():
        gt = pd.read_csv(os.path.join(DATA_DIR, f"gt_events_{name}.csv"))
        gt_parts.append(gt[gt["trip_id"].isin(set(cache.keys()))])
    gt_df = pd.concat(gt_parts, ignore_index=True)
    gt_df["gt_suspect"] = gt_df["gt_suspect"].astype(bool)

    print("  corriendo config congelada (logica original, algo_v1_param) ...")
    detail_frozen = build_detail_df(cache, run_v1_param, frozen_params)
    m_frozen, _, fp_frozen = evaluate(detail_frozen, gt_df, window=3, label=frozen_name)

    print("  corriendo candidate_fix (logica corregida, mismos umbrales) ...")
    detail_fix = build_detail_df(cache, run_candidate_fix, DEFAULT_PARAMS)
    m_fix, _, fp_fix = evaluate(detail_fix, gt_df, window=3, label="candidate_fix")

    rows = [
        metrics_row(frozen_name, frozen_params, m_frozen, n_dev, fp_frozen),
        metrics_row("candidate_fix", DEFAULT_PARAMS, m_fix, n_dev, fp_fix),
    ]
    results = pd.DataFrame(rows)
    out_path = os.path.join(CANDIDATE_FIX_DIR, "candidate_fix_vs_frozen_dev.csv")
    results.to_csv(out_path, index=False)

    cols = ["config_name", "TP_confirmado", "FP_confirmado", "precision_confirmado",
            "recall_ANY", "FP_por_1000_viajes", "posible_cambio_por_1000_viajes"]
    print(f"\nResultados sobre DEV completo ({n_dev} viajes):")
    print(results[cols].to_string(index=False))
    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()
