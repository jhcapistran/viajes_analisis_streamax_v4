"""
Corre la configuracion CONGELADA "3 de 4" (winner_config.json,
E_w4_s3_c035_d085_p06: candidate_window=4, candidate_min_support=3,
exige 3 de 4 observaciones coherentes en vez de 2 de 3 de v1 default)
sobre CUALQUIER CSV de viajes con el mismo formato de entrada que los
CSV grandes del experimento (columnas: trip_id, asset_id, gs_path,
embedding, speed, identity_id opcional).

No modifica algo_v1_param.py ni winner_config.json; solo los reutiliza.
Si el CSV trae identity_id, ademas construye GT y evalua (precision/recall,
FP estrictos); si no, solo genera el detalle de decisiones por frame.

Uso:
    uv run python run_single_dataset.py /ruta/a/mi_dataset.csv
    uv run python run_single_dataset.py /ruta/a/mi_dataset.csv --out-dir /tmp/salida
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prototype_memory"))
from common import load_raw_csv, preprocess_trip, load_emb, build_gt_for_trip  # noqa: E402

TUNING_DIR = os.path.dirname(os.path.abspath(__file__))
WINNER_CONFIG_PATH = os.path.join(TUNING_DIR, "winner_config.json")

sys.path.insert(0, TUNING_DIR)
from algo_v1_param import run_v1_param  # noqa: E402
from evaluate import evaluate  # noqa: E402


def load_frozen_params():
    """Lee winner_config.json (config congelada 'E_w4_s3_c035_d085_p06',
    3 de 4 observaciones coherentes) y devuelve el dict de parametros que
    espera run_v1_param."""
    with open(WINNER_CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    return {
        "candidate_window": cfg["candidate_window"],
        "candidate_min_support": cfg["candidate_min_support"],
        "coherence_threshold": cfg["coherence_threshold"],
        "min_avg_dist_confirm": cfg["min_avg_dist_confirm"],
        "min_phys_confirm": cfg["min_phys_confirm"],
        "memory_size": cfg["memory_size"],
    }, cfg["config_name"]


def precompute_trip_frames(df_trip):
    """Misma logica de precompute.py pero sobre un df ya preprocesado con
    preprocess_trip, sin depender de CSV_FILES fijos."""
    frames = []
    for _, row in df_trip.iterrows():
        frames.append({
            "emb": load_emb(row.get("embedding")),
            "speed": float(row.get("speed", 0.0)),
            "ts_seconds": int(row["ts_seconds"]),
            "driver_id": row.get("driver_id", "N/A"),
            "identity_id": row.get("identity_id", ""),
            "empty_cabin": row.get("empty_cabin", ""),
            "image_file": row.get("image_file", "N/A"),
        })
    return frames


def run_single_dataset(csv_path, out_dir=None, label=None):
    params, config_name = load_frozen_params()
    label = label or os.path.splitext(os.path.basename(csv_path))[0]
    out_dir = out_dir or TUNING_DIR

    print(f"Config congelada usada: {config_name} -> {params}")
    print(f"Cargando {csv_path} ...")
    df = load_raw_csv(csv_path)
    if "trip_id" not in df.columns:
        raise ValueError("El CSV debe tener columna 'trip_id' para agrupar por viaje.")
    trips = df["trip_id"].unique()
    print(f"  {len(trips)} viajes encontrados")

    has_identity = "identity_id" in df.columns

    all_records = []
    all_gt = []
    n_skipped = 0
    for t in trips:
        df_trip = df[df["trip_id"] == t]
        df_clean = preprocess_trip(df_trip)
        if df_clean is None:
            n_skipped += 1
            continue

        frames = precompute_trip_frames(df_clean)
        recs = run_v1_param(frames, params, trip_id=t)
        all_records.extend(recs)

        if has_identity:
            events = build_gt_for_trip(df_clean)
            for e in events:
                e["trip_id"] = t
            all_gt.extend(events)

    if n_skipped:
        print(f"  {n_skipped} viajes descartados (menos de min_images_per_asset frames validos)")

    detail_df = pd.DataFrame(all_records)
    os.makedirs(out_dir, exist_ok=True)
    detail_path = os.path.join(out_dir, f"single_detail_{label}.csv")
    detail_df.to_csv(detail_path, index=False)
    print(f"  {len(detail_df)} filas -> {detail_path}")

    n_confirmado = (detail_df["DECISION_SISTEMA"] == "CAMBIO_CONFIRMADO").sum() if len(detail_df) else 0
    n_posible = (detail_df["DECISION_SISTEMA"] == "POSIBLE_CAMBIO").sum() if len(detail_df) else 0
    print(f"  CAMBIO_CONFIRMADO: {n_confirmado}  POSIBLE_CAMBIO: {n_posible}")

    if not has_identity:
        print("  (CSV sin columna 'identity_id': no se calcula GT ni metricas de precision/recall)")
        return detail_df, None, None

    gt_df = pd.DataFrame(all_gt)
    if len(gt_df):
        gt_df["gt_suspect"] = gt_df["gt_suspect"].astype(bool)
    gt_path = os.path.join(out_dir, f"single_gt_{label}.csv")
    gt_df.to_csv(gt_path, index=False)
    print(f"  {len(gt_df)} eventos GT -> {gt_path}")

    metrics, tp_df, fp_df = evaluate(detail_df, gt_df, label=label)
    fp_path = os.path.join(out_dir, f"single_fp_{label}.csv")
    tp_path = os.path.join(out_dir, f"single_tp_{label}.csv")
    fp_df.to_csv(fp_path, index=False)
    tp_df.to_csv(tp_path, index=False)

    print("\nMetricas:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return detail_df, gt_df, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Corre la config congelada '3 de 4' (winner_config.json) sobre cualquier CSV de viajes."
    )
    parser.add_argument("csv_path", help="Ruta al CSV de entrada (mismo formato que los CSV del experimento).")
    parser.add_argument("--out-dir", default=None, help="Carpeta de salida (default: esta misma carpeta).")
    parser.add_argument("--label", default=None, help="Etiqueta para nombrar los archivos de salida (default: nombre del CSV).")
    args = parser.parse_args()

    run_single_dataset(args.csv_path, out_dir=args.out_dir, label=args.label)


if __name__ == "__main__":
    main()
