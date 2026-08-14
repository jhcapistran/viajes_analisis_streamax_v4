"""
EXPLORACION (solo lectura, no escribe nada) de los 9 casos "cambio_real"
de la revision manual de los 45 FP de la config congelada "3-de-4"
(frozen_fp_review_E_w4_s3_c035_d085_p06.csv, exportado por el usuario
desde reviews/frozen_fp_review_45/index.html).

Objetivo: para cada caso, reconstruir con la MISMA logica de preprocesado
(common.preprocess_trip) + el mismo algoritmo instrumentado
(algo_v1_instrumented, config congelada) el frame de inicio del candidato
que confirmo el cambio, y el run de identity_id (en el CSV crudo) que
"tapaba" ese cambio real. Imprime, por caso, el plan de correccion
propuesto (que filas de que CSV, de que gs_path a que gs_path, cambiarian
identity_id de que valor a que valor nuevo) SIN escribir nada. Se revisa
el plan antes de aplicar ningun fix real al CSV.

Uso:
    uv run python explore_real_changes.py /ruta/al/frozen_fp_review_E_w4_s3_c035_d085_p06.csv
"""
import os
import sys

import pandas as pd

TUNING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TUNING_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(TUNING_DIR), "prototype_memory"))

from common import CSV_FILES, EXP_DIR, DATA_DIR, load_raw_csv, preprocess_trip  # noqa: E402
from precompute import precompute_trips  # noqa: E402
from algo_v1_instrumented import run_v1_instrumented  # noqa: E402

FROZEN_PARAMS = dict(
    candidate_window=4,
    candidate_min_support=3,
    coherence_threshold=0.35,
    min_avg_dist_confirm=0.85,
    min_phys_confirm=0.6,
)


def main():
    review_csv = sys.argv[1]
    review = pd.read_csv(review_csv, dtype=str)
    real = review[review["respuesta"] == "cambio_real"].copy()
    print(f"{len(real)} casos 'cambio_real' en {review_csv}")

    splits = pd.read_csv(os.path.join(EXP_DIR, "splits", "splits.csv"))
    dev = splits[splits["split"] == "DEV"].copy()
    trip_to_dataset = dict(zip(dev["trip_id"].astype(str), dev["dataset"]))

    raw_cache = {}

    for row in real.itertuples():
        trip_id_str = row.trip_id
        dataset = row.dataset
        alert_frame = int(row.alert_frame)

        if dataset not in raw_cache:
            print(f"\nCargando CSV crudo de {dataset}...")
            raw_cache[dataset] = load_raw_csv(CSV_FILES[dataset])
        raw = raw_cache[dataset]

        # trip_id en splits.csv/CSV crudo puede ser int; comparar como str.
        df_trip = raw[raw["trip_id"].astype(str) == trip_id_str]
        if df_trip.empty:
            print(f"CASO trip={trip_id_str} frame={alert_frame}: NO ENCONTRADO en {dataset}, se omite")
            continue

        df_clean = preprocess_trip(df_trip)
        if df_clean is None:
            print(f"CASO trip={trip_id_str} frame={alert_frame}: df_clean vacio, se omite")
            continue
        df_clean = df_clean.reset_index(drop=True)

        frames = []
        for _, r in df_clean.iterrows():
            frames.append({
                "emb": None,  # no hace falta para este script (no reusamos la conf. exacta de embeddings aqui)
            })
        # Necesitamos los embeddings reales para CANDIDATE_FRAMES: recargar via precompute-like.
        from common import load_emb
        frames = []
        for _, r in df_clean.iterrows():
            frames.append({
                "emb": load_emb(r.get("embedding")),
                "speed": float(r.get("speed", 0.0)),
                "ts_seconds": int(r["ts_seconds"]),
                "driver_id": r.get("driver_id", "N/A"),
                "identity_id": r.get("identity_id", ""),
                "empty_cabin": r.get("empty_cabin", ""),
                "image_file": r.get("image_file", "N/A"),
            })

        recs = run_v1_instrumented(frames, FROZEN_PARAMS, trip_id=trip_id_str)
        confirm_rec = recs[alert_frame]
        print(f"\n=== CASO trip={trip_id_str} dataset={dataset} alert_frame={alert_frame} ===")
        print(f"  decision en alert_frame: {confirm_rec['DECISION_SISTEMA']}")
        if confirm_rec["DECISION_SISTEMA"] != "CAMBIO_CONFIRMADO":
            print("  AVISO: no coincide con CAMBIO_CONFIRMADO, revisar manualmente")
            continue

        candidate_frames = confirm_rec["CANDIDATE_FRAMES"] or [alert_frame]
        start_frame = min(candidate_frames)

        old_identity = str(df_clean.iloc[start_frame]["identity_id"])
        # extender el run: desde start_frame hacia atras y hacia adelante mientras identity_id == old_identity
        run_start = start_frame
        while run_start > 0 and str(df_clean.iloc[run_start - 1]["identity_id"]) == old_identity:
            run_start -= 1
        run_end = start_frame
        n = len(df_clean)
        while run_end + 1 < n and str(df_clean.iloc[run_end + 1]["identity_id"]) == old_identity:
            run_end += 1

        existing_ids = sorted(set(str(v) for v in df_clean["identity_id"].dropna().unique() if str(v).strip() != ''))
        print(f"  identity_id existentes en el viaje: {existing_ids}")
        print(f"  candidate_frames={candidate_frames} start_frame={start_frame} old_identity='{old_identity}'")
        print(f"  run completo de '{old_identity}' en df_clean: [{run_start}, {run_end}] (len={run_end-run_start+1})")
        print(f"  PROPUESTA: relabelar filas [{start_frame}, {run_end}] (desde el inicio del candidato hasta el "
              f"final del run) de identity_id '{old_identity}' -> nuevo valor, dejando "
              f"[{run_start}, {start_frame-1}] (antes del candidato) sin tocar.")
        print(f"  gs_path de start_frame: {df_clean.iloc[start_frame]['gs_path']}")
        print(f"  gs_path de run_end: {df_clean.iloc[run_end]['gs_path']}")


if __name__ == "__main__":
    main()
