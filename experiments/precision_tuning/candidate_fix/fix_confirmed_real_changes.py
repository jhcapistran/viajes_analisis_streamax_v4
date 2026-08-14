"""
CORRECCION de GT: identity_id que "tapaba" un cambio real de conductor.

Contexto: la revision manual visual de los 45 FP de la config congelada
"3-de-4" (reviews/frozen_fp_review_45/index.html, revision hecha SIN ver
identity_id) marco 9 de esos 45 casos como "cambio_real": el algoritmo
confirmo un cambio de conductor que en realidad SI ocurrio, pero el CSV
crudo seguia etiquetando esas filas con el mismo identity_id de ANTES del
cambio (el labeler/reviewer original no lo detecto dentro de ese run), por
lo que el GT derivado de identity_id (common.build_gt_for_trip) no lo conto
como evento -> conto como FP estricto siendo en realidad un TP.

Este script:
  1. Recalcula, para cada uno de los 9 casos, el frame de inicio del
     candidato que confirmo el cambio (misma logica que
     algo_v1_instrumented sobre la config congelada) y el run de
     identity_id en el que cae ese inicio.
  2. Verifica que TODAS las filas de [start_frame, run_end] tengan
     exactamente el mismo identity_id viejo (si no, aborta ese caso sin
     tocar nada: la logica asume un bloque uniforme).
  3. Relabela esas filas a un identity_id NUEVO (proximo numero libre
     dentro del mismo viaje, ej. si existen Identity_1/Identity_3 usa
     Identity_4), dejando SIN TOCAR las filas anteriores al inicio del
     candidato.
  4. Escribe los CSV crudos correspondientes (random_trips_data_2026_04.csv
     / all_reviewed_trips_data_2026_07.csv) con el modulo csv (streaming,
     sin pandas, para no arriesgar reformateo de floats/embeddings),
     haciendo backup antes si no existe ya uno.
  5. Guarda un log de auditoria (correction_log_real_changes.csv) con el
     detalle de cada cambio aplicado.

Uso:
    uv run python fix_confirmed_real_changes.py /ruta/al/frozen_fp_review_E_w4_s3_c035_d085_p06.csv
    uv run python fix_confirmed_real_changes.py /ruta/al/....csv --dry-run
"""
import argparse
import csv
import os
import shutil
import sys

import pandas as pd

TUNING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(TUNING_DIR))
sys.path.insert(0, TUNING_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(TUNING_DIR), "prototype_memory"))

from common import CSV_FILES, EXP_DIR, load_raw_csv, preprocess_trip, load_emb  # noqa: E402
from algo_v1_instrumented import run_v1_instrumented  # noqa: E402

FROZEN_PARAMS = dict(
    candidate_window=4,
    candidate_min_support=3,
    coherence_threshold=0.35,
    min_avg_dist_confirm=0.85,
    min_phys_confirm=0.6,
)

CANDIDATE_FIX_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(CANDIDATE_FIX_DIR, "correction_log_real_changes.csv")


def next_identity_value(existing_ids):
    """existing_ids: valores tipo 'Identity_3'. Devuelve el proximo numero
    libre en ese mismo formato."""
    max_n = 0
    for v in existing_ids:
        if v.startswith("Identity_"):
            try:
                n = int(v.split("_", 1)[1])
                max_n = max(max_n, n)
            except ValueError:
                continue
    return f"Identity_{max_n + 1}"


def build_plan(review_csv):
    review = pd.read_csv(review_csv, dtype=str)
    real = review[review["respuesta"] == "cambio_real"].copy()

    splits = pd.read_csv(os.path.join(EXP_DIR, "splits", "splits.csv"))
    dev = splits[splits["split"] == "DEV"].copy()
    trip_to_dataset = dict(zip(dev["trip_id"].astype(str), dev["dataset"]))

    raw_cache = {}
    plan = []  # list of dict: dataset, trip_id, gs_paths (set), old_identity, new_identity, ...

    for row in real.itertuples():
        trip_id_str = row.trip_id
        dataset = row.dataset
        alert_frame = int(row.alert_frame)

        if dataset not in raw_cache:
            raw_cache[dataset] = load_raw_csv(CSV_FILES[dataset])
        raw = raw_cache[dataset]

        df_trip = raw[raw["trip_id"].astype(str) == trip_id_str]
        if df_trip.empty:
            print(f"AVISO: trip={trip_id_str} no encontrado en {dataset}, se omite")
            continue
        df_clean = preprocess_trip(df_trip)
        if df_clean is None:
            print(f"AVISO: trip={trip_id_str} df_clean vacio, se omite")
            continue
        df_clean = df_clean.reset_index(drop=True)

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
        if confirm_rec["DECISION_SISTEMA"] != "CAMBIO_CONFIRMADO":
            print(f"AVISO: trip={trip_id_str} frame={alert_frame} ya no es CAMBIO_CONFIRMADO, se omite")
            continue

        candidate_frames = confirm_rec["CANDIDATE_FRAMES"] or [alert_frame]
        start_frame = min(candidate_frames)
        old_identity = str(df_clean.iloc[start_frame]["identity_id"])

        run_end = start_frame
        n = len(df_clean)
        while run_end + 1 < n and str(df_clean.iloc[run_end + 1]["identity_id"]) == old_identity:
            run_end += 1

        block = df_clean.iloc[start_frame:run_end + 1]
        if not (block["identity_id"].astype(str) == old_identity).all():
            print(f"AVISO: trip={trip_id_str} bloque [{start_frame},{run_end}] no es uniforme, se omite")
            continue

        existing_ids = sorted(set(
            str(v) for v in df_clean["identity_id"].dropna().unique() if str(v).strip() != ""
        ))
        new_identity = next_identity_value(existing_ids)

        gs_paths = set(block["gs_path"].astype(str).tolist())

        plan.append({
            "dataset": dataset,
            "trip_id": trip_id_str,
            "alert_frame": alert_frame,
            "start_frame": start_frame,
            "run_end": run_end,
            "n_rows": len(gs_paths),
            "old_identity": old_identity,
            "new_identity": new_identity,
            "gs_paths": gs_paths,
        })

    return plan


def apply_plan(plan, dry_run=True):
    by_dataset = {}
    for item in plan:
        by_dataset.setdefault(item["dataset"], []).append(item)

    log_rows = []
    for dataset, items in by_dataset.items():
        input_file = CSV_FILES[dataset]
        backup_file = os.path.splitext(input_file)[0] + ".pre_real_change_fix.bak.csv"
        tmp_file = input_file + ".tmp_fix"

        # mapa (trip_id, gs_path) -> new_identity para este dataset
        path_to_new = {}
        for item in items:
            for gp in item["gs_paths"]:
                path_to_new[(item["trip_id"], gp)] = item["new_identity"]

        print(f"\nDataset {dataset}: {len(items)} viaje(s), {len(path_to_new)} fila(s) a relabelar")
        for item in items:
            print(f"  trip={item['trip_id']} frames[{item['start_frame']},{item['run_end']}] "
                  f"'{item['old_identity']}' -> '{item['new_identity']}' ({item['n_rows']} filas)")
            log_rows.append(item)

        if dry_run:
            continue

        if not os.path.exists(backup_file):
            print(f"  Creando backup en {backup_file} ...")
            shutil.copy2(input_file, backup_file)
        else:
            print(f"  Backup ya existe, no se sobreescribe: {backup_file}")

        n_rows = 0
        n_fixed = 0
        with open(input_file, "r", newline="", encoding="utf-8") as fin, \
             open(tmp_file, "w", newline="", encoding="utf-8") as fout:
            reader = csv.reader(fin)
            writer = csv.writer(fout)
            header = next(reader)
            writer.writerow(header)
            idx_trip = header.index("trip_id")
            idx_gs = header.index("gs_path")
            idx_identity = header.index("identity_id")

            for row in reader:
                n_rows += 1
                key = (row[idx_trip], row[idx_gs])
                new_val = path_to_new.get(key)
                if new_val is not None:
                    row[idx_identity] = new_val
                    n_fixed += 1
                writer.writerow(row)

        os.replace(tmp_file, input_file)
        print(f"  Filas totales: {n_rows}, filas corregidas: {n_fixed} (esperadas {len(path_to_new)})")
        if n_fixed != len(path_to_new):
            print("  ATENCION: el numero de filas corregidas no coincide con lo esperado, revisar.")

    if log_rows:
        log_df = pd.DataFrame([{k: v for k, v in r.items() if k != "gs_paths"} for r in log_rows])
        log_df.to_csv(LOG_PATH, index=False)
        print(f"\nLog de auditoria guardado en {LOG_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("review_csv")
    parser.add_argument("--dry-run", action="store_true",
                         help="Solo muestra el plan de correccion, no escribe ningun CSV.")
    args = parser.parse_args()

    plan = build_plan(args.review_csv)
    print(f"\n{len(plan)} caso(s) 'cambio_real' con plan de correccion valido.")
    apply_plan(plan, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
