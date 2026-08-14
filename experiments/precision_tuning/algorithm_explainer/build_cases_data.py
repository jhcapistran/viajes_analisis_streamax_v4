"""
Construye el dataset JSON con los casos reales elegidos para el HTML
explicador de la config congelada "3-de-4" (E_w4_s3_c035_d085_p06,
algo_v1_param.py). Solo lectura/visualizacion: no cambia el algoritmo, no
corre experimentos nuevos, no toca TEST. Usa algo_v1_explainer.py (copia
exacta de la logica de decision, con diagnosticos adicionales de solo
lectura).

Casos elegidos (justificacion en README de esta carpeta / mensaje al usuario):
  - 3 TP: cambios reales reclasificados (correction_log_real_changes.csv),
    confirmados correctamente por el algoritmo.
  - 3 FP: de los 45 FP originales de la config congelada, los que NO estan
    en la lista de reclasificados siguen siendo FP "misma persona"
    confirmados por revision manual humana (ver memoria del repo).
  - 1 caso "vuelve": una ventana candidata que llega a soporte completo
    (4/4) pero no supera el umbral de distancia promedio (0.85) y el
    conductor anterior reaparece (dist_mem < 0.5) antes de confirmar.
  - 1 caso "dificil": el mismo caso FP #3 (716821775367857) ya construye
    varias ventanas candidatas fallidas antes de la confirmacion final,
    ilustrando la mecanica de ventana deslizante.

Imagenes: reusa lo ya descargado en
experiments/precision_tuning/reviews/frozen_fp_review_45/images/ y descarga
SOLO los frames adicionales que hagan falta (gcloud storage cp), sin tocar
nada de esa carpeta de revision original.
"""
import os
import sys
import json
import subprocess

import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TUNING_DIR = os.path.dirname(HERE)
PROTOTYPE_DIR = os.path.join(os.path.dirname(TUNING_DIR), "prototype_memory")
sys.path.insert(0, TUNING_DIR)
sys.path.insert(0, PROTOTYPE_DIR)
sys.path.insert(0, HERE)

from common import CSV_FILES, load_raw_csv, preprocess_trip, cosine_dist  # noqa: E402
from precompute import precompute_trips  # noqa: E402
from algo_v1_explainer import run_v1_explainer, FROZEN_PARAMS, VISUAL_MATCH  # noqa: E402

OUT_DIR = os.path.join(HERE, "output")
IMAGES_DIR = os.path.join(OUT_DIR, "images")
EXISTING_IMAGES_DIR = os.path.join(TUNING_DIR, "reviews", "frozen_fp_review_45", "images")
SPLITS_CSV = os.path.join(PROTOTYPE_DIR, "splits", "splits.csv")

CONFIG_NAME = "E_w4_s3_c035_d085_p06"

# --- Definicion de los casos (trip_id, alert_frame de interes, ventana a mostrar) ---
CASES = [
    dict(case_id="tp1", kind="correcto", dataset="random_04", trip_id=1867131776405754,
         alert_frame=17, lo=10, hi=20,
         titulo="Cambio real detectado correctamente #1",
         gt_label="CAMBIO_REAL (confirmado por revision manual: identity_id crudo seguia mal etiquetado, el algoritmo detecto el cambio real que el GT original no habia marcado)"),
    dict(case_id="tp2", kind="correcto", dataset="reviewed_07", trip_id=1211261784361700,
         alert_frame=4, lo=0, hi=8,
         titulo="Cambio real detectado correctamente #2",
         gt_label="CAMBIO_REAL (confirmado por revision manual, identity_id crudo corregido de Identity_3 a Identity_4 en frames [1,93])"),
    dict(case_id="tp3", kind="correcto", dataset="reviewed_07", trip_id=1065661783416050,
         alert_frame=10, lo=4, hi=14,
         titulo="Cambio real detectado correctamente #3",
         gt_label="CAMBIO_REAL (confirmado por revision manual, identity_id crudo corregido de Identity_4 a Identity_5 en frames [7,57])"),
    dict(case_id="fp1", kind="fp", dataset=None, trip_id=1866621775112847,
         alert_frame=4, lo=0, hi=8,
         titulo="Falso positivo confirmado #1 (misma persona)",
         gt_label="MISMA PERSONA (revision manual humana, causa: variabilidad de embedding, no confusion de identidad)"),
    dict(case_id="fp2", kind="fp", dataset=None, trip_id=1052671784811920,
         alert_frame=61, lo=49, hi=65,
         titulo="Falso positivo confirmado #2 (misma persona, distancia baja)",
         gt_label="MISMA PERSONA (revision manual humana)"),
    dict(case_id="fp3", kind="fp_dificil", dataset=None, trip_id=716821775367857,
         alert_frame=21, lo=9, hi=24,
         titulo="Falso positivo confirmado #3 (dificil: varias ventanas candidatas fallidas antes de confirmar)",
         gt_label="MISMA PERSONA (revision manual humana)"),
    dict(case_id="vuelve", kind="vuelve", dataset=None, trip_id=1808801775184559,
         alert_frame=None, lo=3, hi=13,
         titulo="Anomalia con ventana completa que NO confirma: vuelve el conductor anterior",
         gt_label="identity_id crudo: no cambia en este tramo (el algoritmo abre candidato pero lo descarta correctamente)"),
]

TRIP_IDS = sorted({c["trip_id"] for c in CASES})


def get_dataset_map():
    splits = pd.read_csv(SPLITS_CSV)
    return dict(zip(splits["trip_id"], splits["dataset"]))


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    trip_to_dataset = get_dataset_map()
    for c in CASES:
        if c["dataset"] is None:
            c["dataset"] = trip_to_dataset[c["trip_id"]]

    trip_ids_by_dataset = {"random_04": [], "reviewed_07": []}
    for t in TRIP_IDS:
        trip_ids_by_dataset[trip_to_dataset[t]].append(t)

    print("Precomputando embeddings de los", len(TRIP_IDS), "viajes elegidos...")
    cache = precompute_trips(trip_ids_by_dataset)

    print("Corriendo algo_v1_explainer (config congelada 3-de-4) sobre cada viaje...")
    records_by_trip = {t: run_v1_explainer(f, FROZEN_PARAMS, trip_id=t) for t, f in cache.items()}

    # df_clean crudo (para basename/gs_path de cada frame)
    raw_cache = {}

    def get_df_clean(trip_id, dataset):
        if dataset not in raw_cache:
            print(f"Cargando CSV crudo de {dataset}...")
            raw_cache[dataset] = load_raw_csv(CSV_FILES[dataset])
        raw = raw_cache[dataset]
        df_trip = raw[raw["trip_id"] == trip_id]
        df_clean = preprocess_trip(df_trip)
        df_clean = df_clean.reset_index(drop=True)
        df_clean["basename"] = df_clean["gs_path"].apply(lambda p: os.path.basename(str(p)))
        return df_clean

    output_cases = []
    need_download = {}  # trip_id -> set(gs_path)

    for c in CASES:
        trip_id = c["trip_id"]
        dataset = c["dataset"]
        recs = records_by_trip[trip_id]
        df_clean = get_df_clean(trip_id, dataset)
        n = len(recs)
        lo = max(0, c["lo"])
        hi = min(n - 1, c["hi"])

        frames_out = []
        for idx in range(lo, hi + 1):
            rec = recs[idx]
            row = df_clean.iloc[idx]
            gs_path = str(row["gs_path"])
            basename = row["basename"]
            frames_out.append({
                "frame_idx": idx,
                "ts_seconds": rec["TS_SECONDS"],
                "decision": rec["DECISION_SISTEMA"],
                "dist_mem": rec["DIST_MEM"],
                "dist_mem_breakdown": rec.get("DIST_MEM_BREAKDOWN"),
                "prev_frame_dist": rec["PREV_FRAME_DIST"],
                "prev_frame_dist_breakdown": rec.get("PREV_FRAME_DIST_BREAKDOWN"),
                "p_fisica": rec["P_FISICA"],
                "physics_breakdown": rec["PHYSICS_BREAKDOWN"],
                "speed": rec["SPEED"],
                "delta_t": rec["DELTA_T"],
                "memory_size": rec["MEMORY_SIZE"],
                "memory_frame_idx": rec["MEMORY_FRAME_IDX"],
                "pending_frames": rec["PENDING_FRAMES"],
                "pending_dists": rec["PENDING_DISTS"],
                "coherent_local": rec["COHERENT_LOCAL"],
                "support": rec["SUPPORT"],
                "avg_dist_vs_old": rec["AVG_DIST_VS_OLD"],
                "candidate_frames": rec["CANDIDATE_FRAMES"],
                "coherent_frames": rec["COHERENT_FRAMES"],
                "confirm_checks": rec["CONFIRM_CHECKS"],
                "gs_path": gs_path,
                "basename": basename,
            })
            need_download.setdefault(str(trip_id), set()).add(gs_path)

        output_cases.append({
            "case_id": c["case_id"],
            "kind": c["kind"],
            "titulo": c["titulo"],
            "gt_label": c["gt_label"],
            "trip_id": str(trip_id),
            "dataset": dataset,
            "alert_frame": c["alert_frame"],
            "n_total_frames": n,
            "frames": frames_out,
        })

    # -----------------------------------------------------------------
    # Imagenes: reusar lo ya descargado en frozen_fp_review_45, descargar
    # solo lo que falte, a la carpeta local de este explicador.
    # -----------------------------------------------------------------
    for trip_dir, gs_paths in need_download.items():
        existing_trip_dir = os.path.join(EXISTING_IMAGES_DIR, trip_dir)
        local_trip_dir = os.path.join(IMAGES_DIR, trip_dir)
        os.makedirs(local_trip_dir, exist_ok=True)
        missing = []
        for gs_path in gs_paths:
            basename = os.path.basename(gs_path)
            existing_path = os.path.join(existing_trip_dir, basename)
            local_path = os.path.join(local_trip_dir, basename)
            if os.path.exists(local_path):
                continue
            if os.path.exists(existing_path):
                # copiar (barato, local) en vez de re-descargar de GCS
                import shutil
                shutil.copyfile(existing_path, local_path)
            else:
                missing.append(gs_path)
        if missing:
            print(f"Descargando {len(missing)} imagenes faltantes para trip {trip_dir}...")
            cmd = ["gcloud", "storage", "cp", *sorted(missing), local_trip_dir + "/"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  AVISO: error descargando: {result.stderr[:400]}")

    # Verificar y asignar local_path relativo, quitar gs_path del JSON final
    missing_count = 0
    for case in output_cases:
        for fr in case["frames"]:
            trip_dir = case["trip_id"]
            basename = fr["basename"]
            local_path = os.path.join("images", trip_dir, basename)
            abs_path = os.path.join(OUT_DIR, local_path)
            if os.path.exists(abs_path):
                fr["local_path"] = local_path
            else:
                fr["local_path"] = None
                missing_count += 1
            fr.pop("gs_path", None)
    print(f"Verificacion de imagenes: {'todas existen' if missing_count == 0 else f'{missing_count} faltantes'}")

    with open(os.path.join(OUT_DIR, "cases.json"), "w", encoding="utf-8") as f:
        json.dump(output_cases, f, ensure_ascii=False, indent=1)
    print(f"cases.json escrito con {len(output_cases)} casos en {OUT_DIR}")


if __name__ == "__main__":
    main()
