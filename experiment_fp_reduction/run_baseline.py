"""
Corre el detector BASELINE (main_analisis_completo_v2.process_asset_group,
SIN modificar) por viaje (trip_id) sobre los 2 CSV grandes, y construye el
GT (con GT_SUSPECT) para cada viaje usando el mismo set de frames que vio
el detector.

Salidas (en este mismo folder, aisladas del resto del repo):
  - baseline_detail_<dataset>.csv   (una fila por frame procesado)
  - gt_events_<dataset>.csv         (un evento por cambio de conductor GT)
"""
import os
import pandas as pd

from common import CSV_FILES, EXP_DIR, load_raw_csv, preprocess_trip, build_gt_for_trip
from main_analisis_completo_v2 import process_asset_group


def run_dataset(name, path):
    print(f"\n=== {name}: {path} ===")
    df = load_raw_csv(path)
    trips = df['trip_id'].unique()
    print(f"  {len(trips)} viajes, {len(df)} filas con embedding")

    all_records = []
    all_gt = []
    n_ok = 0
    for t in trips:
        df_trip = df[df['trip_id'] == t]
        results = process_asset_group(df_trip, str(t))
        if not results:
            continue
        n_ok += 1
        all_records.extend(results)

        df_clean = preprocess_trip(df_trip)
        if df_clean is None:
            continue
        events = build_gt_for_trip(df_clean)
        for e in events:
            e['trip_id'] = t
            all_gt.append(e)

    print(f"  viajes con salida: {n_ok}")
    detail = pd.DataFrame(all_records)
    gt = pd.DataFrame(all_gt)
    detail.to_csv(os.path.join(EXP_DIR, f"baseline_detail_{name}.csv"), index=False)
    gt.to_csv(os.path.join(EXP_DIR, f"gt_events_{name}.csv"), index=False)
    print(f"  detalle: {len(detail)} filas -> baseline_detail_{name}.csv")
    print(f"  GT eventos: {len(gt)} ({gt['gt_suspect'].sum() if len(gt) else 0} suspect) -> gt_events_{name}.csv")
    return detail, gt


if __name__ == "__main__":
    for name, path in CSV_FILES.items():
        run_dataset(name, path)
