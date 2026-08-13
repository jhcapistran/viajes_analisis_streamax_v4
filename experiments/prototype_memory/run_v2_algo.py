"""
Corre el algoritmo v2 (multiprototipo + candidato + CUSUM) sobre TODOS los
viajes de ambos datasets (igual que run_memory_algo.py para v1), generando
memory_detail_v2_{name}.csv. No filtra por split aca: el filtrado DEV/TEST
se hace despues, en tiempo de evaluacion (eval_split.py), para poder reusar
el mismo detalle sin tener que re-correr nada al pasar de DEV a TEST.
"""
import os
import pandas as pd

from common import CSV_FILES, EXP_DIR, load_raw_csv, preprocess_trip
from algo_memory_v2 import run_prototype_memory_v2

OUT_DIR = os.path.join(EXP_DIR, "splits")
os.makedirs(OUT_DIR, exist_ok=True)


def run_dataset(name, path):
    print(f"\n=== {name} (prototype memory v2) ===")
    df = load_raw_csv(path)
    trips = df['trip_id'].unique()

    all_records = []
    for t in trips:
        df_trip = df[df['trip_id'] == t]
        df_clean = preprocess_trip(df_trip)
        if df_clean is None:
            continue
        recs = run_prototype_memory_v2(df_clean)
        for r in recs:
            r['ASSET_ID'] = t
        all_records.extend(recs)

    detail = pd.DataFrame(all_records)
    out_path = os.path.join(OUT_DIR, f"memory_detail_v2_{name}.csv")
    detail.to_csv(out_path, index=False)
    print(f"  {len(detail)} filas -> {out_path}")


if __name__ == "__main__":
    for name, path in CSV_FILES.items():
        run_dataset(name, path)
