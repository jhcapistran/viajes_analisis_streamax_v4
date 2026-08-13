"""
Corre el algoritmo nuevo (Prototype Memory + confirmacion temporal) por
viaje sobre los 2 CSV grandes (mismo preprocesado que el baseline) y lo
evalua igual que a este, para comparar manzanas con manzanas.
"""
import os
import pandas as pd

from common import CSV_FILES, EXP_DIR, load_raw_csv, preprocess_trip
from algo_memory import run_prototype_memory
from evaluate import evaluate


def run_dataset(name, path):
    print(f"\n=== {name} (prototype memory) ===")
    df = load_raw_csv(path)
    trips = df['trip_id'].unique()

    all_records = []
    for t in trips:
        df_trip = df[df['trip_id'] == t]
        df_clean = preprocess_trip(df_trip)
        if df_clean is None:
            continue
        recs = run_prototype_memory(df_clean)
        for r in recs:
            r['ASSET_ID'] = t
        all_records.extend(recs)

    detail = pd.DataFrame(all_records)
    detail.to_csv(os.path.join(EXP_DIR, f"memory_detail_{name}.csv"), index=False)
    print(f"  {len(detail)} filas -> memory_detail_{name}.csv")

    gt = pd.read_csv(os.path.join(EXP_DIR, f"gt_events_{name}.csv"))
    gt['gt_suspect'] = gt['gt_suspect'].astype(bool)
    m, tp, fp = evaluate(detail, gt, label=f"memory_{name}")
    for k, v in m.items():
        print(f"  {k}: {v}")
    fp.to_csv(os.path.join(EXP_DIR, f"memory_fp_{name}.csv"), index=False)
    tp.to_csv(os.path.join(EXP_DIR, f"memory_tp_{name}.csv"), index=False)
    return detail, m


if __name__ == "__main__":
    for name, path in CSV_FILES.items():
        run_dataset(name, path)
