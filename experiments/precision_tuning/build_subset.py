"""
PASO 1: subset de ajuste (dentro de DEV) para el grid search de precision
de CAMBIO_CONFIRMADO sobre memory v1 (algo_memory.py). NUNCA toca TEST.

Subset = union de:
  (a) TODOS los viajes DEV donde memory v1 tuvo un FP de CAMBIO_CONFIRMADO
      (v2_dev_test/memory_v1_fp_DEV.csv, ya calculado sobre DEV).
  (b) TODOS los viajes DEV con >=1 cambio real limpio (gt_clean, splits.csv).
  (c) 150 viajes DEV sin cambio real, elegidos al azar (seed=42), como
      control (para no destruir precision/():viajes tranquilos).

Guarda experiment_fp_reduction/precision_tuning/subset_trips.csv con
columnas: trip_id, dataset, is_fp_v1, is_change, is_control.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prototype_memory"))
from common import EXP_DIR

SEED = 42
N_CONTROL = 150

TUNING_DIR = os.path.join(EXP_DIR, "precision_tuning")
os.makedirs(TUNING_DIR, exist_ok=True)


def main():
    splits = pd.read_csv(os.path.join(EXP_DIR, "splits", "splits.csv"))
    dev = splits[splits["split"] == "DEV"].copy()

    fp_v1 = pd.read_csv(os.path.join(EXP_DIR, "splits", "memory_v1_fp_DEV.csv"))
    fp_trip_ids = set(fp_v1["trip_id"].unique().tolist())

    change_trip_ids = set(dev.loc[dev["has_change"], "trip_id"].tolist())

    no_change_dev = dev.loc[~dev["has_change"] & ~dev["trip_id"].isin(fp_trip_ids), "trip_id"].tolist()
    rng = np.random.default_rng(SEED)
    control_trip_ids = set(rng.choice(no_change_dev, size=min(N_CONTROL, len(no_change_dev)), replace=False).tolist())

    all_ids = fp_trip_ids | change_trip_ids | control_trip_ids

    out = dev[dev["trip_id"].isin(all_ids)][["trip_id", "dataset"]].copy()
    out["is_fp_v1"] = out["trip_id"].isin(fp_trip_ids)
    out["is_change"] = out["trip_id"].isin(change_trip_ids)
    out["is_control"] = out["trip_id"].isin(control_trip_ids)

    out_path = os.path.join(TUNING_DIR, "subset_trips.csv")
    out.to_csv(out_path, index=False)

    print(f"FP v1 (DEV): {len(fp_trip_ids)} viajes")
    print(f"Cambio real (DEV): {len(change_trip_ids)} viajes")
    print(f"Control sin cambio (DEV): {len(control_trip_ids)} viajes")
    print(f"Subset total (union, sin duplicados): {len(out)} viajes")
    print(f"  por dataset: {out['dataset'].value_counts().to_dict()}")
    print(f"Guardado en {out_path}")


if __name__ == "__main__":
    main()
