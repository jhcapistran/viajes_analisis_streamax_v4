"""
DEV/TEST split reproducible, por trip_id (nunca por frames).

~75% DEV / ~25% TEST, estratificado por (dataset, tiene_cambio_real) para
garantizar que ambos splits tengan cambios reales de conductor en
proporcion similar. TEST queda congelado: no se debe mirar ni ajustar nada
con el hasta la evaluacion final.

Fuente de verdad de "cambio real" = gt_events_{dataset}.csv (evento con
gt_suspect=False), generado en la fase anterior del experimento (no se
vuelve a construir GT aca).

Salida: splits.csv (trip_id, dataset, n_gt_clean, n_gt_suspect, split)
"""
import os

import numpy as np
import pandas as pd

from common import EXP_DIR, DATA_DIR

SEED = 42
TEST_FRACTION = 0.25
OUT_DIR = os.path.join(EXP_DIR, "splits")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    rng = np.random.default_rng(SEED)

    rows = []
    for name in ["random_04", "reviewed_07"]:
        bd = pd.read_csv(os.path.join(DATA_DIR, f"baseline_detail_{name}.csv"), usecols=["ASSET_ID"])
        all_trips = sorted(bd["ASSET_ID"].unique().tolist())

        gt = pd.read_csv(os.path.join(DATA_DIR, f"gt_events_{name}.csv"))
        gt["gt_suspect"] = gt["gt_suspect"].astype(bool)
        n_clean = gt[~gt["gt_suspect"]].groupby("trip_id").size()
        n_suspect = gt[gt["gt_suspect"]].groupby("trip_id").size()

        for t in all_trips:
            rows.append({
                "trip_id": t,
                "dataset": name,
                "n_gt_clean": int(n_clean.get(t, 0)),
                "n_gt_suspect": int(n_suspect.get(t, 0)),
            })

    df = pd.DataFrame(rows)
    df["has_change"] = df["n_gt_clean"] > 0
    df["strata"] = df["dataset"] + "_" + df["has_change"].map({True: "cambio", False: "sin_cambio"})

    split = pd.Series(index=df.index, dtype=object)
    for stratum, idx in df.groupby("strata").groups.items():
        idx = list(idx)
        rng.shuffle(idx)
        n_test = max(1, round(len(idx) * TEST_FRACTION)) if df.loc[idx, "has_change"].any() else round(len(idx) * TEST_FRACTION)
        test_idx = idx[:n_test]
        dev_idx = idx[n_test:]
        split.loc[test_idx] = "TEST"
        split.loc[dev_idx] = "DEV"

    df["split"] = split
    out_path = os.path.join(OUT_DIR, "splits.csv")
    df.to_csv(out_path, index=False)

    print(f"Seed: {SEED}")
    print(f"Total viajes: {len(df)}")
    for s in ["DEV", "TEST"]:
        sub = df[df["split"] == s]
        print(f"\n{s}: {len(sub)} viajes ({100 * len(sub) / len(df):.1f}%)")
        for name in ["random_04", "reviewed_07"]:
            sub2 = sub[sub["dataset"] == name]
            print(f"  {name}: {len(sub2)} viajes, {sub2['n_gt_clean'].sum()} cambios reales "
                  f"({(sub2['has_change']).sum()} viajes con >=1 cambio)")
    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()
