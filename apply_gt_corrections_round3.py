"""Aplica las correcciones de GT confirmadas visualmente en esta ronda (post
borrado accidental de los scripts anteriores).

Casos revisados:
- Trip 1264001776343000: unico cambio real es Identity_1 -> Identity_2 @ 15:15:52.
  Todo lo que aparece como "Identity_1" desde 2026-04-16 15:14:52 en adelante
  (incluye el blip de 16:09-16:10, el blip de 15:34, 16:32-16:33, y el tramo
  sostenido 17:48:52 -> fin de viaje) es error de GT / drift de embedding,
  no cambio real de conductor. Se corrige todo a Identity_2.
- Trip 1648911775324423: sin cambio real. Se unifica a la identidad mayoritaria
  (Identity_2, 71 filas vs 15 de Identity_1).
- Trip 1809001775245550: sin cambio real. Se unifica a la identidad mayoritaria
  (Identity_1, 69 filas vs 57 de Identity_3).

Hace backup del CSV si no existe ya un backup de esta ronda.
"""

from pathlib import Path
import shutil

import pandas as pd

CSV_INPUT = Path("random_trips_data_2026_04.csv")
BACKUP = Path("random_trips_data_2026_04.pre_gt_fix_round3.bak.csv")

FULL_TRIP_MERGES = {
    "1648911775324423": ("Identity_1", "Identity_2"),
    "1809001775245550": ("Identity_3", "Identity_1"),
}

# trip_id -> (desde_timestamp, valor_incorrecto, valor_correcto)
SCOPED_PATCH = ("1264001776343000", "2026-04-16 15:14:52", "Identity_1", "Identity_2")


def main() -> None:
    if not BACKUP.exists():
        shutil.copy(CSV_INPUT, BACKUP)
        print(f"Backup creado en {BACKUP}")
    else:
        print(f"Backup ya existia en {BACKUP}, no se sobreescribe")

    df = pd.read_csv(CSV_INPUT, dtype=str)

    # 1) Merges de trip completo
    for trip_id, (wrong, correct) in FULL_TRIP_MERGES.items():
        mask = (df["trip_id"] == trip_id) & (df["identity_id"] == wrong)
        n = mask.sum()
        df.loc[mask, "identity_id"] = correct
        print(f"Trip {trip_id}: {n} filas {wrong} -> {correct}")

    # 2) Patch de trip 1264001776343000 desde 15:14:52 en adelante
    trip_id, from_ts, wrong, correct = SCOPED_PATCH
    mask = (
        (df["trip_id"] == trip_id)
        & (df["timestamp"] >= from_ts)
        & (df["identity_id"] == wrong)
    )
    n = mask.sum()
    df.loc[mask, "identity_id"] = correct
    print(f"Trip {trip_id} (>= {from_ts}): {n} filas {wrong} -> {correct}")

    df.to_csv(CSV_INPUT, index=False)
    print(f"CSV actualizado: {CSV_INPUT}")


if __name__ == "__main__":
    main()
