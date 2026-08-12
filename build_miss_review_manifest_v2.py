"""Reconstruye el manifiesto de revisión visual para los 5 misses detectados
tras el rerun (derivación fresca de GT por cambio de DRIVER_ID entre filas
consecutivas, sin filtro de ruido).

Genera, para cada caso, una ventana de filas alrededor del timestamp del
supuesto cambio, con su gs_path para poder descargar las imágenes y revisar
visualmente si es un cambio real de conductor o ruido/error de GT.

Salida:
- outputs_csv_comparison/miss_review_manifest_v2.csv
"""

from pathlib import Path

import pandas as pd

CSV_INPUT = Path("random_trips_data_2026_04.csv")
OUT_DIR = Path("outputs_csv_comparison")
OUT_DIR.mkdir(exist_ok=True)
MANIFEST_CSV = OUT_DIR / "miss_review_manifest_v2.csv"

# (trip_id, gt_timestamp) de los 5 misses detectados en la derivación fresca
CASES = [
    ("1264001776343000", "2026-04-16 15:15:52", "miss_01"),
    ("1264001776343000", "2026-04-16 16:09:52", "miss_02"),
    ("1264001776343000", "2026-04-16 17:48:52", "miss_03"),
    ("1648911775324423", "2026-04-04 18:57:46", "miss_04"),
    ("1809001775245550", "2026-04-03 23:45:41", "miss_05"),
]

WINDOW_ROWS_BEFORE = 8
WINDOW_ROWS_AFTER = 8


def main() -> None:
    df = pd.read_csv(CSV_INPUT, dtype=str)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    rows_out = []
    for trip_id, ts_str, case_id in CASES:
        ts = pd.to_datetime(ts_str)
        g = df[df["trip_id"] == trip_id].sort_values("timestamp").reset_index(drop=True)
        idx_matches = g.index[g["timestamp"] == ts]
        if len(idx_matches) == 0:
            print(f"[WARN] {case_id}: no se encontró timestamp {ts_str} en trip {trip_id}")
            continue
        center_idx = idx_matches[0]
        lo = max(0, center_idx - WINDOW_ROWS_BEFORE)
        hi = min(len(g), center_idx + WINDOW_ROWS_AFTER + 1)
        window = g.iloc[lo:hi].copy()
        window["case_id"] = case_id
        window["is_event_row"] = window.index == center_idx
        rows_out.append(window)

    manifest = pd.concat(rows_out, ignore_index=True)
    manifest.to_csv(MANIFEST_CSV, index=False)
    print(f"Manifiesto guardado en {MANIFEST_CSV} ({len(manifest)} filas)")
    print(manifest.groupby("case_id").size())


if __name__ == "__main__":
    main()
