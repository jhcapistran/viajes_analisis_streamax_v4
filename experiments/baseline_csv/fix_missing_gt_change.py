"""
CORRECCIÓN PUNTUAL: agrega un cambio real de conductor que el GT (identity_id)
NO capturó (el algoritmo sí lo detectó como CAMBIO_CONFIRMADO, correctamente,
pero el identity_id original conservó la misma etiqueta para ambas personas).

Caso confirmado manualmente por el usuario:
- Asset 105267: desde el frame #83 en adelante es una persona distinta a la de
  antes, pero ambas estaban etiquetadas 'Identity_2'. Se relabelea el tramo
  posterior con una nueva identidad ('Identity_4', libre en este asset).

(Asset 121138 FP#14 se revisó y NO requiere corrección: el cambio real ya
está bien capturado en el GT en el frame #71 -> Identity_1; el FP en el
frame #70 es solo una alerta duplicada un frame antes del cambio real.)

Uso:
    python fix_missing_gt_change.py
"""
import os
import shutil

import pandas as pd

from main_analisis_completo_v2 import load_and_process, INPUT_FILE

BACKUP_FILE = os.path.splitext(INPUT_FILE)[0] + ".pre_missing_gt_fix.bak.csv"

ASSET_ID = "105267"
SPLIT_FROM_POSITION = 83   # inclusive, posición dentro del grupo del asset
NEW_IDENTITY_LABEL = "Identity_4"


def main():
    print("🚀 Corriendo pipeline para ubicar los frames a corregir...")
    out_df = load_and_process(INPUT_FILE)
    if out_df is None:
        print("❌ No se pudo generar el detalle. Abortando.")
        return

    group = out_df[out_df['ASSET_ID'] == ASSET_ID].reset_index(drop=True)
    if group.empty:
        print(f"❌ No se encontró el asset {ASSET_ID}.")
        return

    n = len(group)
    gs_paths = set(str(p) for p in group['GS_PATH'].tolist()[SPLIT_FROM_POSITION:n])
    print(f"🧹 Asset {ASSET_ID}: relabeleando {len(gs_paths)} filas (frames #{SPLIT_FROM_POSITION}-#{n - 1}) a '{NEW_IDENTITY_LABEL}'.")

    if not os.path.exists(BACKUP_FILE):
        shutil.copy2(INPUT_FILE, BACKUP_FILE)
        print(f"💾 Backup creado en: {BACKUP_FILE}")
    else:
        print(f"ℹ️ Backup ya existía, no se sobreescribe: {BACKUP_FILE}")

    df = pd.read_csv(INPUT_FILE)
    mask = (df['asset_id'].astype(str) == ASSET_ID) & (df['gs_path'].astype(str).isin(gs_paths))
    n_matched = mask.sum()
    if n_matched != len(gs_paths):
        print(f"   ⚠️ Esperaba {len(gs_paths)} filas, encontró {n_matched}.")
    df.loc[mask, 'identity_id'] = NEW_IDENTITY_LABEL

    df.to_csv(INPUT_FILE, index=False)
    print(f"✅ CSV corregido: {n_matched} filas relabeleadas.")
    print(f"💾 Guardado en: {INPUT_FILE}")


if __name__ == "__main__":
    main()
