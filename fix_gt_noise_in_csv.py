"""
CORRECCIÓN DE RUIDO EN EL GROUND TRUTH (identity_id) DEL CSV
=============================================================

Tras revisar manualmente los misses del reporte HTML
(outputs_miss_review/index.html), se confirma cuáles son fallas reales del
algoritmo (si las hay) y cuáles son errores/ruido de etiquetado en
`identity_id` (parpadeos que pasaron el filtro de "sostenido >= 2 frames").

Este script corrige el CSV (el que esté configurado como INPUT_FILE en
main_analisis_completo_v2.py): para cada miss que NO fue confirmado como
real, relabeliza el tramo (run) de la "nueva identidad" con la identidad
previa, fusionando ambos tramos en uno solo (elimina el cambio falso del GT).

Uso:
    python fix_gt_noise_in_csv.py
"""
import os
import shutil

import pandas as pd

from main_analisis_completo_v2 import load_and_process, INPUT_FILE

BACKUP_FILE = os.path.splitext(INPUT_FILE)[0] + ".pre_gt_fix.bak.csv"

# Misses confirmados como fallas reales del algoritmo (1-indexado, mismo orden
# que "Miss #N" en el reporte HTML). Todos los demás misses son ruido de GT.
# trips_data_enriched.csv (12 misses revisados): ninguno es cambio de
# conductor real -> conjunto vacío.
CONFIRMED_REAL_MISSES = set()


def compute_gt_changes_with_run_bounds(out_df):
    """Igual que compute_gt_changes, pero además devuelve, por asset, la lista
    completa de valid_runs (con sus gs_paths) para poder encadenar
    correctamente las fusiones cuando hay varios tramos de ruido seguidos."""
    changes = []
    assets_runs = {}
    for asset_id, group in out_df.groupby('ASSET_ID', sort=False):
        identity = group['IDENTITY_ID'].astype(str).str.strip().tolist()
        gs_paths = group['GS_PATH'].tolist()
        decision = group['DECISION_SISTEMA'].tolist()
        n = len(identity)

        runs = []
        i = 0
        while i < n:
            j = i
            while j + 1 < n and identity[j + 1] == identity[i]:
                j += 1
            runs.append((identity[i], i, j))
            i = j + 1

        valid_runs = [r for r in runs if r[0] not in ('', 'nan') and (r[2] - r[1] + 1) >= 2]
        assets_runs[asset_id] = {'valid_runs': valid_runs, 'gs_paths': gs_paths}

        for k in range(1, len(valid_runs)):
            prev_val, _, prev_end = valid_runs[k - 1]
            cur_val, cur_start, cur_end = valid_runs[k]
            if prev_val != cur_val:
                detected = decision[cur_start] in ('POSIBLE_CAMBIO', 'CAMBIO_CONFIRMADO')
                changes.append({
                    'asset_id': asset_id,
                    'prev_identity': prev_val,
                    'new_identity': cur_val,
                    'cur_start': cur_start,
                    'cur_end': cur_end,
                    'run_index': k,
                    'prev_run_end': prev_end,
                    'detected': detected,
                })
    return changes, assets_runs


def main():
    print("🚀 Corriendo pipeline para recalcular runs de identity_id...")
    out_df = load_and_process(INPUT_FILE)
    if out_df is None:
        print("❌ No se pudo generar el detalle. Abortando.")
        return

    gt_changes, assets_runs = compute_gt_changes_with_run_bounds(out_df)
    misses = [c for c in gt_changes if not c['detected']]
    print(f"ℹ️ Cambios reales (GT): {len(gt_changes)} | Misses: {len(misses)}")

    # Marcar, por (asset_id, run_index), si el miss es ruido a corregir (True)
    # o una falla real confirmada que no se debe tocar (False).
    noise_run_keys = set()
    n_real = 0
    for i, miss in enumerate(misses, 1):
        if i in CONFIRMED_REAL_MISSES:
            n_real += 1
        else:
            noise_run_keys.add((miss['asset_id'], miss['run_index']))
    print(f"🧹 Misses a corregir (ruido de GT): {len(noise_run_keys)}")
    print(f"✅ Misses confirmados como fallas reales (no se tocan): {n_real}")

    # Backup del CSV original (solo si no existe ya uno)
    if not os.path.exists(BACKUP_FILE):
        shutil.copy2(INPUT_FILE, BACKUP_FILE)
        print(f"💾 Backup creado en: {BACKUP_FILE}")
    else:
        print(f"ℹ️ Backup ya existía, no se sobreescribe: {BACKUP_FILE}")

    df = pd.read_csv(INPUT_FILE)

    total_rows_fixed = 0
    total_runs_fixed = 0
    for asset_id, info in assets_runs.items():
        valid_runs = info['valid_runs']
        gs_paths_all = info['gs_paths']
        if not valid_runs:
            continue

        # Recorre los runs en orden, encadenando la identidad "efectiva"
        # vigente. Si un run es ruido a corregir, se relabelea con la
        # identidad efectiva actual (sin actualizarla). Si es una identidad
        # real (detectada o miss confirmado), se actualiza la identidad
        # efectiva vigente.
        effective_identity = valid_runs[0][0]
        for k in range(1, len(valid_runs)):
            cur_val, cur_start, cur_end = valid_runs[k]
            if (asset_id, k) in noise_run_keys:
                gs_paths = set(str(p) for p in gs_paths_all[cur_start:cur_end + 1])
                mask = (df['asset_id'].astype(str) == str(asset_id)) & (df['gs_path'].astype(str).isin(gs_paths))
                n_matched = mask.sum()
                if n_matched != len(gs_paths):
                    print(f"   ⚠️ Asset {asset_id} run {k}: esperaba {len(gs_paths)} filas, encontró {n_matched}.")
                df.loc[mask, 'identity_id'] = effective_identity
                total_rows_fixed += n_matched
                total_runs_fixed += 1
                # effective_identity NO se actualiza: seguimos "siendo" la identidad previa real.
            else:
                effective_identity = cur_val

    df.to_csv(INPUT_FILE, index=False)
    print(f"✅ CSV corregido: {total_rows_fixed} filas relabeleadas en {total_runs_fixed} tramos.")
    print(f"💾 Guardado en: {INPUT_FILE}")


if __name__ == "__main__":
    main()
