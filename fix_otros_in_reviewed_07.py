"""
CORRECCION: identity_id == "Otros" en all_reviewed_trips_data_2026_07.csv
==========================================================================

Igual criterio ya aplicado antes a random_trips_data_2026_04.csv (ver
README.md, seccion 3): la etiqueta "Otros" en identity_id NO es una
identidad real, es una falla de FQA (el reviewer no pudo asignar una cara
valida). Se debe:
  - poner fqa_valid = False en esas filas;
  - vaciar identity_id (dejarla en blanco), para que NO se cuente como
    cambio de conductor en ningun GT derivado de identity_id.

Este script procesa el CSV linea por linea con el modulo csv (no con
pandas) para no correr NINGUN riesgo de reformateo de floats/embeddings al
reescribir columnas que no se tocan. Crea un backup antes de escribir.

Uso:
    uv run python fix_otros_in_reviewed_07.py
"""
import csv
import os
import shutil

INPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "all_reviewed_trips_data_2026_07.csv")
BACKUP_FILE = os.path.splitext(INPUT_FILE)[0] + ".pre_otros_fix.bak.csv"
TMP_FILE = INPUT_FILE + ".tmp_fix"


def main():
    if not os.path.exists(BACKUP_FILE):
        print(f"Creando backup en {BACKUP_FILE} ...")
        shutil.copy2(INPUT_FILE, BACKUP_FILE)
    else:
        print(f"Backup ya existe, no se sobreescribe: {BACKUP_FILE}")

    n_rows = 0
    n_fixed = 0

    with open(INPUT_FILE, "r", newline="", encoding="utf-8") as fin, \
         open(TMP_FILE, "w", newline="", encoding="utf-8") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)

        header = next(reader)
        writer.writerow(header)
        idx_identity = header.index("identity_id")
        idx_fqa = header.index("fqa_valid")

        for row in reader:
            n_rows += 1
            if row[idx_identity] == "Otros":
                row[idx_identity] = ""
                row[idx_fqa] = "False"
                n_fixed += 1
            writer.writerow(row)

    os.replace(TMP_FILE, INPUT_FILE)
    print(f"Filas totales: {n_rows}")
    print(f"Filas corregidas (Otros -> vacio, fqa_valid=False): {n_fixed}")


if __name__ == "__main__":
    main()
