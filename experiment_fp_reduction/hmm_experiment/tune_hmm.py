"""
Ajuste rapido de pocos parametros del HMM SOLO en DEV (TEST no se toca hasta
congelar). NO es un grid gigante: variantes de transicion x variantes de
emision x valores de threshold_confirm, cada combinacion corrida sobre el
split DEV completo (los 2 CSV, restringidos a trip_id en DEV via
v2_dev_test/splits.csv, que NO se modifica).

Restriccion dura: recall_ANY >= MIN_RECALL_ANY (idealmente 1.00).
Restriccion blanda: evitar la "solucion tonta" de no confirmar casi nada
(se exige un minimo de TP_confirmado).
Objetivo: maximizar precision de CAMBIO_CONFIRMADO.

Uso:
    uv run python tune_hmm.py

Guarda grid_results_DEV.csv (todas las combinaciones) y hmm_params.json
(la combinacion elegida, "congelada").
"""
import os
import sys
import json
import multiprocessing as mp

import numpy as np
import pandas as pd
from rich.progress import (
    Progress, BarColumn, TextColumn, MofNCompleteColumn,
    TimeElapsedColumn, TimeRemainingColumn,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import CSV_FILES, load_raw_csv, preprocess_trip, load_emb  # noqa: E402
from evaluate import evaluate  # noqa: E402

from hmm_model import CausalHMM3, DEFAULT_EMISSION  # noqa: E402
from feature_extractor import run_memory_hmm  # noqa: E402

HMM_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(HMM_DIR)
HMM_DATA_DIR = os.path.join(HMM_DIR, "data")
os.makedirs(HMM_DATA_DIR, exist_ok=True)

SPLITS_PATH = os.path.join(EXP_DIR, "v2_dev_test", "splits.csv")
GT_DIR = os.path.join(EXP_DIR, "data")

MIN_RECALL_ANY = 0.98
MIN_TP_CONFIRMADO = 5  # evita la "solucion tonta" de casi nunca confirmar

TRANSITION_VARIANTS = {
    "base": np.array([
        [0.95, 0.05, 0.00],
        [0.15, 0.65, 0.20],
        [0.02, 0.03, 0.95],
    ]),
    "mas_facil_confirmar": np.array([
        [0.95, 0.05, 0.00],
        [0.10, 0.55, 0.35],
        [0.02, 0.03, 0.95],
    ]),
}

# Variante "conservadora": actualizaciones mas debiles por frame (en
# particular ANOMALY_STRONG ya no es casi-certeza de CAMBIO_REAL) para que
# haga falta MAS persistencia (mas frames seguidos de evidencia) antes de
# cruzar threshold_confirm. Esto ataca directamente el patron ya conocido
# (ver memoria del repo) de que muchos FP son anomalias transitorias de
# 2 frames que "vuelven" al conductor anterior poco despues: si confirmamos
# mas lento, les da tiempo a la senal RETURN a bajar la creencia ANTES de
# que se cruce el umbral.
EMISSION_VARIANTS = {
    "default": DEFAULT_EMISSION,
}

THRESHOLD_CONFIRM_VALUES = [0.35, 0.40, 0.45, 0.50, 0.55, 0.65]


def load_dev_trips():
    splits = pd.read_csv(SPLITS_PATH)
    return set(splits.loc[splits['split'] == 'DEV', 'trip_id'].tolist())


def load_dev_data():
    """Precarga y preprocesa SOLO los viajes DEV (una vez), para no repetir
    la parte cara (parseo de embeddings JSON) en cada punto del grid. Los
    embeddings se parsean UNA sola vez aca (load_emb hace json.loads) y se
    guardan ya como arrays de numpy en la columna 'embedding'; asi los N
    combos del grid reusan los mismos arrays en vez de volver a parsear el
    string JSON por cada combinacion x frame (esto era el cuello de botella
    real: sin esto, cada combo re-parseaba TODOS los embeddings de TODOS los
    viajes DEV desde cero). TEST no se toca en ningun momento de esta funcion."""
    dev_trips = load_dev_trips()
    per_dataset = {}
    for name, path in CSV_FILES.items():
        df = load_raw_csv(path)
        trips = [t for t in df['trip_id'].unique() if t in dev_trips]
        clean_trips = {}
        for t in trips:
            df_clean = preprocess_trip(df[df['trip_id'] == t])
            if df_clean is not None:
                df_clean = df_clean.copy()
                df_clean['embedding'] = df_clean['embedding'].apply(load_emb)
                clean_trips[t] = df_clean
        per_dataset[name] = clean_trips

        gt = pd.read_csv(os.path.join(GT_DIR, f"gt_events_{name}.csv"))
        gt['gt_suspect'] = gt['gt_suspect'].astype(bool)
        per_dataset[name + "_gt"] = gt[gt['trip_id'].isin(dev_trips)]
    return per_dataset


# Global poblado en el proceso padre ANTES de crear el Pool. Con
# multiprocessing start method 'fork' (default en Linux) los procesos hijo
# heredan la memoria por copy-on-write: ningun DataFrame/embedding se
# vuelve a serializar/pickelar por tarea, solo se copian las paginas de
# memoria que un hijo llega a modificar (aca ninguna, son solo lecturas).
_GLOBAL_DEV_DATA = None


def _run_one_trip(args):
    """Corre UN viaje con UNA combinacion de parametros. Unidad de trabajo
    chica a proposito (viaje x combo) para repartir parejo entre todos los
    cores disponibles, en vez de 1 tarea = 1 combo completo (que dejaria
    nucleos ociosos si hay menos combos que cores)."""
    name, t, transition, emission, threshold_confirm = args
    df_clean = _GLOBAL_DEV_DATA[name][t]
    hmm = CausalHMM3(transition=transition, emission=emission)
    recs = run_memory_hmm(df_clean, hmm, threshold_confirm)
    for r in recs:
        r['ASSET_ID'] = t
    return recs


def run_all_combos(dev_data, combos, combo_meta, pool, checkpoint_path):
    """Evalua todas las combinaciones del grid en paralelo (nucleos =
    min(N_workers, cpu_count)), repartiendo el trabajo a nivel (viaje, combo)
    con imap (chunksize chico) para balancear carga (viajes largos vs
    cortos) y mostrar progreso real con rich.

    Resumible: a medida que CADA combinacion termina (todos sus viajes
    procesados) se evalua y se agrega de inmediato a checkpoint_path (CSV,
    modo append). Si el proceso se corta a la mitad, correr de nuevo salta
    las combinaciones ya escritas ahi (ver main())."""
    trip_keys = [(name, t) for name in CSV_FILES for t in dev_data[name]]
    n_trips = len(trip_keys)

    tasks = []
    task_combo_idx = []
    for ci, (transition, emission, threshold_confirm) in enumerate(combos):
        for name, t in trip_keys:
            tasks.append((name, t, transition, emission, threshold_confirm))
            task_combo_idx.append(ci)

    detail_rows = {ci: [] for ci in range(len(combos))}
    pending_count = {ci: n_trips for ci in range(len(combos))}
    gt_all = pd.concat([dev_data[name + "_gt"] for name in CSV_FILES], ignore_index=True)

    results = {}
    chunksize = max(1, len(tasks) // (pool._processes * 8))

    header_written = os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("tareas (viaje x combo)"),
        TimeElapsedColumn(),
        TextColumn("restante:"),
        TimeRemainingColumn(),
    ) as progress:
        task_bar = progress.add_task(f"Combos 0/{len(combos)}", total=len(tasks))
        n_combos_done = 0
        for idx, recs in zip(task_combo_idx, pool.imap(_run_one_trip, tasks, chunksize=chunksize)):
            detail_rows[idx].extend(recs)
            pending_count[idx] -= 1
            progress.advance(task_bar)

            if pending_count[idx] == 0:
                detail = pd.DataFrame(detail_rows[idx])
                m, _tp, _fp = evaluate(detail, gt_all, label="hmm_DEV")
                results[idx] = m
                detail_rows[idx] = None  # liberar memoria apenas se puede

                tname, ename, thr_conf = combo_meta[idx]
                row = {
                    'transition': tname,
                    'emission': ename,
                    'threshold_confirm': thr_conf,
                    'tp_confirmado': m['cambio_confirmado_correctos'],
                    'fp_confirmado': m['cambio_confirmado_incorrectos_fp_estrictos'],
                    'precision_confirmado': m['precision_estricta_cambio_confirmado'],
                    'recall_any': m['recall_sin_suspect_cualquier_alerta'],
                    'posible_cambio_total': m['posible_cambio_total'],
                    'delay_medio': m['delay_medio_frames_confirmado'],
                    'delay_max': m['delay_max_frames_confirmado'],
                }
                pd.DataFrame([row]).to_csv(checkpoint_path, mode='a', index=False, header=not header_written)
                header_written = True

                n_combos_done += 1
                progress.update(task_bar, description=f"Combos {n_combos_done}/{len(combos)}")
                progress.console.print(row)

    return results


def main():
    global _GLOBAL_DEV_DATA
    print("Cargando y preprocesando viajes DEV (una sola vez)...")
    dev_data = load_dev_data()
    _GLOBAL_DEV_DATA = dev_data
    n_trips = sum(len(dev_data[name]) for name in CSV_FILES)
    print(f"  {n_trips} viajes DEV listos (TEST no se toca)")

    combo_meta_all = []  # (tname, ename, thr_conf)
    combos_all = []  # (transition, emission, threshold_confirm) - lo que ve el worker
    for tname, transition in TRANSITION_VARIANTS.items():
        for ename, emission in EMISSION_VARIANTS.items():
            for thr_conf in THRESHOLD_CONFIRM_VALUES:
                combo_meta_all.append((tname, ename, thr_conf))
                combos_all.append((transition, emission, thr_conf))

    checkpoint_path = os.path.join(HMM_DATA_DIR, "grid_results_DEV.partial.csv")
    done_rows = []
    done_keys = set()
    if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 0:
        prev = pd.read_csv(checkpoint_path)
        # de-duplicar por si una corrida anterior se corto justo despues de
        # escribir pero antes de que Python terminara de flushear (append
        # puede dejar una fila repetida en un corte muy puntual).
        prev = prev.drop_duplicates(subset=['transition', 'emission', 'threshold_confirm'], keep='last')
        done_rows = prev.to_dict('records')
        done_keys = {(r['transition'], r['emission'], float(r['threshold_confirm'])) for r in done_rows}
        print(f"Checkpoint encontrado: {len(done_keys)}/{len(combo_meta_all)} combinaciones ya resueltas "
              f"en {checkpoint_path} (se saltan, resume desde ahi).")

    pending = [(cm, cb) for cm, cb in zip(combo_meta_all, combos_all)
               if (cm[0], cm[1], float(cm[2])) not in done_keys]
    combo_meta = [p[0] for p in pending]
    combos = [p[1] for p in pending]

    results = {}
    if combos:
        n_workers = min(len(os.sched_getaffinity(0)), 20)
        print(f"Corriendo {len(combos)} combinaciones pendientes x {n_trips} viajes "
              f"({len(combos) * n_trips} tareas) en paralelo con {n_workers} procesos...")

        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_workers) as pool:
            results = run_all_combos(dev_data, combos, combo_meta, pool, checkpoint_path)
    else:
        print("Todas las combinaciones ya estaban resueltas en el checkpoint, no hace falta correr nada.")

    results_list = [results[i] for i in range(len(combos))] if combos else []
    new_rows = []
    for (tname, ename, thr_conf), m in zip(combo_meta, results_list):
        row = {
            'transition': tname,
            'emission': ename,
            'threshold_confirm': thr_conf,
            'tp_confirmado': m['cambio_confirmado_correctos'],
            'fp_confirmado': m['cambio_confirmado_incorrectos_fp_estrictos'],
            'precision_confirmado': m['precision_estricta_cambio_confirmado'],
            'recall_any': m['recall_sin_suspect_cualquier_alerta'],
            'posible_cambio_total': m['posible_cambio_total'],
            'delay_medio': m['delay_medio_frames_confirmado'],
            'delay_max': m['delay_max_frames_confirmado'],
        }
        new_rows.append(row)

    results = done_rows + new_rows

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(HMM_DATA_DIR, "grid_results_DEV.csv"), index=False)

    valid = res_df[(res_df['recall_any'] >= MIN_RECALL_ANY) & (res_df['tp_confirmado'] >= MIN_TP_CONFIRMADO)]
    if valid.empty:
        print(f"\nNingun combo cumple recall_any>={MIN_RECALL_ANY} y tp_confirmado>={MIN_TP_CONFIRMADO}; "
              "se relaja el minimo de TP para no sacrificar recall.")
        valid = res_df[res_df['recall_any'] >= MIN_RECALL_ANY]
    if valid.empty:
        print(f"\nNingun combo cumple recall_any>={MIN_RECALL_ANY} en absoluto; se usa el de mayor recall_any.")
        valid = res_df

    best = valid.sort_values(['precision_confirmado', 'tp_confirmado'], ascending=[False, False]).iloc[0]
    print(f"\nMEJOR COMBO (DEV): {best.to_dict()}")

    best_transition = TRANSITION_VARIANTS[best['transition']].tolist()
    best_emission = {k: (v.tolist() if isinstance(v, np.ndarray) else list(v))
                      for k, v in EMISSION_VARIANTS[best['emission']].items()}
    params = {
        'transition_variant': best['transition'],
        'transition': best_transition,
        'emission_variant': best['emission'],
        'emission': best_emission,
        'threshold_confirm': float(best['threshold_confirm']),
        'dev_metrics': {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                        for k, v in best.to_dict().items()},
    }
    with open(os.path.join(HMM_DIR, "hmm_params.json"), "w") as f:
        json.dump(params, f, indent=2)
    print("\nParametros congelados guardados en hmm_params.json")


if __name__ == "__main__":
    main()
