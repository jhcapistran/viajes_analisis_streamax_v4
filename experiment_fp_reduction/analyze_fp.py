"""
Diagnostico TP vs FP estrictos de CAMBIO_CONFIRMADO del baseline.

Para cada evento CAMBIO_CONFIRMADO (matched=TP o unmatched=FP estricto)
calcula, usando solo informacion disponible hasta ese frame (nada de
futuro salvo la propia persistencia posterior que se usa unicamente para
DESCRIBIR el patron, no para decidir):
  - dist_vs_frame_anterior          (ya en DISTANCIA_VISUAL)
  - dist_vs_centroide_conductor_vigente  (memoria historica del conductor
    que estaba activo antes del evento)
  - persistencia_nueva_identidad    (cuantos frames seguidos mantiene el
    PERSONA_ID nuevo asignado por el baseline)
  - vuelve_al_conductor_anterior    (si el PERSONA_ID previo reaparece en
    los siguientes 10 frames del viaje)
  - coherencia_candidatos_nuevos    (distancia promedio entre los primeros
    frames de la nueva identidad, si hay >= 2)
  - velocidad, empty_cabin, ignition del frame del evento
"""
import os
import numpy as np
import pandas as pd

from common import CSV_FILES, EXP_DIR, load_raw_csv, preprocess_trip, cosine_dist, load_emb
from evaluate import evaluate, _add_frame_idx

PERSIST_LOOKAHEAD = 10


def diagnose_dataset(name, path):
    detail = pd.read_csv(os.path.join(EXP_DIR, f"baseline_detail_{name}.csv"))
    gt = pd.read_csv(os.path.join(EXP_DIR, f"gt_events_{name}.csv"))
    gt['gt_suspect'] = gt['gt_suspect'].astype(bool)
    metrics, tp_df, fp_df = evaluate(detail, gt, label=f"baseline_{name}")

    detail = _add_frame_idx(detail)
    raw = load_raw_csv(path)

    tp_frames = {(r.trip_id, r.alert_frame) for r in tp_df.itertuples()} if len(tp_df) else set()
    fp_frames = {(r.trip_id, r.alert_frame) for r in fp_df.itertuples()} if len(fp_df) else set()
    all_events = [(t, f, True) for (t, f) in tp_frames] + [(t, f, False) for (t, f) in fp_frames]

    rows = []
    cache_clean = {}
    for trip, frame_idx, is_tp in all_events:
        if trip not in cache_clean:
            df_trip = raw[raw['trip_id'] == trip]
            cache_clean[trip] = preprocess_trip(df_trip)
        df_clean = cache_clean[trip]
        if df_clean is None or frame_idx >= len(df_clean):
            continue
        d_trip = detail[detail['ASSET_ID'] == trip].reset_index(drop=True)
        if frame_idx >= len(d_trip):
            continue

        emb_cur = load_emb(df_clean.iloc[frame_idx].get('embedding'))
        cur_person = d_trip.iloc[frame_idx]['PERSONA_ID']
        prev_person = d_trip.iloc[frame_idx - 1]['PERSONA_ID'] if frame_idx > 0 else None

        # Memoria del conductor vigente ANTES del evento: todas las
        # embeddings previas con ese PERSONA_ID (excluye el frame actual).
        prev_mask = (d_trip.index < frame_idx) & (d_trip['PERSONA_ID'] == prev_person)
        prev_idxs = d_trip.index[prev_mask].tolist()
        prev_embs = [load_emb(df_clean.iloc[i].get('embedding')) for i in prev_idxs]
        prev_embs = [e for e in prev_embs if e is not None]
        dist_vs_memoria = cosine_dist(np.mean(prev_embs, axis=0), emb_cur) if prev_embs and emb_cur is not None else None

        # Persistencia de la nueva identidad asignada por el baseline
        persist = 0
        j = frame_idx
        while j < len(d_trip) and d_trip.iloc[j]['PERSONA_ID'] == cur_person:
            persist += 1
            j += 1

        # Vuelve al conductor anterior en los siguientes N frames
        lookahead_end = min(len(d_trip), frame_idx + 1 + PERSIST_LOOKAHEAD)
        vuelve_atras = bool((d_trip.iloc[frame_idx + 1:lookahead_end]['PERSONA_ID'] == prev_person).any()) if prev_person is not None else False

        # Coherencia de los primeros frames de la nueva identidad
        new_idxs = [k for k in range(frame_idx, min(len(d_trip), frame_idx + 3)) if d_trip.iloc[k]['PERSONA_ID'] == cur_person]
        new_embs = [load_emb(df_clean.iloc[k].get('embedding')) for k in new_idxs]
        new_embs = [e for e in new_embs if e is not None]
        coherencia = None
        if len(new_embs) >= 2:
            dists = []
            for a in range(len(new_embs)):
                for b in range(a + 1, len(new_embs)):
                    dd = cosine_dist(new_embs[a], new_embs[b])
                    if dd is not None:
                        dists.append(dd)
            coherencia = float(np.mean(dists)) if dists else None

        row = d_trip.iloc[frame_idx]
        rows.append({
            'trip_id': trip,
            'frame_idx': frame_idx,
            'es_tp': is_tp,
            'dist_vs_frame_anterior': row['DISTANCIA_VISUAL'],
            'dist_vs_memoria_conductor_vigente': dist_vs_memoria,
            'prob_fisica': row['PROBABILIDAD_FISICA'],
            'velocidad': row['VELOCIDAD'],
            'empty_cabin': row['EMPTY_CABIN'],
            'persistencia_nueva_identidad': persist,
            'vuelve_al_conductor_anterior_10f': vuelve_atras,
            'coherencia_candidatos_nuevos': coherencia,
        })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(EXP_DIR, f"fp_diagnostics_{name}.csv"), index=False)

    summary = out.groupby('es_tp').agg(
        n=('frame_idx', 'count'),
        dist_frame_ant_mean=('dist_vs_frame_anterior', 'mean'),
        dist_memoria_mean=('dist_vs_memoria_conductor_vigente', 'mean'),
        persistencia_mean=('persistencia_nueva_identidad', 'mean'),
        persistencia_mediana=('persistencia_nueva_identidad', 'median'),
        pct_persist_1=('persistencia_nueva_identidad', lambda s: (s == 1).mean()),
        pct_vuelve_atras=('vuelve_al_conductor_anterior_10f', 'mean'),
        coherencia_mean=('coherencia_candidatos_nuevos', 'mean'),
        prob_fisica_mean=('prob_fisica', 'mean'),
        velocidad_mean=('velocidad', 'mean'),
    ).reset_index()
    print(f"\n=== {name}: TP (True) vs FP (False) ===")
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(EXP_DIR, f"fp_diagnostics_summary_{name}.csv"), index=False)
    return out, summary


if __name__ == "__main__":
    for name, path in CSV_FILES.items():
        diagnose_dataset(name, path)
