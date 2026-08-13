"""
Utilidades compartidas del experimento de reduccion de FP estrictos.

No modifica main_analisis_completo_v2.py; solo importa lo que necesita
(load_emb, CONSTANTS, process_asset_group) y replica el mismo preprocesado
(filtrado de calidad + velocidad + skip primer frame) para que el algoritmo
nuevo vea EXACTAMENTE el mismo set de frames que el baseline, y la
comparacion sea justa.
"""
import os
import sys
import json
import math

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_analisis_completo_v2 import load_emb, CONSTANTS, process_asset_group  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_FILES = {
    "random_04": os.path.join(REPO_ROOT, "random_trips_data_2026_04.csv"),
    "reviewed_07": os.path.join(REPO_ROOT, "all_reviewed_trips_data_2026_07.csv"),
}


def cosine_dist(a, b):
    if a is None or b is None:
        return None
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 0:
        return None
    return 1.0 - float(np.dot(a, b) / denom)


def load_raw_csv(path):
    df = pd.read_csv(path, low_memory=False)
    df = df[df['embedding'].notna() & (df['embedding'].astype(str).str.strip() != '')]
    df['image_file'] = df['asset_id'].astype(str) + "_" + df['gs_path'].apply(lambda p: os.path.basename(str(p)))
    return df


def preprocess_trip(df_trip):
    """Replica EXACTA (misma logica, mismo orden) del preprocesado que hace
    process_asset_group antes del loop principal: filtro de calidad, filtro
    de velocidad top-down, orden temporal y skip del primer frame.
    Devuelve df_clean listo para iterar frame a frame (o vacio/None si no
    alcanza min_images_per_asset)."""
    df_clean = df_trip.copy()

    if CONSTANTS['filter_face_small'] and 'face_small' in df_clean.columns:
        df_clean = df_clean[df_clean['face_small'] == False]  # noqa: E712
    if CONSTANTS['filter_ipd_small'] and 'IPD_small' in df_clean.columns:
        df_clean = df_clean[df_clean['IPD_small'] == 0.0]

    img_col = 'image_file'

    def get_ts_from_file(val):
        try:
            parts = str(val).split('_')
            if len(parts) >= 2:
                return int(parts[1]) // 1000
            return np.nan
        except Exception:
            return np.nan

    df_clean['ts_seconds'] = df_clean[img_col].apply(get_ts_from_file)
    df_clean = df_clean.dropna(subset=['ts_seconds'])
    df_clean['ts_seconds'] = df_clean['ts_seconds'].astype(int)
    df_clean = df_clean.sort_values('ts_seconds').reset_index(drop=True)

    speeds = df_clean['speed'].values if 'speed' in df_clean.columns else np.zeros(len(df_clean))
    min_stat = CONSTANTS['min_stationary_speed']
    keep_mask = speeds >= min_stat
    df_clean = df_clean[keep_mask].reset_index(drop=True)

    if len(df_clean) > 1:
        df_clean = df_clean.iloc[1:].reset_index(drop=True)

    if df_clean.empty or len(df_clean) <= CONSTANTS['min_images_per_asset']:
        return None
    return df_clean


def physics_prob(speed_prev, speed_cur, delta_t):
    """Misma formula de probabilidad fisica que el baseline (sigmoide sobre
    tiempo sobrante sobre el tiempo de maniobra requerido)."""
    a_decel = CONSTANTS['a_decel']
    a_accel = CONSTANTS['a_accel']
    k = CONSTANTS['k']
    v_prev = speed_prev / 3.6
    v_curr = speed_cur / 3.6
    t_frenado = v_prev / a_decel
    t_arranque = v_curr / a_accel
    v_max = max(v_prev, v_curr)
    if v_max < 1.0:
        t_maniobra_dinamico = CONSTANTS['t_maniobra_stationary']
    elif v_max < 5.0:
        t_maniobra_dinamico = CONSTANTS['t_maniobra_slow']
    else:
        t_maniobra_dinamico = CONSTANTS['t_maniobra']
    t_req = t_frenado + t_maniobra_dinamico + t_arranque
    t_sobra = delta_t - t_req
    return 1.0 / (1.0 + math.exp(-k * t_sobra))


# ---------------------------------------------------------------------------
# GROUND TRUTH: runs de identity_id, filtrado de glitches de 1 frame y
# deteccion de casos ambiguos / contradichos por embeddings -> GT_SUSPECT.
# ---------------------------------------------------------------------------

def _build_runs(identity_list):
    """Runs de valores consecutivos iguales (incluye vacios/NaN como su
    propio run, para no romper el orden)."""
    runs = []
    n = len(identity_list)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and identity_list[j + 1] == identity_list[i]:
            j += 1
        runs.append((identity_list[i], i, j))
        i = j + 1
    return runs


def build_gt_for_trip(df_clean, embed_col='embedding', identity_col='identity_id'):
    """Construye eventos GT de cambio de conductor para un viaje ya
    preprocesado (mismo orden/filas que vera el detector).

    Reglas:
      - runs de identity_id (ignorando vacios como run propio);
      - runs no vacios de longitud 1 ("glitch" de 1 frame):
          * si el valor previo y siguiente run validos son iguales -> glitch
            puro, se ignora (no es cambio real);
          * si son distintos (A A B C C) -> ambiguo, se marca GT_SUSPECT en
            la transicion resultante;
      - contradiccion fuerte embeddings vs etiqueta: si el evento de cambio
        "limpio" tiene el centroide de los primeros frames de la nueva
        identidad muy parecido (dist < visual_match) al centroide de la
        identidad previa -> GT_SUSPECT (posible error de etiquetado).

    Devuelve lista de dicts: {frame_idx (posicion en df_clean donde arranca
    la nueva identidad), prev_identity, new_identity, run_len_new,
    gt_suspect (bool), reason}.
    """
    identity = df_clean[identity_col].astype(str).where(df_clean[identity_col].notna(), '').tolist()
    identity = ['' if v in ('nan', 'None') else v for v in identity]
    embeddings = [load_emb(e) for e in df_clean[embed_col].tolist()]

    runs = _build_runs(identity)

    # Runs no vacios (candidatos a identidad real)
    nonblank_idx = [k for k, r in enumerate(runs) if r[0] != '']

    events = []
    # Construir secuencia "efectiva" de runs validos (>=2 frames), marcando
    # los runs de 1 frame como glitches (posiblemente ambiguos).
    effective = []  # lista de (identity_value, start, end, suspect_glitch_before)
    k = 0
    while k < len(nonblank_idx):
        ridx = nonblank_idx[k]
        val, start, end = runs[ridx]
        length = end - start + 1
        if length >= 2:
            effective.append({'val': val, 'start': start, 'end': end, 'suspect': False})
            k += 1
        else:
            # Run de 1 frame: glitch candidato. Mirar vecino valido previo y
            # siguiente (si existen) para decidir si es limpio o ambiguo.
            prev_val = effective[-1]['val'] if effective else None
            next_val = None
            if k + 1 < len(nonblank_idx):
                next_val = runs[nonblank_idx[k + 1]][0]
            if prev_val is not None and next_val is not None and prev_val == next_val:
                pass  # glitch puro, se descarta sin dejar rastro
            elif prev_val is None or next_val is None:
                pass  # glitch al borde del viaje, se descarta (poca evidencia)
            else:
                # Ambiguo: A A B C C -> marcar la transicion siguiente (a next_val)
                # como sospechosa.
                effective.append({'val': next_val, 'start': None, 'end': None,
                                   'suspect_pending': True})
                # Se procesara como parte del siguiente run valido (fusion abajo)
            k += 1

    # Fusionar marcadores 'suspect_pending' (del bloque ambiguo) con el
    # siguiente run real del mismo valor.
    merged = []
    pending_suspect = False
    for item in effective:
        if item.get('suspect_pending'):
            pending_suspect = True
            continue
        if pending_suspect:
            item = dict(item)
            item['suspect'] = True
            pending_suspect = False
        merged.append(item)

    # Generar eventos de cambio entre runs efectivos consecutivos de valor distinto
    for i in range(1, len(merged)):
        prev_r = merged[i - 1]
        cur_r = merged[i]
        if prev_r['val'] == cur_r['val']:
            continue
        suspect = bool(cur_r.get('suspect', False))
        reason = 'ambiguous_1frame_glitch' if suspect else 'clean_transition'

        # Contradiccion por embeddings: comparar centroide de los ultimos
        # frames de la identidad previa vs los primeros de la nueva.
        if cur_r['start'] is not None and prev_r['end'] is not None:
            prev_embs = [embeddings[j] for j in range(max(prev_r['start'], prev_r['end'] - 2), prev_r['end'] + 1)
                         if embeddings[j] is not None]
            new_embs = [embeddings[j] for j in range(cur_r['start'], min(cur_r['end'], cur_r['start'] + 2) + 1)
                        if embeddings[j] is not None]
            if prev_embs and new_embs:
                c_prev = np.mean(prev_embs, axis=0)
                c_new = np.mean(new_embs, axis=0)
                d = cosine_dist(c_prev, c_new)
                new_run_len = cur_r['end'] - cur_r['start'] + 1
                if d is not None and d < CONSTANTS['visual_match']:
                    suspect = True
                    reason = 'embeddings_contradict_label(dist=%.3f)' % d
                elif d is not None and new_run_len < 3 and d < 0.55:
                    suspect = True
                    reason = 'weak_persistence_and_similar_embedding(dist=%.3f,len=%d)' % (d, new_run_len)

        events.append({
            'frame_idx': cur_r['start'],
            'prev_identity': prev_r['val'],
            'new_identity': cur_r['val'],
            'new_run_len': (cur_r['end'] - cur_r['start'] + 1) if cur_r['start'] is not None else None,
            'gt_suspect': suspect,
            'reason': reason,
        })

    return events


def match_events_window(gt_frames, alert_frames, window=3):
    """Matching greedy por ventana temporal (en # de frames del viaje ya
    preprocesado, no en filas crudas). Cada GT se casa con como maximo una
    alerta y viceversa, priorizando la distancia de frames mas chica.
    Devuelve (matches, unmatched_gt, unmatched_alert) donde matches es lista
    de (gt_frame, alert_frame, delay)."""
    pairs = []
    for gi, gf in enumerate(gt_frames):
        for ai, af in enumerate(alert_frames):
            d = abs(af - gf)
            if d <= window:
                pairs.append((d, gi, ai))
    pairs.sort(key=lambda x: x[0])
    used_gt, used_alert = set(), set()
    matches = []
    for d, gi, ai in pairs:
        if gi in used_gt or ai in used_alert:
            continue
        used_gt.add(gi)
        used_alert.add(ai)
        matches.append((gt_frames[gi], alert_frames[ai], alert_frames[ai] - gt_frames[gi]))
    unmatched_gt = [gt_frames[i] for i in range(len(gt_frames)) if i not in used_gt]
    unmatched_alert = [alert_frames[i] for i in range(len(alert_frames)) if i not in used_alert]
    return matches, unmatched_gt, unmatched_alert
