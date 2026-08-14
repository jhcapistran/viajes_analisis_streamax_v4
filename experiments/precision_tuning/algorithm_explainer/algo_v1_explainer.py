"""
COPIA EXACTA de la logica de decision de algo_v1_param.py / algo_v1_instrumented.py
(config congelada "3-de-4", E_w4_s3_c035_d085_p06). No se cambia NINGUN
umbral ni condicion de decision. Unicamente se agregan campos de
diagnostico de SOLO LECTURA para poder visualizar, frame a frame, el
estado interno completo del algoritmo (memoria, ventana candidata,
coherencia, fisica) en el explicador HTML. No se usa para evaluar metricas
ni para recalibrar nada - solo para el HTML explicativo.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                 "experiments", "prototype_memory"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prototype_memory"))
from common import cosine_dist, physics_prob, CONSTANTS  # noqa: E402

VISUAL_MATCH = CONSTANTS['visual_match']  # 0.5, fijo


def physics_breakdown(speed_prev, speed_cur, delta_t):
    """Recalcula, de solo lectura, los terminos intermedios de la MISMA
    formula que usa common.physics_prob (no se toca esa funcion, se llama
    aparte solo para poder mostrar t_frenado/t_arranque/t_maniobra/t_req/
    t_sobra en el HTML explicativo). Incluye tambien las variables de
    entrada (v_prev, v_curr, a_decel, a_accel) para poder mostrar la
    sustitucion variable-por-variable, no solo el resultado final."""
    a_decel = CONSTANTS['a_decel']
    a_accel = CONSTANTS['a_accel']
    v_prev = speed_prev / 3.6
    v_curr = speed_cur / 3.6
    t_frenado = v_prev / a_decel
    t_arranque = v_curr / a_accel
    v_max = max(v_prev, v_curr)
    if v_max < 1.0:
        t_maniobra = CONSTANTS['t_maniobra_stationary']
        t_maniobra_reason = 'v_max < 1.0 m/s (vehiculo detenido)'
    elif v_max < 5.0:
        t_maniobra = CONSTANTS['t_maniobra_slow']
        t_maniobra_reason = 'v_max < 5.0 m/s (vehiculo lento)'
    else:
        t_maniobra = CONSTANTS['t_maniobra']
        t_maniobra_reason = 'v_max >= 5.0 m/s (velocidad normal)'
    t_req = t_frenado + t_maniobra + t_arranque
    t_sobra = delta_t - t_req
    return {
        't_frenado': t_frenado,
        't_arranque': t_arranque,
        't_maniobra': t_maniobra,
        't_req': t_req,
        't_sobra': t_sobra,
        'v_prev': v_prev,
        'v_curr': v_curr,
        'a_decel': a_decel,
        'a_accel': a_accel,
        'v_max': v_max,
        't_maniobra_reason': t_maniobra_reason,
    }


def cosine_breakdown(a, b):
    """Descompone la distancia coseno en sus terminos intermedios (producto
    punto y normas) para poder mostrar la sustitucion variable-por-variable
    en el HTML explicativo. No cambia el calculo de common.cosine_dist, solo
    lo recalcula aparte con fines de visualizacion."""
    if a is None or b is None:
        return None
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    denom = norm_a * norm_b
    if denom <= 0:
        return None
    dot = float(np.dot(a, b))
    cos_sim = dot / denom
    return {
        'dot': dot,
        'norm_a': norm_a,
        'norm_b': norm_b,
        'cos_sim': cos_sim,
        'dist': 1.0 - cos_sim,
    }

FROZEN_PARAMS = dict(
    memory_size=6,
    candidate_window=4,
    candidate_min_support=3,
    coherence_threshold=0.35,
    min_avg_dist_confirm=0.85,
    min_phys_confirm=0.6,
)

_ID_COUNTER = [0]


def _new_id():
    _ID_COUNTER[0] += 1
    return f"MP{_ID_COUNTER[0]:07d}"


def run_v1_explainer(frames, params=None, trip_id=None):
    """Devuelve una lista de records (uno por frame) con TODO el estado
    interno relevante para explicar la decision, ademas de la decision
    misma. La logica de control (que decide MISMO/POSIBLE/CONFIRMADO) es
    identica, caracter por caracter, a algo_v1_instrumented.py."""
    p = dict(FROZEN_PARAMS)
    if params:
        p.update(params)

    n = len(frames)
    records = []
    if n == 0:
        return records

    first = frames[0]
    current_id = _new_id()
    # memory guarda embeddings; memory_idx guarda el indice de frame (en
    # esta lista `frames`) de cada embedding en memoria, para poder mostrar
    # miniaturas reales.
    memory = [first['emb']] if first['emb'] is not None else []
    memory_idx = [0] if first['emb'] is not None else []
    id_pool = {current_id: list(memory)}

    records.append({
        'frame_idx': 0,
        'PERSONA_ID': current_id,
        'DECISION_SISTEMA': 'INICIO_VIAJE',
        'DELAY_CONFIRMACION': 0,
        'DIST_MEM': None,
        'DIST_MEM_BREAKDOWN': None,
        'PREV_FRAME_DIST': None,
        'PREV_FRAME_DIST_BREAKDOWN': None,
        'P_FISICA': None,
        'PHYSICS_BREAKDOWN': None,
        'SPEED': first.get('speed'),
        'TS_SECONDS': first.get('ts_seconds'),
        'DELTA_T': None,
        'MEMORY_SIZE': len(memory),
        'MEMORY_FRAME_IDX': list(memory_idx),
        'PENDING_FRAMES': None,
        'PENDING_DISTS': None,
        'COHERENT_LOCAL': None,
        'SUPPORT': None,
        'AVG_DIST_VS_OLD': None,
        'CANDIDATE_FRAMES': None,
        'COHERENT_FRAMES': None,
        'CONFIRM_CHECKS': None,
    })

    pending = None
    pending_frame_idx = None  # indices de frame para cada elemento de pending['embs']
    prev = first

    for i in range(1, n):
        cur = frames[i]
        emb_cur = cur['emb']
        speed_prev = prev['speed']
        speed_cur = cur['speed']
        delta_t = cur['ts_seconds'] - prev['ts_seconds']
        p_fisica = physics_prob(speed_prev, speed_cur, delta_t)
        phys_breakdown = physics_breakdown(speed_prev, speed_cur, delta_t)
        prev_frame_dist = cosine_dist(prev['emb'], emb_cur) if prev.get('emb') is not None else None
        prev_frame_dist_breakdown = cosine_breakdown(prev['emb'], emb_cur) if prev.get('emb') is not None else None

        decision = 'MISMO_CONDUCTOR'
        delay_confirm = 0
        dist_mem_out = None
        dist_mem_breakdown_out = None
        candidate_frames_out = None
        coherent_frames_out = None
        pending_frames_out = None
        pending_dists_out = None
        coherent_local_out = None
        support_out = None
        avg_dist_out = None
        confirm_checks_out = None

        if emb_cur is None:
            decision = 'INDETERMINADO'
        else:
            mem_centroid = np.mean(memory, axis=0) if memory else None
            dist_mem = cosine_dist(mem_centroid, emb_cur) if mem_centroid is not None else None
            dist_mem_out = dist_mem
            dist_mem_breakdown_out = cosine_breakdown(mem_centroid, emb_cur) if mem_centroid is not None else None

            if dist_mem is not None and dist_mem < VISUAL_MATCH:
                decision = 'MISMO_CONDUCTOR'
                pending = None
                pending_frame_idx = None
                memory.append(emb_cur)
                memory_idx.append(i)
                if len(memory) > p['memory_size']:
                    memory.pop(0)
                    memory_idx.pop(0)
                id_pool.setdefault(current_id, []).append(emb_cur)
            elif dist_mem is not None and p_fisica <= CONSTANTS['phys_impossible']:
                decision = 'MISMO_CONDUCTOR'
                memory.append(emb_cur)
                memory_idx.append(i)
                if len(memory) > p['memory_size']:
                    memory.pop(0)
                    memory_idx.pop(0)
                id_pool.setdefault(current_id, []).append(emb_cur)
            else:
                if pending is None:
                    pending = {'embs': [emb_cur], 'frames': [i], 'dists': [dist_mem]}
                    pending_frame_idx = [i]
                else:
                    pending['embs'].append(emb_cur)
                    pending['frames'].append(i)
                    pending['dists'].append(dist_mem)
                    pending_frame_idx.append(i)
                    if len(pending['embs']) > p['candidate_window']:
                        pending['embs'].pop(0)
                        pending['frames'].pop(0)
                        pending['dists'].pop(0)
                        pending_frame_idx.pop(0)

                embs = pending['embs']
                coherent_group = []
                for a in range(len(embs)):
                    cluster = [a]
                    for b in range(len(embs)):
                        if a == b:
                            continue
                        dd = cosine_dist(embs[a], embs[b])
                        if dd is not None and dd < p['coherence_threshold']:
                            cluster.append(b)
                    if len(cluster) > len(coherent_group):
                        coherent_group = cluster

                avg_dist_vs_old = float(np.mean([pending['dists'][k] for k in coherent_group])) if coherent_group else 0.0

                support_out = len(coherent_group)
                avg_dist_out = avg_dist_vs_old
                pending_frames_out = list(pending['frames'])
                pending_dists_out = [round(float(d), 4) if d is not None else None for d in pending['dists']]
                coherent_local_out = list(coherent_group)
                confirm_checks_out = {
                    'support_ok': support_out >= p['candidate_min_support'],
                    'avg_dist_ok': avg_dist_vs_old >= p['min_avg_dist_confirm'],
                    'phys_ok': p_fisica >= p['min_phys_confirm'],
                }

                if (len(coherent_group) >= p['candidate_min_support']
                        and avg_dist_vs_old >= p['min_avg_dist_confirm']
                        and p_fisica >= p['min_phys_confirm']):
                    cand_embs = [embs[k] for k in coherent_group]
                    cand_frame_idx = [pending['frames'][k] for k in coherent_group]
                    cand_centroid = np.mean(cand_embs, axis=0)

                    reused_id = None
                    best_frames = 0
                    best_dist = float('inf')
                    for pid, pool in id_pool.items():
                        if pid == current_id or not pool:
                            continue
                        centroid = np.mean(pool, axis=0)
                        dd = cosine_dist(centroid, cand_centroid)
                        if dd is not None and dd < CONSTANTS['id_reuse_match_threshold']:
                            if len(pool) > best_frames or (len(pool) == best_frames and dd < best_dist):
                                best_frames = len(pool)
                                best_dist = dd
                                reused_id = pid

                    current_id = reused_id if reused_id is not None else _new_id()
                    memory = list(cand_embs)
                    memory_idx = list(cand_frame_idx)
                    id_pool.setdefault(current_id, [])
                    id_pool[current_id].extend(cand_embs)

                    decision = 'CAMBIO_CONFIRMADO'
                    start_frame = pending['frames'][coherent_group[0]]
                    delay_confirm = i - start_frame
                    candidate_frames_out = list(pending['frames'])
                    coherent_frames_out = [pending['frames'][k] for k in coherent_group]
                    pending = None
                    pending_frame_idx = None
                else:
                    decision = 'POSIBLE_CAMBIO'

        records.append({
            'frame_idx': i,
            'PERSONA_ID': current_id,
            'DECISION_SISTEMA': decision,
            'DELAY_CONFIRMACION': delay_confirm,
            'DIST_MEM': dist_mem_out,
            'DIST_MEM_BREAKDOWN': dist_mem_breakdown_out,
            'PREV_FRAME_DIST': prev_frame_dist,
            'PREV_FRAME_DIST_BREAKDOWN': prev_frame_dist_breakdown,
            'P_FISICA': p_fisica,
            'PHYSICS_BREAKDOWN': phys_breakdown,
            'SPEED': cur.get('speed'),
            'TS_SECONDS': cur.get('ts_seconds'),
            'DELTA_T': delta_t,
            'MEMORY_SIZE': len(memory),
            'MEMORY_FRAME_IDX': list(memory_idx),
            'PENDING_FRAMES': pending_frames_out,
            'PENDING_DISTS': pending_dists_out,
            'COHERENT_LOCAL': coherent_local_out,
            'SUPPORT': support_out,
            'AVG_DIST_VS_OLD': avg_dist_out,
            'CANDIDATE_FRAMES': candidate_frames_out,
            'COHERENT_FRAMES': coherent_frames_out,
            'CONFIRM_CHECKS': confirm_checks_out,
        })
        prev = cur

    return records
