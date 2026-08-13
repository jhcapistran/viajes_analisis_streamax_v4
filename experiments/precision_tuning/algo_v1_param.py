"""
Version parametrizada de memory v1 (algo_memory.py::run_prototype_memory),
EXACTAMENTE la misma logica, pero:
  - Recibe una lista de frames YA precomputados (embeddings ya parseados),
    en vez de un DataFrame crudo -> evita re-parsear JSON en cada corrida
    del grid search.
  - Los umbrales de confirmacion son parametros de la funcion (con default
    = valores actuales de algo_memory.py) en vez de constantes de modulo,
    para poder barrer un grid sin duplicar/editar algo_memory.py.

No se modifica algo_memory.py. identity_id NUNCA se usa como input del
algoritmo (solo se copia al registro de salida para evaluacion/GT).
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prototype_memory"))
from common import cosine_dist, physics_prob, CONSTANTS

VISUAL_MATCH = CONSTANTS['visual_match']  # 0.5, fijo (no se toca: se quiere
                                            # mantener igual de sensible
                                            # POSIBLE_CAMBIO / recall)

DEFAULT_PARAMS = {
    'memory_size': 6,
    'candidate_window': 3,
    'candidate_min_support': 2,
    'coherence_threshold': 0.35,
    'min_avg_dist_confirm': 0.85,
    'min_phys_confirm': 0.6,
}

_ID_COUNTER = [0]


def _new_id():
    _ID_COUNTER[0] += 1
    return f"MP{_ID_COUNTER[0]:07d}"


def run_v1_param(frames, params=None, trip_id=None):
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)

    n = len(frames)
    records = []
    if n == 0:
        return records

    first = frames[0]
    current_id = _new_id()
    memory = [first['emb']] if first['emb'] is not None else []
    id_pool = {current_id: list(memory)}

    records.append({
        'ASSET_ID': trip_id,
        'PERSONA_ID': current_id,
        'IDENTITY_ID': first.get('identity_id', ''),
        'ARCHIVO': first.get('image_file', 'N/A'),
        'DECISION_SISTEMA': 'INICIO_VIAJE',
        'DELAY_CONFIRMACION': 0,
    })

    pending = None
    prev = first

    for i in range(1, n):
        cur = frames[i]
        emb_cur = cur['emb']
        speed_prev = prev['speed']
        speed_cur = cur['speed']
        delta_t = cur['ts_seconds'] - prev['ts_seconds']
        p_fisica = physics_prob(speed_prev, speed_cur, delta_t)

        decision = 'MISMO_CONDUCTOR'
        delay_confirm = 0

        if emb_cur is None:
            decision = 'INDETERMINADO'
        else:
            mem_centroid = np.mean(memory, axis=0) if memory else None
            dist_mem = cosine_dist(mem_centroid, emb_cur) if mem_centroid is not None else None

            if dist_mem is not None and dist_mem < VISUAL_MATCH:
                decision = 'MISMO_CONDUCTOR'
                pending = None
                memory.append(emb_cur)
                if len(memory) > p['memory_size']:
                    memory.pop(0)
                id_pool.setdefault(current_id, []).append(emb_cur)
            elif dist_mem is not None and p_fisica <= CONSTANTS['phys_impossible']:
                decision = 'MISMO_CONDUCTOR'
                memory.append(emb_cur)
                if len(memory) > p['memory_size']:
                    memory.pop(0)
                id_pool.setdefault(current_id, []).append(emb_cur)
            else:
                if pending is None:
                    pending = {'embs': [emb_cur], 'frames': [i], 'dists': [dist_mem]}
                else:
                    pending['embs'].append(emb_cur)
                    pending['frames'].append(i)
                    pending['dists'].append(dist_mem)
                    if len(pending['embs']) > p['candidate_window']:
                        pending['embs'].pop(0)
                        pending['frames'].pop(0)
                        pending['dists'].pop(0)

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

                if (len(coherent_group) >= p['candidate_min_support']
                        and avg_dist_vs_old >= p['min_avg_dist_confirm']
                        and p_fisica >= p['min_phys_confirm']):
                    cand_embs = [embs[k] for k in coherent_group]
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
                    id_pool.setdefault(current_id, [])
                    id_pool[current_id].extend(cand_embs)

                    decision = 'CAMBIO_CONFIRMADO'
                    start_frame = pending['frames'][coherent_group[0]]
                    delay_confirm = i - start_frame
                    pending = None
                else:
                    decision = 'POSIBLE_CAMBIO'

        records.append({
            'ASSET_ID': trip_id,
            'PERSONA_ID': current_id,
            'IDENTITY_ID': cur.get('identity_id', ''),
            'ARCHIVO': cur.get('image_file', 'N/A'),
            'DECISION_SISTEMA': decision,
            'DELAY_CONFIRMACION': delay_confirm,
        })
        prev = cur

    return records
