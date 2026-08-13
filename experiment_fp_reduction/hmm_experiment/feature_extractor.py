"""
Extraccion de features + corrida del HMM causal sobre un viaje ya
preprocesado (mismo preprocesado que baseline/v1, ver common.preprocess_trip).

IMPORTANTE: Memory v1 (algo_memory.py) NO se modifica ni se reemplaza. Este
modulo reimplementa el MISMO bucle de memoria/candidato de v1 (memoria
acotada del conductor vigente, ventana de candidato anomalo, coherencia,
fisica, reutilizacion de PERSONA_ID) porque necesitamos exponer, frame a
frame, senales internas que v1 ya calcula pero no deja ver afuera (distancia
vs memoria del conductor actual, tamano del grupo coherente del candidato,
persistencia del candidato, si volvio el conductor anterior...).

La UNICA diferencia real de comportamiento vs v1 es el gate de
confirmacion: en vez de la regla fija de v1 ("2 de 3 observaciones
coherentes en ventana de 3"), la confirmacion la decide el HMM
(P_CAMBIO = P(S_t = CAMBIO_REAL) >= threshold_confirm), pudiendo esperar
mas de 3 frames si hace falta (streaming, causal, ver hmm_model.py).

NO se usa identity_id ni driver_id como feature (solo se copian a la salida
para inspeccion humana). NO se mira el futuro. NO se usa GT durante la
inferencia (el GT solo se usa despues, para evaluar en evaluate.py).
"""
import numpy as np

from common import load_emb, cosine_dist, physics_prob
import algo_memory as v1

VISUAL_MATCH = v1.VISUAL_MATCH
PHYS_IMPOSSIBLE = v1.PHYS_IMPOSSIBLE
COHERENCE_THRESHOLD = v1.COHERENCE_THRESHOLD
CANDIDATE_MIN_SUPPORT = v1.CANDIDATE_MIN_SUPPORT
MEMORY_SIZE = v1.MEMORY_SIZE
ID_REUSE_THRESHOLD = v1.ID_REUSE_THRESHOLD
MIN_AVG_DIST_CONFIRM = v1.MIN_AVG_DIST_CONFIRM
MIN_PHYS_CONFIRM = v1.MIN_PHYS_CONFIRM

# A diferencia de v1 (ventana fija de 3 obs.), aca se permite esperar mas
# frames para confirmar; se acota igual (streaming, memoria constante) para
# no acumular un candidato infinito.
CANDIDATE_BUFFER_MAX = 12

_ID_COUNTER = [0]


def _new_id():
    _ID_COUNTER[0] += 1
    return f"H{_ID_COUNTER[0]:06d}"


def _largest_coherent_group(embs):
    """Igual que v1: el subconjunto mutuamente cercano (dist < COHERENCE_THRESHOLD)
    mas grande dentro de la ventana de candidato."""
    best = []
    for a in range(len(embs)):
        cluster = [a]
        for b in range(len(embs)):
            if a == b:
                continue
            dd = cosine_dist(embs[a], embs[b])
            if dd is not None and dd < COHERENCE_THRESHOLD:
                cluster.append(b)
        if len(cluster) > len(best):
            best = cluster
    return best


def run_memory_hmm(df_clean, hmm, threshold_confirm, img_col='image_file'):
    """Corre Memory v1 (misma logica de memoria/candidato/fisica) + HMM
    causal encima, sobre un viaje ya preprocesado. Devuelve lista de dicts,
    un registro por frame, con el mismo esquema base que algo_memory.py
    (ASSET_ID/DECISION_SISTEMA/etc.) para poder reusar evaluate.py sin
    tocarlo, mas columnas de diagnostico del HMM (SIMBOLO, P_ESTABLE,
    P_SOSPECHA, P_CAMBIO, COHERENCIA, PERSISTENCIA_CANDIDATO).

    POSIBLE_CAMBIO se dispara igual que en v1 (CUALQUIER frame anomalo,
    fisicamente posible, vs la memoria del conductor vigente) para no
    destruir recall_ANY: eso es solo la cola de revision humana, no una
    decision fuerte. La UNICA pieza que reemplaza el HMM es el gate de
    CAMBIO_CONFIRMADO: en vez de la regla fija de v1 (2 de 3 coherentes),
    exige P_CAMBIO = P(S_t=CAMBIO_REAL) >= threshold_confirm."""
    n = len(df_clean)
    records = []
    if n == 0:
        return records

    first = df_clean.iloc[0]
    current_id = _new_id()
    emb0 = load_emb(first.get('embedding'))
    memory = [emb0] if emb0 is not None else []
    id_pool = {current_id: [emb0] if emb0 is not None else []}

    belief = hmm.initial_belief()

    records.append({
        'ASSET_ID': None,
        'PERSONA_ID': current_id,
        'DRIVER_ID': first.get('driver_id', 'N/A'),
        'IDENTITY_ID': first.get('identity_id', ''),
        'ARCHIVO': first.get(img_col, 'N/A'),
        'VELOCIDAD': float(first.get('speed', 0.0)),
        'DISTANCIA_MEMORIA': None,
        'PROBABILIDAD_FISICA': 1.0,
        'DELTA_T': 0,
        'SIMBOLO': 'INICIO',
        'COHERENCIA': 0,
        'PERSISTENCIA_CANDIDATO': 0,
        'P_ESTABLE': float(belief[0]),
        'P_SOSPECHA': float(belief[1]),
        'P_CAMBIO': float(belief[2]),
        'DECISION_SISTEMA': 'INICIO_VIAJE',
        'EXPLICACION': 'Primer frame del viaje (post-filtrado)',
        'DELAY_CONFIRMACION': 0,
    })

    pending = []  # lista de {'i', 'emb', 'dist'} - observaciones anomalas consecutivas
    prev_row = first
    prev_decision = 'INICIO_VIAJE'

    for i in range(1, n):
        cur = df_clean.iloc[i]
        emb_cur = load_emb(cur.get('embedding'))
        speed_prev = float(prev_row.get('speed', 0.0))
        speed_cur = float(cur.get('speed', 0.0))
        prev_ts = int(prev_row['ts_seconds'])
        cur_ts = int(cur['ts_seconds'])
        delta_t = cur_ts - prev_ts
        p_fisica = physics_prob(speed_prev, speed_cur, delta_t)

        symbol = 'NO_EMB'
        dist_mem = None
        coherence = 0
        persistence = len(pending)

        if emb_cur is not None:
            mem_centroid = np.mean(memory, axis=0) if memory else None
            dist_mem = cosine_dist(mem_centroid, emb_cur) if mem_centroid is not None else None

            if dist_mem is not None and dist_mem < VISUAL_MATCH:
                # Coincide con la memoria del conductor vigente.
                symbol = 'RETURN' if pending else 'MATCH'
                pending = []
                persistence = 0
                memory.append(emb_cur)
                if len(memory) > MEMORY_SIZE:
                    memory.pop(0)
                id_pool.setdefault(current_id, []).append(emb_cur)
            elif dist_mem is not None and p_fisica <= PHYS_IMPOSSIBLE:
                # Anomalia pero fisicamente imposible -> ruido/outlier, igual que v1.
                symbol = 'PHYS_IMPOSSIBLE'
                memory.append(emb_cur)
                if len(memory) > MEMORY_SIZE:
                    memory.pop(0)
                id_pool.setdefault(current_id, []).append(emb_cur)
            elif dist_mem is not None:
                # Anomalia vs memoria, fisicamente posible: candidato a nuevo conductor.
                pending.append({'i': i, 'emb': emb_cur, 'dist': dist_mem})
                if len(pending) > CANDIDATE_BUFFER_MAX:
                    pending.pop(0)
                embs = [p['emb'] for p in pending]
                coherent_group = _largest_coherent_group(embs)
                coherence = len(coherent_group)
                persistence = len(pending)
                if coherence < CANDIDATE_MIN_SUPPORT:
                    symbol = 'ANOMALY_WEAK'
                else:
                    avg_dist_vs_old = float(np.mean([pending[k]['dist'] for k in coherent_group]))
                    # Mismo criterio de confirmacion que v1 (coherente + lejos
                    # de la memoria vieja + fisica solida) se usa aca solo
                    # para DISCRETIZAR la evidencia en un simbolo mas fuerte,
                    # no para confirmar directamente: la decision final la
                    # toma el HMM acumulando evidencia en el tiempo.
                    if avg_dist_vs_old >= MIN_AVG_DIST_CONFIRM and p_fisica >= MIN_PHYS_CONFIRM:
                        symbol = 'ANOMALY_STRONG'
                    else:
                        symbol = 'ANOMALY_MEDIUM'

        belief = hmm.step(belief, symbol)
        p_estable, p_sospecha, p_cambio = (float(belief[0]), float(belief[1]), float(belief[2]))

        anomaly_active = symbol in ('ANOMALY_WEAK', 'ANOMALY_MEDIUM', 'ANOMALY_STRONG') and len(pending) > 0
        if anomaly_active and p_cambio >= threshold_confirm:
            decision = 'CAMBIO_CONFIRMADO'
        elif anomaly_active:
            decision = 'POSIBLE_CAMBIO'
        else:
            decision = 'MISMO_CONDUCTOR'

        delay_confirm = 0
        explicacion = f'P_CAMBIO={p_cambio:.3f} (simbolo={symbol})'

        # --- Accion de swap: solo en el frame ANOMALO que dispara la
        # confirmacion por primera vez (evita re-disparar cada frame
        # mientras la creencia se mantiene alta, y evita "confirmar" en un
        # frame que en realidad matchea/vuelve al conductor vigente). ---
        if (decision == 'CAMBIO_CONFIRMADO' and prev_decision != 'CAMBIO_CONFIRMADO'
                and symbol in ('ANOMALY_WEAK', 'ANOMALY_MEDIUM', 'ANOMALY_STRONG') and pending):
            embs = [p['emb'] for p in pending]
            coherent_group = _largest_coherent_group(embs)
            cand_idx = coherent_group if coherent_group else list(range(len(embs)))
            cand_embs = [embs[k] for k in cand_idx]
            cand_centroid = np.mean(cand_embs, axis=0)
            start_frame = pending[cand_idx[0]]['i']
            delay_confirm = i - start_frame

            reused_id = None
            best_frames, best_dist = 0, float('inf')
            for pid, pool in id_pool.items():
                if pid == current_id or not pool:
                    continue
                centroid = np.mean(pool, axis=0)
                dd = cosine_dist(centroid, cand_centroid)
                if dd is not None and dd < ID_REUSE_THRESHOLD:
                    if len(pool) > best_frames or (len(pool) == best_frames and dd < best_dist):
                        best_frames, best_dist, reused_id = len(pool), dd, pid

            current_id = reused_id if reused_id is not None else _new_id()
            memory = list(cand_embs)
            id_pool.setdefault(current_id, [])
            id_pool[current_id].extend(cand_embs)
            pending = []
            belief = hmm.initial_belief()  # nuevo conductor vigente = nuevo punto de partida ESTABLE
            explicacion = (f'HMM confirma: P_CAMBIO={p_cambio:.3f} cruzo threshold_confirm en frame '
                            f'anomalo (delay={delay_confirm}, coherencia={len(cand_idx)})')
        elif decision == 'POSIBLE_CAMBIO':
            explicacion = f'P_CAMBIO={p_cambio:.3f} en zona posible (simbolo={symbol})'

        records.append({
            'ASSET_ID': None,
            'PERSONA_ID': current_id,
            'DRIVER_ID': cur.get('driver_id', 'N/A'),
            'IDENTITY_ID': cur.get('identity_id', ''),
            'ARCHIVO': cur.get(img_col, 'N/A'),
            'VELOCIDAD': speed_cur,
            'DISTANCIA_MEMORIA': round(dist_mem, 4) if dist_mem is not None else None,
            'PROBABILIDAD_FISICA': round(p_fisica, 4),
            'DELTA_T': delta_t,
            'SIMBOLO': symbol,
            'COHERENCIA': coherence,
            'PERSISTENCIA_CANDIDATO': persistence,
            'P_ESTABLE': round(p_estable, 4),
            'P_SOSPECHA': round(p_sospecha, 4),
            'P_CAMBIO': round(p_cambio, 4),
            'DECISION_SISTEMA': decision,
            'EXPLICACION': explicacion,
            'DELAY_CONFIRMACION': delay_confirm,
        })
        prev_row = cur
        prev_decision = decision

    return records
