"""
ALGORITMO v2: Memoria MULTIPROTOTIPO + candidato temporal + acumulacion
tipo CUSUM (streaming, causal). Una sola version "fuerte", sin combinar
10 ideas sueltas.

Motivacion (post revision manual de los 60 FP estrictos de v1 en
reviewed_07, ver experiment_fp_reduction/fp_simple_review/): TODOS esos 60
casos son la MISMA persona generando embeddings distintos pero coherentes
(cambio de pose/luz/angulo). v1 comparaba contra un solo centroide de
memoria, que se "diluye" cuando el conductor tiene 2+ modos visuales
distintos (con/sin lentes, distinta luz), generando anomalias falsas. v2
ataca esto directamente con varios prototipos por conductor en vez de uno.

Cambios vs v1 (algo_memory.py):
  1. MEMORIA MULTIPROTOTIPO: hasta MAX_PROTOTYPES centroides por conductor
     vigente (en vez de 1 centroide global). El frame nuevo se compara
     contra el prototipo MAS CERCANO. Si el frame matchea (dist < umbral)
     pero no es muy parecido a ningun prototipo existente (esta en la zona
     media) y hay lugar, se crea un prototipo nuevo (captura un "modo"
     visual distinto del mismo conductor) en vez de forzarlo a diluir un
     centroide unico. Solo se actualizan prototipos con frames YA
     confirmados como del conductor vigente (nunca con frames de un
     candidato en evaluacion): memoria nunca se contamina con anomalias.
  2. CANDIDATO CONGELADO: mientras hay un candidato a nuevo conductor
     abierto, la memoria del conductor vigente (A) NO se actualiza con esos
     frames (se "congela"); si el candidato se cancela (A reaparece), la
     memoria de A sigue intacta.
  3. ACUMULACION CUSUM: en vez de la regla rigida "2 de 3 observaciones
     coherentes", se acumula evidencia continua (sube si el candidato
     persiste, es distinto de A y fisicamente posible; se resetea a 0 en
     cuanto A reaparece). Confirma cuando la evidencia acumulada cruza un
     umbral Y hay coherencia interna minima del candidato.
  4. TELEMETRIA: el paso de evidencia se pesa por la probabilidad fisica
     (mismo modelo que el baseline). Ademas, un hueco de tiempo largo entre
     frames consecutivos ya filtrados (proxy causal de parada real: motor
     detenido / cabina vacia un rato largo, evidenciado indirectamente por
     el salto en ts_seconds entre frames validos) permite confirmar mas
     rapido (menos observaciones) porque la probabilidad fisica ya es
     altisima y una parada larga es exactamente el contexto donde un
     relevo de conductor es plausible. No se inventan señales nuevas: solo
     se usa speed/tiempo (ya en physics_prob) y el propio delta_t entre
     frames validos.
  5. Streaming/causal: igual que v1, solo pasado + hasta CANDIDATE_WINDOW
     observaciones de espera.

No se modifica main_analisis_completo_v2.py; se reutilizan sus funciones
puras via common.py.
"""
import numpy as np

from common import load_emb, cosine_dist, physics_prob, CONSTANTS

VISUAL_MATCH = CONSTANTS['visual_match']        # 0.5 - "misma persona"
PHYS_IMPOSSIBLE = CONSTANTS['phys_impossible']  # 0.1
ID_REUSE_THRESHOLD = CONSTANTS['id_reuse_match_threshold']  # 0.7

MAX_PROTOTYPES = 3            # prototipos (modos visuales) por conductor vigente
PROTOTYPE_CAP = 5             # ventana efectiva de cada prototipo (media movil acotada)
PROTO_NEW_THRESHOLD = 0.4     # si el frame matchea (< VISUAL_MATCH) pero esta a
                              # >= esto del prototipo mas cercano, y hay lugar,
                              # se crea un prototipo nuevo en vez de diluir

CANDIDATE_WINDOW = 4          # observaciones anomalas recientes consideradas para coherencia
COHERENCE_THRESHOLD = 0.35    # candidato debe ser coherente consigo mismo
CANDIDATE_MIN_SUPPORT = 2     # minimo de observaciones coherentes para poder confirmar (SIEMPRE, incluso con parada larga)
CUSUM_THRESHOLD = 0.85        # evidencia acumulada minima para confirmar
CUSUM_DECAY = 0.9             # decaimiento por frame (evita que evidencia vieja persista para siempre)
MIN_PHYS_CONFIRM = 0.6        # fisica minima en el frame que confirma

LONG_GAP_SECONDS = 900        # 15 min sin frames validos = parada real plausible
LONG_GAP_STEP_BONUS = 1.5     # multiplica la evidencia (no el n de observaciones exigidas)
                               # requeridas cuando el frame anomalo llega tras una parada larga:
                               # acelera la confirmacion en paradas reales sin bajar el minimo
                               # de observaciones coherentes exigido (CANDIDATE_MIN_SUPPORT)

_ID_COUNTER = [0]


def _new_id():
    _ID_COUNTER[0] += 1
    return f"V2{_ID_COUNTER[0]:06d}"


class _DriverMemory:
    """Memoria multiprototipo de un conductor: hasta MAX_PROTOTYPES
    centroides, cada uno una media movil acotada (PROTOTYPE_CAP)."""

    def __init__(self, emb=None):
        self.prototypes = []  # lista de {'emb': np.array, 'n': int}
        if emb is not None:
            self.prototypes.append({'emb': emb, 'n': 1})

    def nearest(self, emb):
        if not self.prototypes or emb is None:
            return None, None
        dists = [cosine_dist(p['emb'], emb) for p in self.prototypes]
        best_i = int(np.argmin(dists))
        return best_i, dists[best_i]

    def update(self, emb):
        if emb is None:
            return
        idx, dist = self.nearest(emb)
        if idx is None:
            self.prototypes.append({'emb': emb, 'n': 1})
            return
        if dist >= PROTO_NEW_THRESHOLD and len(self.prototypes) < MAX_PROTOTYPES:
            self.prototypes.append({'emb': emb, 'n': 1})
            return
        proto = self.prototypes[idx]
        n = min(proto['n'], PROTOTYPE_CAP)
        proto['emb'] = (proto['emb'] * n + emb) / (n + 1)
        proto['n'] = proto['n'] + 1

    def centroid(self):
        if not self.prototypes:
            return None
        weights = np.array([p['n'] for p in self.prototypes], dtype=float)
        embs = np.array([p['emb'] for p in self.prototypes])
        return np.average(embs, axis=0, weights=weights)

    def all_embeddings_flat(self):
        return [p['emb'] for p in self.prototypes]


def run_prototype_memory_v2(df_clean, img_col='image_file'):
    """Misma interfaz/esquema de salida que run_prototype_memory (v1) para
    poder reusar evaluate.py sin cambios."""
    n = len(df_clean)
    records = []
    if n == 0:
        return records

    first = df_clean.iloc[0]
    current_id = _new_id()
    emb0 = load_emb(first.get('embedding'))
    memory = _DriverMemory(emb0)
    id_pool = {current_id: memory}

    records.append({
        'ASSET_ID': None,
        'PERSONA_ID': current_id,
        'DRIVER_ID': first.get('driver_id', 'N/A'),
        'IDENTITY_ID': first.get('identity_id', ''),
        'ARCHIVO': first.get(img_col, 'N/A'),
        'EMPTY_CABIN': first.get('empty_cabin', ''),
        'VELOCIDAD': float(first.get('speed', 0.0)),
        'DISTANCIA_VISUAL': None,
        'PROBABILIDAD_FISICA': 1.0,
        'DECISION_SISTEMA': 'INICIO_VIAJE',
        'EXPLICACION': 'Primer frame del viaje (post-filtrado)',
        'DELAY_CONFIRMACION': 0,
    })

    pending = None  # {'embs', 'frames', 'dists', 'cusum'}
    prev_row = first

    for i in range(1, n):
        cur = df_clean.iloc[i]
        emb_cur = load_emb(cur.get('embedding'))
        speed_prev = float(prev_row.get('speed', 0.0))
        speed_cur = float(cur.get('speed', 0.0))

        prev_ts = int(prev_row['ts_seconds'])
        cur_ts = int(cur['ts_seconds'])
        delta_t = cur_ts - prev_ts
        p_fisica = physics_prob(speed_prev, speed_cur, delta_t)

        dist_prev_frame = cosine_dist(load_emb(prev_row.get('embedding')), emb_cur)

        decision = 'MISMO_CONDUCTOR'
        explicacion = ''
        delay_confirm = 0

        if emb_cur is None:
            decision = 'INDETERMINADO'
            explicacion = 'Sin embedding'
        else:
            idx, dist_mem = memory.nearest(emb_cur)

            if dist_mem is not None and dist_mem < VISUAL_MATCH:
                decision = 'MISMO_CONDUCTOR'
                explicacion = f'Match vs prototipo mas cercano ({dist_mem:.3f})'
                if pending is not None:
                    explicacion += ' | candidato cancelado (volvio el conductor vigente)'
                pending = None
                memory.update(emb_cur)
            elif dist_mem is not None and p_fisica <= PHYS_IMPOSSIBLE:
                decision = 'MISMO_CONDUCTOR'
                explicacion = f'Anomalia visual ({dist_mem:.3f}) pero fisica imposible -> se ignora'
                memory.update(emb_cur)
            else:
                if pending is None:
                    pending = {'embs': [emb_cur], 'frames': [i], 'dists': [dist_mem], 'cusum': 0.0}
                else:
                    pending['embs'].append(emb_cur)
                    pending['frames'].append(i)
                    pending['dists'].append(dist_mem)
                    if len(pending['embs']) > CANDIDATE_WINDOW:
                        pending['embs'].pop(0)
                        pending['frames'].pop(0)
                        pending['dists'].pop(0)

                step = max(0.0, dist_mem - VISUAL_MATCH) * p_fisica
                if delta_t >= LONG_GAP_SECONDS and p_fisica >= MIN_PHYS_CONFIRM:
                    step *= LONG_GAP_STEP_BONUS
                pending['cusum'] = pending['cusum'] * CUSUM_DECAY + step

                embs = pending['embs']
                coherent_group = []
                for a in range(len(embs)):
                    cluster = [a]
                    for b in range(len(embs)):
                        if a == b:
                            continue
                        dd = cosine_dist(embs[a], embs[b])
                        if dd is not None and dd < COHERENCE_THRESHOLD:
                            cluster.append(b)
                    if len(cluster) > len(coherent_group):
                        coherent_group = cluster

                normal_confirm = (pending['cusum'] >= CUSUM_THRESHOLD
                                  and len(coherent_group) >= CANDIDATE_MIN_SUPPORT
                                  and p_fisica >= MIN_PHYS_CONFIRM)

                if normal_confirm:
                    cand_embs = [embs[k] for k in coherent_group]
                    cand_centroid = np.mean(cand_embs, axis=0)

                    reused_id = None
                    best_frames = 0
                    best_dist = float('inf')
                    for pid, mem_obj in id_pool.items():
                        if pid == current_id or not mem_obj.prototypes:
                            continue
                        centroid = mem_obj.centroid()
                        dd = cosine_dist(centroid, cand_centroid)
                        pool_n = sum(p['n'] for p in mem_obj.prototypes)
                        if dd is not None and dd < ID_REUSE_THRESHOLD:
                            if pool_n > best_frames or (pool_n == best_frames and dd < best_dist):
                                best_frames = pool_n
                                best_dist = dd
                                reused_id = pid

                    current_id = reused_id if reused_id is not None else _new_id()
                    if current_id in id_pool:
                        memory = id_pool[current_id]
                        for e in cand_embs:
                            memory.update(e)
                    else:
                        memory = _DriverMemory(cand_embs[0])
                        for e in cand_embs[1:]:
                            memory.update(e)
                        id_pool[current_id] = memory

                    decision = 'CAMBIO_CONFIRMADO'
                    start_frame = pending['frames'][coherent_group[0]]
                    delay_confirm = i - start_frame
                    explicacion = (f'evidencia_acumulada: {len(coherent_group)} obs. coherentes, '
                                   f'cusum={pending["cusum"]:.2f}, delay={delay_confirm} frames, '
                                   f'P_fisica={p_fisica:.2f}, delta_t={delta_t}s')
                    pending = None
                else:
                    decision = 'POSIBLE_CAMBIO'
                    explicacion = (f'Anomalia vs prototipos ({dist_mem:.3f}), cusum={pending["cusum"]:.2f}, '
                                   f'soporte={len(coherent_group)}')

        records.append({
            'ASSET_ID': None,
            'PERSONA_ID': current_id,
            'DRIVER_ID': cur.get('driver_id', 'N/A'),
            'IDENTITY_ID': cur.get('identity_id', ''),
            'ARCHIVO': cur.get(img_col, 'N/A'),
            'EMPTY_CABIN': cur.get('empty_cabin', ''),
            'VELOCIDAD': speed_cur,
            'DISTANCIA_VISUAL': round(dist_prev_frame, 4) if dist_prev_frame is not None else None,
            'PROBABILIDAD_FISICA': round(p_fisica, 4),
            'DECISION_SISTEMA': decision,
            'EXPLICACION': explicacion,
            'DELAY_CONFIRMACION': delay_confirm,
        })
        prev_row = cur

    return records
