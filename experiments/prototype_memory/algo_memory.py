"""
ALGORITMO PROPUESTO: Prototype Memory + confirmacion temporal (streaming).

Diferencias clave vs el baseline (main_analisis_completo_v2.process_asset_group):
  1. La comparacion "misma persona" ya NO es solo frame-a-frame: se compara
     el frame nuevo contra el CENTROIDE de una memoria acotada (ventana
     movil, ultimos K embeddings "buenos") del conductor vigente. Esto
     absorbe outliers de un solo frame (oclusion, blur, mala iluminacion)
     que en el baseline generaban un salto brusco vs el frame anterior.
  2. Una sola anomalia NUNCA genera CAMBIO_CONFIRMADO. Cuando el frame no
     matchea la memoria del conductor vigente, se abre un "candidato"
     (posible nuevo conductor) y se pide evidencia acumulada:
       - al menos 2 de las ultimas <= 3 observaciones anomalas deben ser
         COHERENTES entre si (formar su propio cluster, dist < visual_match);
       - si en el medio el conductor vigente reaparece (el frame vuelve a
         matchear la memoria vieja), el candidato se descarta sin generar
         alerta fuerte (la anomalia se trata como ruido transitorio).
     Antes de la confirmacion, la primera anomalia se marca POSIBLE_CAMBIO
     (si la fisica no la hace imposible), nunca CAMBIO_CONFIRMADO.
  3. Reutilizacion de IDs: si el candidato confirmado matchea el centroide
     de un conductor anterior del mismo viaje (umbral relajado, igual que
     el baseline), se reutiliza ese PERSONA_ID en vez de crear uno nuevo.

Es 100% streaming: en cada frame solo se usa informacion pasada (memoria
acotada) y como maximo 2 observaciones futuras de "espera de confirmacion"
(la tercera como limite), nunca informacion del resto del viaje.

No se modifica main_analisis_completo_v2.py; solo se reutilizan sus
funciones puras (load_emb, physics) a traves de common.py.
"""
import numpy as np

from common import load_emb, cosine_dist, physics_prob, CONSTANTS

VISUAL_MATCH = CONSTANTS['visual_match']          # 0.5 - umbral de "misma persona"
VISUAL_DIFF = CONSTANTS['visual_diff']             # 1.0 - umbral de "claramente distinto"
PHYS_IMPOSSIBLE = CONSTANTS['phys_impossible']     # 0.1
PHYS_POSSIBLE = CONSTANTS['phys_possible']         # 0.85
GRAY_HIGH = CONSTANTS['gray_zone_high']            # 0.73
GRAY_MEDIUM = CONSTANTS['gray_zone_medium']        # 0.63
GRAY_MEDIUM_HIGH = CONSTANTS['gray_zone_medium_high']  # 0.78
ID_REUSE_THRESHOLD = CONSTANTS['id_reuse_match_threshold']  # 0.7

MEMORY_SIZE = 6          # tamano de la ventana de memoria del conductor vigente
CANDIDATE_WINDOW = 3      # maximo de observaciones anomalas a esperar (2 de 3)
CANDIDATE_MIN_SUPPORT = 2  # confirmaciones necesarias dentro de la ventana
COHERENCE_THRESHOLD = 0.35   # mas estricto que visual_match: el candidato debe
                             # ser realmente coherente consigo mismo, no solo
                             # "no matchear" al conductor vigente
MIN_AVG_DIST_CONFIRM = 0.85  # distancia promedio minima vs memoria vieja para
                             # confirmar (evita confirmar drift leve aunque
                             # persista 2 frames)
MIN_PHYS_CONFIRM = 0.6       # fisica minima exigida EN LA CONFIRMACION (mas
                             # estricta que el umbral de apertura de candidato)

_ID_COUNTER = [0]


def _new_id():
    _ID_COUNTER[0] += 1
    return f"M{_ID_COUNTER[0]:06d}"


def run_prototype_memory(df_clean, img_col='image_file'):
    """Corre el algoritmo sobre un viaje ya preprocesado (mismo preprocesado
    que el baseline: filtro de calidad + velocidad + skip primer frame,
    ver common.preprocess_trip). Devuelve lista de dicts (un registro por
    frame), mismo esquema de columnas relevantes que el baseline para poder
    reusar evaluate.py sin cambios."""
    n = len(df_clean)
    records = []
    if n == 0:
        return records

    first = df_clean.iloc[0]
    current_id = _new_id()
    emb0 = load_emb(first.get('embedding'))
    memory = [emb0] if emb0 is not None else []
    id_pool = {current_id: [emb0] if emb0 is not None else []}

    records.append({
        'ASSET_ID': None,  # se completa afuera
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

    pending = None  # {'embs': [...], 'frames': [...], 'start_i': int}
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
            mem_centroid = np.mean(memory, axis=0) if memory else None
            dist_mem = cosine_dist(mem_centroid, emb_cur) if mem_centroid is not None else None

            if dist_mem is not None and dist_mem < VISUAL_MATCH:
                # Coincide con la memoria del conductor vigente.
                decision = 'MISMO_CONDUCTOR'
                explicacion = f'Match vs memoria del conductor vigente ({dist_mem:.3f})'
                if pending is not None:
                    explicacion += ' | candidato descartado (volvio el conductor anterior)'
                pending = None
                memory.append(emb_cur)
                if len(memory) > MEMORY_SIZE:
                    memory.pop(0)
                id_pool.setdefault(current_id, []).append(emb_cur)
            elif dist_mem is not None and p_fisica <= PHYS_IMPOSSIBLE:
                # Anomalia vs memoria pero fisicamente imposible que haya
                # cambiado (teletransporte) -> se trata como ruido/outlier.
                decision = 'MISMO_CONDUCTOR'
                explicacion = f'Anomalia visual ({dist_mem:.3f}) pero fisica imposible -> se ignora'
                memory.append(emb_cur)
                if len(memory) > MEMORY_SIZE:
                    memory.pop(0)
                id_pool.setdefault(current_id, []).append(emb_cur)
            else:
                # Anomalia vs memoria (dist_mem >= visual_match) y fisicamente
                # posible: abrir/continuar candidato a nuevo conductor.
                if pending is None:
                    pending = {'embs': [emb_cur], 'frames': [i], 'dists': [dist_mem]}
                else:
                    pending['embs'].append(emb_cur)
                    pending['frames'].append(i)
                    pending['dists'].append(dist_mem)
                    if len(pending['embs']) > CANDIDATE_WINDOW:
                        pending['embs'].pop(0)
                        pending['frames'].pop(0)
                        pending['dists'].pop(0)

                # Buscar coherencia: >= CANDIDATE_MIN_SUPPORT embeddings
                # mutuamente cercanas (cluster propio) dentro de la ventana.
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

                avg_dist_vs_old = float(np.mean([pending['dists'][k] for k in coherent_group])) if coherent_group else 0.0

                if (len(coherent_group) >= CANDIDATE_MIN_SUPPORT and avg_dist_vs_old >= MIN_AVG_DIST_CONFIRM
                        and p_fisica >= MIN_PHYS_CONFIRM):
                    # --- CONFIRMACION ---
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
                        if dd is not None and dd < ID_REUSE_THRESHOLD:
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
                    explicacion = (f'Evidencia acumulada: {len(coherent_group)} obs. coherentes '
                                   f'en ventana de {len(embs)} (delay={delay_confirm} frames), '
                                   f'dist_prom_vs_anterior={avg_dist_vs_old:.3f}, P_fisica={p_fisica:.2f}')
                    pending = None
                else:
                    decision = 'POSIBLE_CAMBIO'
                    explicacion = f'Anomalia vs memoria ({dist_mem:.3f}), esperando confirmacion (soporte={len(coherent_group)})'

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
