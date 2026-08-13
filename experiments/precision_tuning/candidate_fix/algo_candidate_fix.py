"""
Variante AISLADA de memory v1 (algo_v1_param.py::run_v1_param) que corrige
el manejo del "candidato" (ventana de observaciones anomalas usada para
confirmar CAMBIO_CONFIRMADO), sin tocar la config congelada "3 de 4"
(winner_config.json / experiments/precision_tuning/algo_v1_param.py). Los
umbrales por defecto son EXACTAMENTE los mismos que la config congelada
(window=4, min_support=3, coherence=0.35, min_avg_dist=0.85, min_phys=0.6,
memory_size=6, visual_match=0.5): el objetivo de esta variante es corregir
la LOGICA de estado, no los umbrales.

Bugs corregidos vs algo_v1_param.py:

1) Ventana = ultimas `candidate_window` observaciones ANOMALAS consecutivas
   desde el ultimo reset "duro" (ver 2 y 3), no un acumulador sin limite de
   tiempo/contexto. El reset duro es lo que evita que anomalias separadas
   por observaciones validas (match con A, o descartadas por fisica
   imposible) se mezclen en el mismo candidato.

2) Un match claro con el conductor vigente A (dist_mem < VISUAL_MATCH)
   CANCELA COMPLETAMENTE el candidato en curso (reset duro, ventana vacia).
   La siguiente observacion anomala arranca un candidato nuevo desde cero.
   Ejemplo: A A X A X X X -> el primer X (frame 2) se descarta al llegar la
   A del frame 3; el candidato que confirma en el frame 6 solo contiene los
   X de los frames 4,5,6.

3) Si una observacion es anomala en apariencia (dist_mem >= VISUAL_MATCH)
   pero la fisica indica cambio imposible (p_fisica frame-a-frame <=
   phys_impossible): decision explicita tomada aqui (documentada, no es la
   unica posible) -> se trata como ruido/glitch de deteccion, no se agrega
   a la memoria de A (no contamina su centroide) y TAMBIEN rompe/cancela
   el candidato en curso (reset duro), porque no se puede confiar en su
   evidencia para sostener nada (ni a favor de A ni de un candidato B).

4) El soporte "3 de 4" ya no es solo el clique de pares mutuamente
   coherentes (dist par-a-par < coherence_threshold): despues de encontrar
   ese grupo se recalcula su centroide y se vuelve a exigir que cada
   miembro este a < coherence_threshold del CENTROIDE del grupo, iterando
   hasta converger. Esto descarta grupos que parecian coherentes en pares
   pero no lo son contra un centroide robusto del candidato.

5) Se guardan en el output: DIST_MEM (distancia a memoria, la que realmente
   se usa para decidir, no confundir con DISTANCIA_VISUAL frame-a-frame),
   CANDIDATE_SIZE / CANDIDATE_FRAMES (ventana completa al momento de la
   decision), COHERENT_SUPPORT / COHERENT_FRAMES (subgrupo robusto usado
   para confirmar).

6) Fisica de confirmacion: para un candidato que lleva varias observaciones
   acumuladas, el tiempo disponible para un cambio de conductor se mide
   desde el ULTIMO FRAME ESTABLE de A (el ultimo match confirmado, que es
   el inicio real de la duda) hasta el frame actual, en vez de depender
   solo del frame inmediatamente anterior (que podria ser otra observacion
   anomala del mismo candidato, subestimando el tiempo real transcurrido).
   La deteccion de un frame individual "fisicamente imposible" (punto 3)
   sigue usando la fisica frame-a-frame (glitch de un solo paso), son dos
   preguntas distintas.

No se modifica algo_v1_param.py ni winner_config.json. identity_id nunca se
usa como input del algoritmo (solo se copia al registro de salida).
"""
import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "prototype_memory",
    ),
)
from common import cosine_dist, physics_prob, CONSTANTS  # noqa: E402

VISUAL_MATCH = CONSTANTS["visual_match"]  # 0.5, fijo (igual que v1)
PHYS_IMPOSSIBLE = CONSTANTS["phys_impossible"]  # 0.1, fijo (igual que baseline/v1)

# Mismos valores que la config congelada "3 de 4" (E_w4_s3_c035_d085_p06).
DEFAULT_PARAMS = {
    "memory_size": 6,
    "candidate_window": 4,
    "candidate_min_support": 3,
    "coherence_threshold": 0.35,
    "min_avg_dist_confirm": 0.85,
    "min_phys_confirm": 0.6,
}

_ID_COUNTER = [0]


def _new_id():
    _ID_COUNTER[0] += 1
    return f"MPX{_ID_COUNTER[0]:07d}"


def _robust_coherent_group(embs, coherence_threshold):
    """Punto 4: primero el clique de pares mutuamente coherentes mas grande
    (misma heuristica que v1), despues se refina contra el centroide del
    grupo, quitando miembros que no esten a < coherence_threshold del
    centroide, hasta converger. Devuelve indices (en `embs`) del grupo
    final."""
    n = len(embs)
    if n == 0:
        return []

    best = [0]
    for a in range(n):
        cluster = [a]
        for b in range(n):
            if a == b:
                continue
            dd = cosine_dist(embs[a], embs[b])
            if dd is not None and dd < coherence_threshold:
                cluster.append(b)
        if len(cluster) > len(best):
            best = cluster

    group = list(best)
    for _ in range(len(group)):
        if len(group) < 2:
            break
        centroid = np.mean([embs[k] for k in group], axis=0)
        new_group = [
            k for k in group
            if (d := cosine_dist(embs[k], centroid)) is not None and d < coherence_threshold
        ]
        if set(new_group) == set(group):
            break
        group = new_group
    return group


def run_candidate_fix(frames, params=None, trip_id=None):
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)

    n = len(frames)
    records = []
    if n == 0:
        return records

    first = frames[0]
    current_id = _new_id()
    memory = [first["emb"]] if first["emb"] is not None else []
    id_pool = {current_id: list(memory)}
    # Ultimo frame en que se confirmo/mantuvo un match claro con A: ancla
    # temporal para la fisica de confirmacion (punto 6).
    last_stable = first

    records.append({
        "ASSET_ID": trip_id,
        "PERSONA_ID": current_id,
        "IDENTITY_ID": first.get("identity_id", ""),
        "ARCHIVO": first.get("image_file", "N/A"),
        "DECISION_SISTEMA": "INICIO_VIAJE",
        "DELAY_CONFIRMACION": 0,
        "DIST_MEM": None,
        "DISTANCIA_VISUAL": None,
        "CANDIDATE_SIZE": 0,
        "CANDIDATE_FRAMES": None,
        "COHERENT_SUPPORT": 0,
        "COHERENT_FRAMES": None,
    })

    # Ventana del candidato en curso: lista de dicts {frame_idx, emb, dist_mem}
    # de observaciones ANOMALAS consecutivas desde el ultimo reset duro.
    window = []
    prev = first

    for i in range(1, n):
        cur = frames[i]
        emb_cur = cur["emb"]
        delta_t_frame = cur["ts_seconds"] - prev["ts_seconds"]
        p_fisica_frame = physics_prob(prev["speed"], cur["speed"], delta_t_frame)
        distancia_visual = cosine_dist(prev["emb"], emb_cur)  # frame-a-frame,
        # solo diagnostico: NO es la distancia usada para decidir (esa es
        # DIST_MEM, contra la memoria de A).

        decision = "MISMO_CONDUCTOR"
        delay_confirm = 0
        dist_mem_out = None
        candidate_frames_out = None
        coherent_frames_out = None
        coherent_support_out = 0

        if emb_cur is None:
            decision = "INDETERMINADO"
        else:
            mem_centroid = np.mean(memory, axis=0) if memory else None
            dist_mem = cosine_dist(mem_centroid, emb_cur) if mem_centroid is not None else None
            dist_mem_out = dist_mem

            if dist_mem is not None and dist_mem < VISUAL_MATCH:
                # Match claro con A -> reset duro del candidato (punto 2).
                decision = "MISMO_CONDUCTOR"
                window = []
                memory.append(emb_cur)
                if len(memory) > p["memory_size"]:
                    memory.pop(0)
                id_pool.setdefault(current_id, []).append(emb_cur)
                last_stable = cur
            elif dist_mem is not None and p_fisica_frame <= PHYS_IMPOSSIBLE:
                # Anomalo en apariencia pero fisicamente imposible en un solo
                # paso -> tratado como glitch/ruido (punto 3): NO contamina
                # la memoria de A, y rompe/cancela el candidato en curso
                # (decision explicita, ver docstring del modulo).
                decision = "MISMO_CONDUCTOR"
                window = []
            else:
                window.append({"frame_idx": i, "emb": emb_cur, "dist_mem": dist_mem})
                if len(window) > p["candidate_window"]:
                    window.pop(0)

                win_embs = [w["emb"] for w in window]
                coherent_idx = _robust_coherent_group(win_embs, p["coherence_threshold"])
                coherent_support_out = len(coherent_idx)
                avg_dist_vs_old = (
                    float(np.mean([window[k]["dist_mem"] for k in coherent_idx]))
                    if coherent_idx else 0.0
                )

                delta_t_confirm = cur["ts_seconds"] - last_stable["ts_seconds"]
                p_fisica_confirm = physics_prob(last_stable["speed"], cur["speed"], delta_t_confirm)

                candidate_frames_out = [w["frame_idx"] for w in window]
                coherent_frames_out = [window[k]["frame_idx"] for k in coherent_idx]

                if (len(coherent_idx) >= p["candidate_min_support"]
                        and avg_dist_vs_old >= p["min_avg_dist_confirm"]
                        and p_fisica_confirm >= p["min_phys_confirm"]):
                    cand_embs = [win_embs[k] for k in coherent_idx]
                    cand_centroid = np.mean(cand_embs, axis=0)

                    reused_id = None
                    best_frames = 0
                    best_dist = float("inf")
                    for pid, pool in id_pool.items():
                        if pid == current_id or not pool:
                            continue
                        centroid = np.mean(pool, axis=0)
                        dd = cosine_dist(centroid, cand_centroid)
                        if dd is not None and dd < CONSTANTS["id_reuse_match_threshold"]:
                            if len(pool) > best_frames or (len(pool) == best_frames and dd < best_dist):
                                best_frames = len(pool)
                                best_dist = dd
                                reused_id = pid

                    current_id = reused_id if reused_id is not None else _new_id()
                    memory = list(cand_embs)
                    id_pool.setdefault(current_id, [])
                    id_pool[current_id].extend(cand_embs)

                    decision = "CAMBIO_CONFIRMADO"
                    start_frame = window[coherent_idx[0]]["frame_idx"]
                    delay_confirm = i - start_frame
                    last_stable = cur
                    window = []
                else:
                    decision = "POSIBLE_CAMBIO"

        records.append({
            "ASSET_ID": trip_id,
            "PERSONA_ID": current_id,
            "IDENTITY_ID": cur.get("identity_id", ""),
            "ARCHIVO": cur.get("image_file", "N/A"),
            "DECISION_SISTEMA": decision,
            "DELAY_CONFIRMACION": delay_confirm,
            "DIST_MEM": dist_mem_out,
            "DISTANCIA_VISUAL": distancia_visual,
            "CANDIDATE_SIZE": len(window),
            "CANDIDATE_FRAMES": candidate_frames_out,
            "COHERENT_SUPPORT": coherent_support_out,
            "COHERENT_FRAMES": coherent_frames_out,
        })
        prev = cur

    return records
