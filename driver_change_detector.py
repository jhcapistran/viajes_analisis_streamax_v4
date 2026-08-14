"""
driver_change_detector.py
==========================

Detector STREAMING de cambio de conductor a partir de embeddings faciales
(vision) + velocidad del vehiculo. Consolida en un unico archivo autocontenido
la logica ya validada del experimento de reduccion de falsos positivos
(ver experiments/precision_tuning/ y experiments/prototype_memory/):

  - Umbrales de confirmacion: experiments/precision_tuning/algo_v1_param.py
    + experiments/precision_tuning/winner_config.json (config congelada
    "3 de 4": candidate_window=4, candidate_min_support=3).
  - Correccion de la logica de "candidato" (reset duro al volver el
    conductor vigente, frames fisicamente imposibles no contaminan memoria):
    experiments/precision_tuning/candidate_fix/algo_candidate_fix.py.
  - Formulas de distancia coseno y probabilidad fisica:
    experiments/prototype_memory/common.py.

No incluye HMM, v2 (multiprototipo/CUSUM), grid search, evaluacion contra
Ground Truth, ni ningun campo identity_id/driver_id: solo el algoritmo de
DECISION en streaming, sin mirar al futuro (cada `update()` solo ve el
presente y lo ya acumulado del pasado).

Uso basico:

    from driver_change_detector import DriverChangeDetector

    detector = DriverChangeDetector()
    result = detector.update(embedding=embedding, speed=speed, timestamp=timestamp)
    print(result["decision"], result["person_id"])

Self-tests (sin dependencias externas, solo numpy):

    python driver_change_detector.py --self-test
"""
import math
import sys

import numpy as np

# ---------------------------------------------------------------------------
# 1. CONSTANTES (umbrales y parametros de fisica, valores EXACTOS y
#    congelados; no tocar sin repetir el proceso de tuning documentado en
#    experiments/precision_tuning/report_precision_tuning.md).
# ---------------------------------------------------------------------------

# Memoria del conductor vigente y ventana de candidato.
MEMORY_SIZE = 6                # maximo de embeddings guardados por conductor
CANDIDATE_WINDOW = 4           # maximo de anomalias consecutivas en un candidato
CANDIDATE_MIN_SUPPORT = 3      # minimo de anomalias mutuamente coherentes ("3 de 4")

# Umbrales de distancia coseno (embeddings normalizados -> rango [0, 2]).
VISUAL_MATCH = 0.5             # dist_mem < esto -> mismo conductor (match claro)
COHERENCE_THRESHOLD = 0.35     # dist entre anomalias del candidato -> "coherentes"
MIN_AVG_DIST_CONFIRM = 0.85    # distancia promedio minima vs memoria vieja para confirmar
ID_REUSE_THRESHOLD = 0.7       # dist vs centroide de un ID anterior -> mismo conductor de antes

# Umbrales de probabilidad fisica (ver physics_prob mas abajo).
PHYS_IMPOSSIBLE = 0.1          # p_fisica frame-a-frame <= esto -> glitch, no cambio real
MIN_PHYS_CONFIRM = 0.6         # p_fisica minima (anclada al ultimo frame estable) para confirmar

# Modelo cinematico simplificado usado para estimar cuanto tiempo requiere
# fisicamente un cambio de conductor (frenar + maniobra + arrancar).
A_DECEL = 2.0                  # desaceleracion tipica (m/s^2)
A_ACCEL = 0.5                  # aceleracion tipica de arranque (m/s^2)
T_MANIOBRA_STATIONARY = 15     # tiempo de maniobra (s) con vehiculo casi detenido (v_max < 1 km/h)
T_MANIOBRA_SLOW = 45           # tiempo de maniobra (s) a velocidad muy baja (v_max < 5 km/h)
T_MANIOBRA = 90                # tiempo de maniobra (s) en marcha normal
K_SIGMOID = 0.1                # suavizado de la curva logistica del tiempo sobrante


# ---------------------------------------------------------------------------
# 2. DISTANCIA COSENO
# ---------------------------------------------------------------------------

def cosine_dist(a, b):
    """Distancia coseno (1 - similitud) entre dos embeddings. None si algun
    embedding falta o tiene norma cero (no se puede comparar)."""
    if a is None or b is None:
        return None
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 0:
        return None
    return 1.0 - float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# 3. FISICA: probabilidad de que un cambio de conductor sea fisicamente
#    posible en el tiempo transcurrido (frenar, hacer la maniobra y volver
#    a arrancar).
# ---------------------------------------------------------------------------

def physics_prob(speed_prev, speed_cur, delta_t):
    """Sigmoide sobre el tiempo "sobrante" (delta_t - tiempo requerido para
    frenar + maniobra + arrancar). Valores cercanos a 1 -> hubo tiempo de
    sobra para un cambio real; cercanos a 0 -> fisicamente casi imposible."""
    v_prev = speed_prev / 3.6  # km/h -> m/s
    v_cur = speed_cur / 3.6

    t_frenado = v_prev / A_DECEL
    t_arranque = v_cur / A_ACCEL

    v_max = max(v_prev, v_cur)
    if v_max < 1.0:
        t_maniobra_dinamico = T_MANIOBRA_STATIONARY
    elif v_max < 5.0:
        t_maniobra_dinamico = T_MANIOBRA_SLOW
    else:
        t_maniobra_dinamico = T_MANIOBRA

    t_requerido = t_frenado + t_maniobra_dinamico + t_arranque
    t_sobra = delta_t - t_requerido
    return 1.0 / (1.0 + math.exp(-K_SIGMOID * t_sobra))


# ---------------------------------------------------------------------------
# 4. COHERENCIA: dado un grupo de embeddings anomalos (el candidato),
#    encontrar el subgrupo mas grande que sea mutuamente consistente entre
#    si (misma "cara nueva" repetida, no ruido disperso).
# ---------------------------------------------------------------------------

def _largest_mutual_clique(embeddings, threshold):
    """Para cada embedding, cuenta cuantos otros estan a distancia
    < threshold de el, y devuelve el cluster mas grande encontrado (indices
    en `embeddings`). Heuristica simple (no busca el clique optimo global),
    igual a la usada en el algoritmo original."""
    best = []
    for a in range(len(embeddings)):
        cluster = [a]
        for b in range(len(embeddings)):
            if a == b:
                continue
            d = cosine_dist(embeddings[a], embeddings[b])
            if d is not None and d < threshold:
                cluster.append(b)
        if len(cluster) > len(best):
            best = cluster
    return best


def robust_coherent_group(embeddings, threshold):
    """Grupo coherente robusto: parte del clique de pares mas grande y lo
    refina contra el centroide del propio grupo (quita miembros que no
    esten a < threshold del centroide), iterando hasta converger. Evita
    confirmar con un grupo que parecia coherente en pares pero no lo es
    contra un centroide estable."""
    if not embeddings:
        return []

    group = _largest_mutual_clique(embeddings, threshold)
    for _ in range(len(group)):
        if len(group) < 2:
            break
        centroid = np.mean([embeddings[k] for k in group], axis=0)
        new_group = [
            k for k in group
            if (d := cosine_dist(embeddings[k], centroid)) is not None and d < threshold
        ]
        if set(new_group) == set(group):
            break
        group = new_group
    return group


# ---------------------------------------------------------------------------
# 5. DETECTOR DE CAMBIO DE CONDUCTOR (streaming, solo presente + pasado)
# ---------------------------------------------------------------------------

DECISIONS = (
    "INICIO_VIAJE",
    "MISMO_CONDUCTOR",
    "POSIBLE_CAMBIO",
    "CAMBIO_CONFIRMADO",
    "INDETERMINADO",
)


class DriverChangeDetector:
    """Detector con estado, un frame a la vez, via `update()`.

    Estado interno:
      - person_id: id del conductor vigente (generado internamente, no es
        un identity_id/driver_id externo).
      - memory: hasta MEMORY_SIZE embeddings recientes del conductor vigente.
      - window: ventana FIFO (hasta CANDIDATE_WINDOW) de observaciones
        anomalas consecutivas desde el ultimo reset duro. Se resetea por
        completo (lista vacia) cuando el conductor vigente reaparece
        (match claro) o cuando una observacion anomala resulta fisicamente
        imposible (glitch de deteccion) -> las anomalias separadas por
        cualquiera de esos dos eventos NUNCA se mezclan en un mismo
        candidato (regla "A A X A X X X": el primer X muere solo).
      - last_stable: ultimo frame (embedding/velocidad/timestamp) en el que
        hubo un match claro con el conductor vigente. Se usa como ancla
        temporal para la fisica de CONFIRMACION (el tiempo disponible para
        un cambio real se mide desde ahi, no solo desde el frame anterior,
        que podria ser otra anomalia del mismo candidato).
      - id_pool: embeddings acumulados por cada person_id ya usado en el
        viaje, para poder reutilizar un id anterior si el candidato
        confirmado matchea a un conductor que ya aparecio antes.
    """

    def __init__(self):
        self._next_id_num = 0
        self.person_id = None
        self.memory = []
        self.window = []
        self.last_stable = None
        self.prev = None
        self.id_pool = {}

    def _new_person_id(self):
        self._next_id_num += 1
        return f"P{self._next_id_num:06d}"

    def _push_memory(self, embedding):
        self.memory.append(embedding)
        if len(self.memory) > MEMORY_SIZE:
            self.memory.pop(0)
        self.id_pool.setdefault(self.person_id, []).append(embedding)

    def _reuse_or_new_id(self, candidate_centroid):
        """Regla 10: antes de crear un id nuevo, compara el candidato
        confirmado contra los ids anteriores (memoria acumulada de cada
        uno). Si hay match (dist < ID_REUSE_THRESHOLD), reutiliza el id
        anterior con mas evidencia acumulada (y, en empate, el mas
        cercano) en vez de crear uno nuevo."""
        reused_id = None
        best_pool_size = 0
        best_dist = float("inf")
        for pid, pool in self.id_pool.items():
            if pid == self.person_id or not pool:
                continue
            centroid = np.mean(pool, axis=0)
            d = cosine_dist(centroid, candidate_centroid)
            if d is not None and d < ID_REUSE_THRESHOLD:
                if len(pool) > best_pool_size or (len(pool) == best_pool_size and d < best_dist):
                    best_pool_size = len(pool)
                    best_dist = d
                    reused_id = pid
        return reused_id if reused_id is not None else self._new_person_id()

    def update(self, embedding, speed, timestamp):
        """Procesa un frame nuevo y devuelve la decision. `embedding` debe
        ser un vector (array-like) o None si no se pudo extraer cara en
        este frame; `speed` en km/h; `timestamp` en segundos (creciente).
        Solo usa el presente (este frame) y el pasado acumulado (memoria,
        candidato, ultimo frame estable): nunca mira frames futuros."""

        # --- Primer frame del viaje: arranca memoria e id, sin comparar. ---
        if self.prev is None:
            self.person_id = self._new_person_id()
            current = {"embedding": embedding, "speed": speed, "timestamp": timestamp}
            if embedding is not None:
                self.memory = [embedding]
                self.id_pool[self.person_id] = [embedding]
            self.last_stable = current
            self.prev = current
            return self._result(
                decision="INICIO_VIAJE",
                dist_mem=None,
                physics_score=None,
                explanation="Primer frame del viaje: arranca conductor y memoria nuevos.",
            )

        # --- Sin embedding en este frame: no se puede comparar nada. ---
        if embedding is None:
            self.prev = {"embedding": None, "speed": speed, "timestamp": timestamp}
            return self._result(
                decision="INDETERMINADO",
                dist_mem=None,
                physics_score=None,
                explanation="No hay embedding en este frame (cara no detectada).",
            )

        delta_t_frame = timestamp - self.prev["timestamp"]
        p_fisica_frame = physics_prob(self.prev["speed"], speed, delta_t_frame)

        mem_centroid = np.mean(self.memory, axis=0) if self.memory else None
        dist_mem = cosine_dist(mem_centroid, embedding) if mem_centroid is not None else None

        current = {"embedding": embedding, "speed": speed, "timestamp": timestamp}

        # --- Regla 3: match claro con el conductor vigente. ---
        if dist_mem is not None and dist_mem < VISUAL_MATCH:
            self.window = []  # reset duro: borra cualquier candidato en curso
            self._push_memory(embedding)
            self.last_stable = current
            self.prev = current
            return self._result(
                decision="MISMO_CONDUCTOR",
                dist_mem=dist_mem,
                physics_score=p_fisica_frame,
                explanation=f"dist_mem={dist_mem:.3f} < {VISUAL_MATCH} -> match claro con el conductor vigente.",
            )

        # --- Regla 4: anomalia visual pero fisicamente imposible = glitch. ---
        if dist_mem is not None and p_fisica_frame <= PHYS_IMPOSSIBLE:
            self.window = []  # reset duro: no se puede confiar en esta evidencia
            self.prev = current
            # NO se agrega a memoria (no contamina al conductor vigente).
            return self._result(
                decision="MISMO_CONDUCTOR",
                dist_mem=dist_mem,
                physics_score=p_fisica_frame,
                explanation=(
                    f"dist_mem={dist_mem:.3f} sugiere anomalia, pero "
                    f"physics_score={p_fisica_frame:.3f} <= {PHYS_IMPOSSIBLE} "
                    "(fisicamente imposible en este paso): se trata como ruido/glitch."
                ),
            )

        # --- Regla 5/6: no coincide con el conductor vigente -> candidato. ---
        self.window.append({"embedding": embedding, "dist_mem": dist_mem, "timestamp": timestamp, "speed": speed})
        if len(self.window) > CANDIDATE_WINDOW:
            self.window.pop(0)

        win_embeddings = [w["embedding"] for w in self.window]
        coherent_idx = robust_coherent_group(win_embeddings, COHERENCE_THRESHOLD)
        coherent_support = len(coherent_idx)
        avg_dist_vs_old = (
            float(np.mean([self.window[k]["dist_mem"] for k in coherent_idx]))
            if coherent_idx else 0.0
        )

        # Fisica de confirmacion: anclada al ultimo frame con match claro,
        # no al frame anterior (que puede ser otra anomalia del candidato).
        delta_t_confirm = timestamp - self.last_stable["timestamp"]
        p_fisica_confirm = physics_prob(self.last_stable["speed"], speed, delta_t_confirm)

        self.prev = current

        # --- Regla 7: condiciones para CONFIRMAR el cambio de conductor. ---
        if (coherent_support >= CANDIDATE_MIN_SUPPORT
                and avg_dist_vs_old >= MIN_AVG_DIST_CONFIRM
                and p_fisica_confirm >= MIN_PHYS_CONFIRM):
            candidate_embeddings = [win_embeddings[k] for k in coherent_idx]
            candidate_centroid = np.mean(candidate_embeddings, axis=0)

            # Regla 10: reutilizar id anterior si el candidato matchea.
            self.person_id = self._reuse_or_new_id(candidate_centroid)

            # Regla 9: memoria nueva = solo el candidato confirmado.
            self.memory = list(candidate_embeddings)
            self.id_pool.setdefault(self.person_id, [])
            self.id_pool[self.person_id].extend(candidate_embeddings)

            self.last_stable = current
            self.window = []

            return self._result(
                decision="CAMBIO_CONFIRMADO",
                dist_mem=dist_mem,
                physics_score=p_fisica_confirm,
                candidate_size=len(candidate_embeddings),
                coherent_support=coherent_support,
                explanation=(
                    f"{coherent_support}/{len(win_embeddings)} anomalias coherentes "
                    f"(< {COHERENCE_THRESHOLD}), avg_dist={avg_dist_vs_old:.3f} >= "
                    f"{MIN_AVG_DIST_CONFIRM}, physics_score={p_fisica_confirm:.3f} >= "
                    f"{MIN_PHYS_CONFIRM}: cambio de conductor confirmado."
                ),
            )

        # --- Regla 5/6: todavia no confirma, sigue como posible cambio. ---
        return self._result(
            decision="POSIBLE_CAMBIO",
            dist_mem=dist_mem,
            physics_score=p_fisica_confirm,
            candidate_size=len(self.window),
            coherent_support=coherent_support,
            explanation=(
                f"dist_mem={dist_mem:.3f} >= {VISUAL_MATCH} (no matchea al conductor "
                f"vigente), pero aun no cumple los 3 criterios de confirmacion "
                f"(soporte={coherent_support}/{CANDIDATE_MIN_SUPPORT}, "
                f"avg_dist={avg_dist_vs_old:.3f}/{MIN_AVG_DIST_CONFIRM}, "
                f"physics_score={p_fisica_confirm:.3f}/{MIN_PHYS_CONFIRM})."
            ),
        )

    def _result(self, decision, dist_mem, physics_score, candidate_size=0, coherent_support=0, explanation=""):
        return {
            "person_id": self.person_id,
            "decision": decision,
            "dist_mem": dist_mem,
            "physics_score": physics_score,
            "candidate_size": candidate_size,
            "coherent_support": coherent_support,
            "explanation": explanation,
        }


# ---------------------------------------------------------------------------
# 6. EJEMPLO DE USO
# ---------------------------------------------------------------------------

def _example():
    detector = DriverChangeDetector()

    driver_a = np.array([1.0, 0.0, 0.0, 0.0])
    driver_b = np.array([0.0, 1.0, 0.0, 0.0])

    frames = [
        dict(embedding=driver_a, speed=0.0, timestamp=0),
        dict(embedding=driver_a, speed=0.0, timestamp=60),
        dict(embedding=driver_b, speed=0.0, timestamp=120),
        dict(embedding=driver_b, speed=0.0, timestamp=180),
        dict(embedding=driver_b, speed=0.0, timestamp=240),
    ]
    for i, frame in enumerate(frames):
        result = detector.update(**frame)
        print(f"frame {i}: {result['decision']} (person_id={result['person_id']})")


# ---------------------------------------------------------------------------
# 7. SELF-TESTS
# ---------------------------------------------------------------------------

def _run_self_tests():
    A = np.array([1.0, 0.0, 0.0, 0.0])
    X = np.array([0.0, 1.0, 0.0, 0.0])
    X1 = np.array([0.0, 1.0, 0.0, 0.0])
    X2 = np.array([0.0, 0.0, 1.0, 0.0])
    X3 = np.array([0.0, 0.0, 0.0, 1.0])

    def run(frames):
        d = DriverChangeDetector()
        return [d.update(**f) for f in frames]

    # 1. A A X A X X X: el primer X muere al volver A; los X siguientes
    #    arman un candidato NUEVO que confirma en el frame 6.
    frames = [
        dict(embedding=A, speed=0.0, timestamp=0),
        dict(embedding=A, speed=0.0, timestamp=60),
        dict(embedding=X, speed=0.0, timestamp=120),
        dict(embedding=A, speed=0.0, timestamp=180),
        dict(embedding=X, speed=0.0, timestamp=240),
        dict(embedding=X, speed=0.0, timestamp=300),
        dict(embedding=X, speed=0.0, timestamp=360),
    ]
    results = run(frames)
    decisions = [r["decision"] for r in results]
    assert decisions[2] == "POSIBLE_CAMBIO", decisions
    assert decisions[3] == "MISMO_CONDUCTOR", decisions
    assert decisions[6] == "CAMBIO_CONFIRMADO", decisions
    assert results[6]["candidate_size"] == 3, results[6]  # solo los 3 X finales
    print("OK: A A X A X X X (primer X descartado, confirma con los 3 ultimos)")

    # 2. A A X X X: confirma con 3 anomalias consecutivas coherentes.
    frames = [
        dict(embedding=A, speed=0.0, timestamp=0),
        dict(embedding=A, speed=0.0, timestamp=60),
        dict(embedding=X, speed=0.0, timestamp=120),
        dict(embedding=X, speed=0.0, timestamp=180),
        dict(embedding=X, speed=0.0, timestamp=240),
    ]
    results = run(frames)
    assert results[-1]["decision"] == "CAMBIO_CONFIRMADO", [r["decision"] for r in results]
    print("OK: A A X X X (confirma)")

    # 3. A A X A A: nunca confirma, cada anomalia queda sola.
    frames = [
        dict(embedding=A, speed=0.0, timestamp=0),
        dict(embedding=A, speed=0.0, timestamp=60),
        dict(embedding=X, speed=0.0, timestamp=120),
        dict(embedding=A, speed=0.0, timestamp=180),
        dict(embedding=A, speed=0.0, timestamp=240),
    ]
    results = run(frames)
    assert "CAMBIO_CONFIRMADO" not in [r["decision"] for r in results]
    print("OK: A A X A A (nunca confirma)")

    # 4. Anomalia fisicamente imposible: no contamina la memoria de A.
    frames = [
        dict(embedding=A, speed=0.0, timestamp=0),
        dict(embedding=A, speed=0.0, timestamp=60),
        dict(embedding=X, speed=120.0, timestamp=61),  # 1s despues, a 120 km/h -> imposible
        dict(embedding=A, speed=0.0, timestamp=121),
    ]
    results = run(frames)
    assert results[2]["decision"] == "MISMO_CONDUCTOR", results[2]
    assert results[3]["decision"] == "MISMO_CONDUCTOR", results[3]
    assert results[3]["dist_mem"] < 1e-6, results[3]["dist_mem"]  # matchea A original, sin rastro de X
    print("OK: anomalia fisicamente imposible (memoria de A no se contamina)")

    # 5. Candidato incoherente: 3 anomalias mutuamente lejanas, nunca confirma.
    frames = [
        dict(embedding=A, speed=0.0, timestamp=0),
        dict(embedding=A, speed=0.0, timestamp=60),
        dict(embedding=X1, speed=0.0, timestamp=120),
        dict(embedding=X2, speed=0.0, timestamp=180),
        dict(embedding=X3, speed=0.0, timestamp=240),
    ]
    results = run(frames)
    decisions = [r["decision"] for r in results]
    assert "CAMBIO_CONFIRMADO" not in decisions, decisions
    assert decisions.count("POSIBLE_CAMBIO") == 3, decisions
    print("OK: candidato incoherente (nunca confirma)")

    print("\nTodos los self-tests pasaron.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _run_self_tests()
    else:
        _example()
