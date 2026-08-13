"""
HMM causal pequeno (3 estados ocultos) que se monta ENCIMA de las senales que
ya calcula Memory v1 (algo_memory.py). Memory v1 sigue siendo el extractor de
evidencia (memoria de conductor, deteccion de anomalia, coherencia de
candidato, fisica). El HMM solo decide, frame a frame y de forma 100%
causal (filtrado forward, sin smoothing ni Viterbi retroactivo sobre el
viaje completo), que tan probable es que el conductor haya cambiado de
verdad.

Estados ocultos:
    0 = ESTABLE       (todo normal, el conductor vigente sigue ahi)
    1 = SOSPECHA      (evidencia rara / candidato en evaluacion)
    2 = CAMBIO_REAL   (el candidato es realmente un conductor nuevo)

Emisiones: discretas. No hace falta pedir Gaussianas/GMM para esto: la
evidencia que ya calcula v1 (distancia vs memoria, coherencia, fisica, si
volvio el conductor anterior) se discretiza en un puñado de simbolos
explicables (ver feature_extractor.py). Es simple, rapido y facil de
auditar caso por caso, tal como pide el experimento.

Transiciones fijadas a mano segun sentido operacional (ver
DEFAULT_TRANSITION) y ajustadas con un grid chico SOLO en DEV
(ver tune_hmm.py). ESTABLE->CAMBIO_REAL y CAMBIO_REAL->ESTABLE se dejan en
~0: no tiene sentido operacional saltar directo de "todo normal" a "cambio
confirmado" (o viceversa) sin pasar por SOSPECHA primero.
"""
import numpy as np

STATES = ["ESTABLE", "SOSPECHA", "CAMBIO_REAL"]
N_STATES = 3

SYMBOLS = ["MATCH", "RETURN", "PHYS_IMPOSSIBLE", "ANOMALY_WEAK", "ANOMALY_MEDIUM", "ANOMALY_STRONG", "NO_EMB"]

DEFAULT_TRANSITION = np.array([
    [0.95, 0.05, 0.00],   # ESTABLE ->
    [0.15, 0.65, 0.20],   # SOSPECHA ->
    [0.02, 0.03, 0.95],   # CAMBIO_REAL ->
])

# P(simbolo | estado), elicitado a mano a partir de la logica ya validada de
# v1 (no hace falta EM):
#   - MATCH / PHYS_IMPOSSIBLE son evidencia de estabilidad (mucho mas
#     probables bajo ESTABLE).
#   - RETURN (volvio a matchear la memoria vieja tras una racha anomala) es
#     la senal mas directa de "el conductor nunca cambio" -> favorece
#     ESTABLE/SOSPECHA, casi nula bajo CAMBIO_REAL.
#   - ANOMALY_WEAK: anomalia aislada, aun no coherente consigo misma (podria
#     ser ruido de un solo frame o el arranque de un cambio real).
#   - ANOMALY_MEDIUM: ya hay >=2 obs. coherentes entre si (candidato solido)
#     pero SIN cumplir el criterio completo de confirmacion de v1
#     (avg_dist_vs_old/fisica todavia flojos) -> evidencia intermedia.
#   - ANOMALY_STRONG: cumple el criterio COMPLETO que v1 usa para confirmar
#     (coherente + lejos de la memoria vieja + fisica solida). Es la senal
#     mas fuerte a favor de CAMBIO_REAL, pero v1 ya sabe (25% precision en
#     DEV) que ni siquiera este subconjunto filtrado es siempre un cambio
#     real -> por eso no se confirma en 1 sola observacion, hace falta que
#     el HMM acumule evidencia en el tiempo.
#   - NO_EMB es no informativo (misma probabilidad en los 3 estados) para
#     que un frame sin embedding no mueva la creencia.
DEFAULT_EMISSION = {
    "MATCH":           np.array([0.90, 0.06, 0.02]),
    "RETURN":          np.array([0.60, 0.38, 0.02]),
    "PHYS_IMPOSSIBLE": np.array([0.90, 0.09, 0.01]),
    "ANOMALY_WEAK":    np.array([0.45, 0.45, 0.05]),
    "ANOMALY_MEDIUM":  np.array([0.20, 0.50, 0.30]),
    "ANOMALY_STRONG":  np.array([0.05, 0.30, 0.65]),
    "NO_EMB":          np.array([1 / 3, 1 / 3, 1 / 3]),
}

INITIAL_BELIEF = np.array([1.0, 0.0, 0.0])  # cada viaje (y cada conductor vigente nuevo) arranca en ESTABLE


class CausalHMM3:
    """HMM de 3 estados con filtrado forward puro (streaming, 100% causal:
    en el frame t solo se usa informacion hasta t)."""

    def __init__(self, transition=None, emission=None):
        self.transition = np.array(transition, dtype=float) if transition is not None else DEFAULT_TRANSITION.copy()
        src_emission = emission if emission is not None else DEFAULT_EMISSION
        self.emission = {k: np.array(v, dtype=float) for k, v in src_emission.items()}
        row_sums = self.transition.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), f"Filas de transicion deben sumar 1: {row_sums}"

    def initial_belief(self):
        return INITIAL_BELIEF.copy()

    def step(self, belief, symbol):
        """Un paso de forward filtering: P(S_t | x_1..x_t) a partir de
        P(S_{t-1} | x_1..x_{t-1}) y el simbolo observado en t.
            predict:  belief_pred = belief_{t-1} @ transition
            update:   belief_t propto belief_pred * emission[simbolo]
        NO mira el futuro. NO es Viterbi sobre el viaje completo."""
        predicted = belief @ self.transition
        emis = self.emission.get(symbol, np.ones(N_STATES) / N_STATES)
        updated = predicted * emis
        total = updated.sum()
        if total <= 0:
            return predicted / predicted.sum()
        return updated / total
