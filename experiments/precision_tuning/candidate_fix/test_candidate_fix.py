"""
Tests chicos con secuencias sinteticas para algo_candidate_fix.py. Sin
pytest (no esta instalado en el repo, ver /memories/repo): script plano,
`assert` + prints, se corre con:

    uv run python experiments/precision_tuning/candidate_fix/test_candidate_fix.py

Casos (ver pedido del usuario):
  1. A A X A X X X       -> el primer X se descarta (no confirma con solo 1
                            X previo + 3 nuevos; confirma en frame 6 SOLO
                            con los X de frames 4,5,6).
  2. A A X X X           -> confirma (3 X coherentes seguidos, sin match de
                            A en el medio).
  3. A A X A A           -> nunca confirma (cada X queda solo, se cancela
                            apenas vuelve A).
  4. A A X(fisica         -> no contamina memoria de A (el frame de fisica
     imposible) A            imposible no se agrega a memory ni rompe el
                            match posterior de A contra su memoria original).
  5. candidato incoherente -> nunca confirma (tres anomalias mutuamente
     (X1 X2 X3 no coherentes) lejanas entre si, ninguna alcanza soporte 3).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from algo_candidate_fix import run_candidate_fix, DEFAULT_PARAMS  # noqa: E402

A = np.array([1.0, 0.0, 0.0, 0.0])
B = np.array([0.0, 1.0, 0.0, 0.0])
X1 = np.array([0.0, 1.0, 0.0, 0.0])
X2 = np.array([0.0, 0.0, 1.0, 0.0])
X3 = np.array([0.0, 0.0, 0.0, 1.0])


def frame(emb, speed=0.0, ts=0, identity_id=""):
    return {
        "emb": emb,
        "speed": speed,
        "ts_seconds": ts,
        "driver_id": "N/A",
        "identity_id": identity_id,
        "empty_cabin": "",
        "image_file": "N/A",
    }


def decisions(recs):
    return [r["DECISION_SISTEMA"] for r in recs]


def test_first_anomaly_discarded_after_a_returns():
    # A A X A X X X, un frame cada 60s, velocidad 0 (mucho tiempo de sobra
    # para la fisica de confirmacion).
    frames = [
        frame(A, ts=0),
        frame(A, ts=60),
        frame(B, ts=120),   # X descartable
        frame(A, ts=180),   # cancela el candidato anterior (reset duro)
        frame(B, ts=240),   # candidato nuevo, frame 4
        frame(B, ts=300),   # frame 5
        frame(B, ts=360),   # frame 6 -> deberia confirmar aqui
    ]
    recs = run_candidate_fix(frames, dict(DEFAULT_PARAMS), trip_id="t1")
    d = decisions(recs)
    assert d[2] == "POSIBLE_CAMBIO", d
    assert d[3] == "MISMO_CONDUCTOR", d
    assert d[6] == "CAMBIO_CONFIRMADO", d
    # El candidato confirmado NO debe incluir el frame 2 (el primer X).
    coherent = recs[6]["COHERENT_FRAMES"]
    assert 2 not in coherent, coherent
    assert set(coherent) == {4, 5, 6}, coherent
    print("OK: test_first_anomaly_discarded_after_a_returns")


def test_three_consecutive_anomalies_confirm():
    frames = [
        frame(A, ts=0),
        frame(A, ts=60),
        frame(B, ts=120),
        frame(B, ts=180),
        frame(B, ts=240),
    ]
    recs = run_candidate_fix(frames, dict(DEFAULT_PARAMS), trip_id="t2")
    d = decisions(recs)
    assert d[4] == "CAMBIO_CONFIRMADO", d
    print("OK: test_three_consecutive_anomalies_confirm")


def test_never_confirms_if_a_keeps_returning():
    frames = [
        frame(A, ts=0),
        frame(A, ts=60),
        frame(B, ts=120),
        frame(A, ts=180),
        frame(A, ts=240),
    ]
    recs = run_candidate_fix(frames, dict(DEFAULT_PARAMS), trip_id="t3")
    d = decisions(recs)
    assert "CAMBIO_CONFIRMADO" not in d, d
    print("OK: test_never_confirms_if_a_keeps_returning")


def test_physics_impossible_frame_does_not_contaminate_memory():
    frames = [
        frame(A, ts=0, speed=0.0),
        frame(A, ts=60, speed=0.0),
        # Frame "imposible": aparece B con velocidad muy alta a solo 1s del
        # anterior -> p_fisica frame-a-frame ~0 (no hay tiempo de maniobra).
        frame(B, ts=61, speed=120.0),
        # A vuelve, con tiempo normal otra vez.
        frame(A, ts=121, speed=0.0),
    ]
    recs = run_candidate_fix(frames, dict(DEFAULT_PARAMS), trip_id="t4")
    d = decisions(recs)
    assert d[2] == "MISMO_CONDUCTOR", d  # tratado como glitch, no candidato
    # El frame 3 (A) debe volver a matchear limpio contra la memoria
    # ORIGINAL de A (dist ~0), es decir B nunca se mezclo en la memoria.
    assert recs[3]["DIST_MEM"] is not None and recs[3]["DIST_MEM"] < 1e-6, recs[3]["DIST_MEM"]
    assert d[3] == "MISMO_CONDUCTOR", d
    print("OK: test_physics_impossible_frame_does_not_contaminate_memory")


def test_incoherent_candidate_never_confirms():
    frames = [
        frame(A, ts=0),
        frame(A, ts=60),
        frame(X1, ts=120),
        frame(X2, ts=180),
        frame(X3, ts=240),
    ]
    recs = run_candidate_fix(frames, dict(DEFAULT_PARAMS), trip_id="t5")
    d = decisions(recs)
    assert "CAMBIO_CONFIRMADO" not in d, d
    assert d.count("POSIBLE_CAMBIO") == 3, d
    print("OK: test_incoherent_candidate_never_confirms")


if __name__ == "__main__":
    test_first_anomaly_discarded_after_a_returns()
    test_three_consecutive_anomalies_confirm()
    test_never_confirms_if_a_keeps_returning()
    test_physics_impossible_frame_does_not_contaminate_memory()
    test_incoherent_candidate_never_confirms()
    print("\nTodos los tests pasaron.")
