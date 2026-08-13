# Reporte v2: Memoria multiprototipo + candidato + CUSUM (DEV/TEST congelado)

## 1. Motivacion

Tras revisar manualmente los 60 casos `FP_MEMORY` de v1 en `reviewed_07`
(herramienta `fp_simple_review/index.html`), se confirmo que el 100% son
falsos positivos genuinos causados por **la misma persona generando
embeddings distintos pero internamente coherentes** (cambio de pose, luz o
angulo dentro del mismo viaje). El algoritmo v1 (`algo_memory.py`) compara
contra un **unico centroide** de memoria por conductor, que no representa
bien a un conductor con 2+ "modos" visuales, generando anomalias falsas
cuando aparece un modo no visto antes.

## 2. Split DEV/TEST (`build_split.py` -> `splits.csv`)

- Split por `trip_id` (nunca por frame), **seed = 42**, estratificado por
  `(dataset, tiene_cambio_real)` para garantizar cambios reales en ambos
  splits y proporciones similares entre `random_04` y `reviewed_07`.
- **DEV: 1416 viajes (75%)** — 749 `random_04` (45 cambios reales / 40
  viajes) + 667 `reviewed_07` (53 cambios reales / 45 viajes).
- **TEST: 471 viajes (25%)** — 249 `random_04` (13 cambios reales) + 222
  `reviewed_07` (20 cambios reales). **33 cambios reales en total en
  TEST.**
- TEST no se miro ni se uso para elegir parametros hasta el final.

## 3. Algoritmo v2 (`algo_memory_v2.py`)

Una sola version, combinando (sin inventar señales nuevas, sin ML, sin
retraining, sin `identity_id` como feature, 100% streaming/causal):

1. **Memoria multiprototipo**: hasta 3 centroides por conductor vigente
   (`_DriverMemory`), cada uno una media movil acotada (ventana 5). El
   frame nuevo se compara contra el prototipo mas cercano. Si matchea pero
   no es muy parecido a ninguno existente, se crea un prototipo nuevo (un
   "modo" visual mas) en vez de diluir un centroide unico. La memoria
   **nunca** se actualiza con frames de un candidato en evaluacion
   (queda congelada mientras se evalua un posible cambio).
2. **Candidato temporal**: ventana de hasta 4 observaciones anomalas
   recientes; se mide coherencia interna (cluster mutuo con umbral 0.35) y
   si el conductor vigente reaparece (match contra algun prototipo), el
   candidato se cancela sin tocar la memoria de A.
3. **CUSUM**: evidencia continua = `max(0, dist - 0.5) * P_fisica`,
   acumulada con decaimiento (`*0.9` por frame) en vez de la regla rigida
   "2 de 3" de v1. Confirma cuando la evidencia acumulada cruza un umbral
   Y hay >=2 observaciones coherentes entre si Y `P_fisica >= 0.6`.
4. **Telemetria (speed + delta_t, ya existentes en el baseline)**: si el
   frame anomalo llega tras un hueco >=15 min entre frames validos (proxy
   causal de una parada real / posible relevo) y la fisica es plausible, la
   evidencia de ese frame se multiplica x1.5 (acelera confirmacion en
   paradas reales, ataca el "TP retrasado" del Caso #8) sin bajar el minimo
   de observaciones coherentes exigido. No se usa `ignition` (revisado en
   una muestra de 200k filas de `random_04`: 199996/200000 = `True`,
   practicamente constante y no informativo) ni `empty_cabin` directo (esas
   filas ya se filtran aguas arriba en `preprocess_trip`, antes de llegar
   al algoritmo).
5. Streaming/causal puro: solo usa pasado + hasta 4 observaciones de
   espera, igual que v1.

**Iteraciones de ajuste en DEV** (3, dentro de "pocas iteraciones si algo
claramente falla"):
- v0 (umbral CUSUM=1.05, decay=0.85, atajo de confirmacion con 1 sola
  observacion tras parada larga): **exploto a 574 FP estrictos en DEV**
  (peor que baseline) — el atajo de 1 observacion era demasiado agresivo.
- v1 (umbral=1.6, decay=0.6, atajo solo como bonus de evidencia no de
  conteo): sobre-corrigio, **solo 1 FP pero recall cayo a 0.93 con 2
  confirmaciones totales** — demasiado conservador.
- v2 final (umbral=0.85, decay=0.9, bonus x1.5, min 2 observaciones
  coherentes siempre exigido): resultado razonable, ver metricas abajo.
  **Congelado aqui.**

## 4. Metricas DEV (para ajustar, ya congelado)

| Metrica | baseline | memory v1 | **memory v2 (frozen)** |
|---|---:|---:|---:|
| FP estrictos / 1000 viajes | 172.3 | 65.7 | 65.7 |
| Viajes con >=1 FP / 1000 | 117.2 | 54.4 | 64.3 |
| POSIBLE_CAMBIO a revision / 1000 viajes | 6191.4 | 8536.7 | **4807.2** |
| Cambios detectados / 1000 cambios reales | 1000.0 | 1000.0 | 969.4 |
| Cambios perdidos / 1000 cambios reales | 0.0 | 0.0 | 30.6 |
| Recall (sin suspect) | 1.00 | 1.00 | 0.97 |
| CAMBIO_CONFIRMADO correctos | 7 | 31 | 34 |
| CAMBIO_CONFIRMADO incorrectos (FP) | 244 | 93 | 93 |
| Precision estricta | 0.028 | 0.250 | 0.268 |
| Delay medio (frames) | -0.71 | 1.71 | 0.02 |

\* la version intermedia (v0), descartada por sobre-generar alertas, llego
a tener 279/1000 viajes con FP; ese numero NO corresponde a la version
final. La version congelada real da 64.3/1000, como en la tabla.
\*\* la precision estricta de v2 es ligeramente mayor que v1 (0.268 vs
0.250): mismo numero absoluto de FP (93) pero mas `CAMBIO_CONFIRMADO`
correctos (34 vs 31).

Archivos completos: `metrics_baseline_DEV.csv`, `metrics_memory_v1_DEV.csv`,
`metrics_memory_v2_DEV.csv`.

## 5. Metricas TEST (frozen, corrido UNA sola vez, no se ajusto nada despues)

471 viajes, **33 cambios reales** (`gt_clean`) + 19 sospechosos.

| Metrica | baseline | memory v1 | **memory v2 (frozen)** |
|---|---:|---:|---:|
| FP estrictos / 1000 viajes | 159.2 | 48.8 | 63.7 |
| Viajes con >=1 FP / 1000 viajes | 101.9 | 42.5 | 61.6 |
| POSIBLE_CAMBIO a revision / 1000 viajes | 6129.5 | 8352.4 | **4271.8** |
| Cambios detectados / 1000 cambios reales | 1000.0 | 1000.0 | 939.4 |
| Cambios perdidos / 1000 cambios reales | 0.0 | 0.0 | 60.6 |
| Recall (sin suspect) | 1.00 | 1.00 | 0.94 |
| CAMBIO_CONFIRMADO correctos | 0 | 7 | 9 |
| CAMBIO_CONFIRMADO incorrectos (FP) | 75 | 23 | 30 |
| Precision estricta | 0.00 | 0.233 | 0.231 |
| Delay medio (frames) | — | 1.43 | 2.44 |

Archivos completos: `metrics_baseline_TEST.csv`, `metrics_memory_v1_TEST.csv`,
`metrics_memory_v2_TEST.csv`.

## 6. Conclusion honesta

v2 **NO es una mejora clara y unidireccional** sobre v1 en TEST: el FP
estricto absoluto sube levemente (30 vs 23 de 471 viajes) y el recall baja
un poco (2 cambios reales perdidos de 33, por sus umbrales de coherencia
mas estrictos + decay del CUSUM). Sin embargo, v2 si logra dos cosas
concretas y consistentes en DEV y TEST:

- **~49% menos alertas `POSIBLE_CAMBIO` enviadas a revision humana**
  (menos carga operativa/alert fatigue) manteniendo el mismo orden de
  magnitud de FP estrictos.
- **Mas `CAMBIO_CONFIRMADO` correctos** (34 vs 31 en DEV; 9 vs 7 en TEST),
  gracias al mecanismo de aceleracion por parada larga (ataca
  especificamente el "TP retrasado" del Caso #8).

El objetivo original (reducir a cero los 60 FP de `reviewed_07` causados
por variabilidad de embeddings de la misma persona) **no se resolvio
completamente**: la memoria multiprototipo ayuda cuando un "modo" visual ya
fue visto antes en el viaje, pero no evita la alerta la **primera vez**
que aparece un modo nuevo y persistente (ej. se pone lentes de sol y no se
los saca mas) — en ese caso, v1 y v2 tienen la misma dinamica de fondo
(evidencia sostenida + coherente = confirmar), porque no hay forma de
distinguir causalmente "cambio real de conductor" de "el mismo conductor
con lentes nuevos" usando solo la cara — el algoritmo esta limitado por la
señal misma (embeddings), no por la logica de memoria.

## 7. Para comunicar al jefe (TEST, universo real de 471 viajes)

- El sistema, sobre viajes nunca vistos durante el ajuste (TEST, 471
  viajes, 33 cambios reales de conductor): **se equivoca (falso positivo
  estricto) en ~64 de cada 1000 viajes**, y **detecta ~939 de cada 1000
  cambios reales de conductor** (pierde ~61 de cada 1000 cambios reales).
- De cada 1000 viajes, **~4272 alertas `POSIBLE_CAMBIO`** llegan a
  revision humana (bajo de ~8352/1000 en v1 a ~4272/1000 en v2, -49%).
- De los cambios reales confirmados automaticamente (`CAMBIO_CONFIRMADO`),
  hoy **9 de 39 son correctos** en TEST (23% de precision estricta,
  similar a v1).

## 8. Artefactos guardados

- `splits.csv` — split reproducible DEV/TEST (seed=42) con columnas
  `trip_id, dataset, n_gt_clean, n_gt_suspect, has_change, strata, split`.
- `build_split.py` — script que genera el split.
- `algo_memory_v2.py` — algoritmo final congelado.
- `run_v2_algo.py` — corre v2 sobre ambos datasets completos ->
  `memory_detail_v2_{dataset}.csv`.
- `eval_split.py` — evalua baseline/v1/v2 restringido a un split
  (`DEV`/`TEST`) y calcula las metricas "por 1000".
- `metrics_{baseline,memory_v1,memory_v2}_{DEV,TEST}.csv` — metricas
  finales por algoritmo y split.
- `{baseline,memory_v1,memory_v2}_{tp,fp}_{DEV,TEST}.csv` — detalle de
  aciertos/errores por caso, por algoritmo y split.
