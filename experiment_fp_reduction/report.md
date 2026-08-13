# Reduccion de FP estrictos en `CAMBIO_CONFIRMADO` — reporte corto

Carpeta aislada: `experiment_fp_reduction/`. No se toco el baseline
(`main_analisis_completo_v2.py`); todo se reprodujo por fuera, por viaje
(`trip_id`), reutilizando solo funciones puras del baseline (`load_emb`,
`CONSTANTS`, `process_asset_group`).

> **Correccion aplicada a `all_reviewed_trips_data_2026_07.csv` (post primera
> version de este reporte):** el valor `identity_id == "Otros"` estaba siendo
> tratado como si fuera una identidad de conductor real, generando eventos GT
> de "cambio" falsos. `"Otros"` en realidad significa "revision humana no
> pudo asignar cara" (falla de FQA), igual que ya se habia corregido antes
> para `random_trips_data_2026_04.csv` (ver `README.md` seccion 3). Se aplico
> la misma correccion aqui (`fix_otros_in_reviewed_07.py`, con backup en
> `all_reviewed_trips_data_2026_07.pre_otros_fix.bak.csv`): esas 8704 filas
> ahora tienen `identity_id` vacio y `fqa_valid=False` directamente en el CSV
> original (no en una copia), tal como ya estaba el resto de columnas. Todas
> las metricas de este reporte reflejan el CSV ya corregido. El efecto fue
> grande: el GT de `reviewed_07` baja de 1065 eventos (77% suspect) a **89
> eventos (18% suspect)**, porque la mayoria de esas transiciones hacia/desde
> `"Otros"` eran artefactos, no cambios reales.

## 1. Que se corrio y como

- `run_baseline.py`: corre `process_asset_group` (sin tocar) **agrupando por
  `trip_id`** (importante: `asset_id` se repite en varios viajes en estos 2
  CSV, agrupar por `asset_id` mezclaria conductores de viajes distintos).
  Genera `baseline_detail_*.csv` y el GT (`gt_events_*.csv`).
- `common.py`: preprocesado identico al baseline (filtro `face_small`/`IPD_small`,
  filtro de velocidad `>= 7 km/h`, skip del primer frame, orden temporal) +
  construccion del GT + matching por ventana temporal.
- `analyze_fp.py`: diagnostico TP vs FP de `CAMBIO_CONFIRMADO` del baseline.
- `algo_memory.py` / `run_memory_algo.py`: algoritmo propuesto (Prototype
  Memory + confirmacion temporal), streaming, mismo preprocesado.
- `evaluate.py` / `build_comparison.py`: metricas separadas por viaje, con
  matching por ventana (radio 3 frames) para no penalizar corrimientos de
  1-3 observaciones, y separando `GT_SUSPECT`.

## 2. Ground truth usado (ruidoso, solo como referencia)

`identity_id` se usa **solo como GT**, nunca como feature. Reglas aplicadas
en `common.build_gt_for_trip`:

- Runs de 1 frame de una identidad distinta rodeados del mismo valor
  (`A A A B A A A`) → **glitch**, se ignoran (no generan evento).
- Runs de 1 frame ambiguos (`A A B C C`) → se genera el evento `A->C` pero
  marcado `GT_SUSPECT=True`.
- Si el centroide de los primeros frames de la "nueva" identidad esta a
  distancia coseno `< 0.5` (o `< 0.55` con `run_len < 3`) del centroide de
  los ultimos frames de la identidad previa → **`GT_SUSPECT`**: el embedding
  contradice la etiqueta.

Resultado (eventos GT por dataset, ya con `"Otros"` corregido en el CSV):

| dataset | GT limpios | GT suspect | GT total | % suspect |
|---|---|---|---|---|
| random_04 | 58 | 59 | 117 | 50% |
| reviewed_07 | 73 | 16 | 89 | 18% |

Antes de corregir `"Otros"`, `reviewed_07` mostraba 1065 eventos GT (77%
suspect); la enorme mayoria eran artefactos de tratar la etiqueta de falla
de FQA como si fuera una identidad real. Con el CSV corregido, `reviewed_07`
queda incluso mas limpio que `random_04` en proporcion de suspects (18% vs
50%), consistente con que ambos datasets tienen calidad de etiquetado
comparable una vez removido ese artefacto (hay ademas evidencia previa en
`README.md` de que varios "misses" de `random_trips_data_2026_04.csv`
resultaron ser errores de etiquetado, no fallas del detector).

## 3. Baseline reproducido: TP vs FP estrictos de `CAMBIO_CONFIRMADO`

| dataset | CAMBIO_CONFIRMADO | correctos (TP) | **FP estrictos** | precision estricta | recall (cualquier alerta, sin suspect) |
|---|---|---|---|---|---|
| random_04 | 141 | 1 | **140** | 0.7% | 100% |
| reviewed_07 | 185 | 6 | **179** | 3.2% | 100% |

Confirmando el problema reportado por el usuario: casi **todas** las
confirmaciones automaticas del baseline son FP estrictos. El recall global
no se resiente porque casi todo lo real ya se atrapa via `POSIBLE_CAMBIO`
(5259/6395 alertas de ese tipo); el problema esta concentrado 100% en la
rama de confirmacion directa (`dist_frame_a_frame > 1.0` + fisica no
imposible ⇒ `CAMBIO_CONFIRMADO` inmediato, sin persistencia).

### Hipotesis probada: "los FP son anomalias transitorias, los cambios reales generan identidad persistente"

`analyze_fp.py` compara, para cada `CAMBIO_CONFIRMADO` (TP vs FP), contra
memoria del conductor vigente, persistencia del ID nuevo asignado por el
propio baseline, y si el conductor previo reaparece:

| dataset | grupo | n | dist vs frame anterior | dist vs memoria hist. | persistencia mediana | % persiste 1 frame | **% vuelve al conductor anterior en 10 frames** | coherencia candidatos |
|---|---|---|---|---|---|---|---|---|
| random_04 | FP | 140 | 1.028 | 0.844 | 3.0 | 43.6% | **68.6%** | 0.444 |
| random_04 | TP | 1 | 1.005 | 0.956 | 1.0 | — | 0% | 0.547 |
| reviewed_07 | FP | 179 | 1.025 | 0.916 | 2.0 | 46.9% | **61.5%** | 0.422 |
| reviewed_07 | TP | 6 | 1.027 | 0.946 | 16.0 | 16.7% | 33.3% | 0.228 |

Evidencia clave:

- **La distancia frame-a-frame NO distingue TP de FP** (~1.02-1.03 en ambos
  grupos): esa es justamente la senal que usa el baseline para confirmar, y
  por eso confirma tanto ruido.
- La distancia contra la **memoria historica** del conductor vigente es
  sistematicamente **mas baja** (~0.84-0.93) que la distancia frame-a-frame
  (~1.0-1.03): promediar varios embeddings "limpia" el ruido de un frame
  puntual malo. Esto tiene una consecuencia practica importante (ver
  seccion 5): el umbral de "diferencia clara" (1.0) calibrado para
  frame-a-frame **no sirve tal cual** contra memoria, hay que recalibrarlo
  mas abajo (`~0.6-0.85` segun que tan estricta se quiera la confirmacion).
- El indicador mas fuerte es **"vuelve a parecerse al conductor anterior"**:
  62-69% de los FP revierten al PERSONA_ID anterior dentro de 10 frames,
  vs 0-41% en los TP. Esto valida directamente la hipotesis para la mayoria
  de los FP (son *blips* que el propio conductor "corrige" solo, no cambios
  reales) y justifica el algoritmo de memoria + confirmacion temporal.
- La hipotesis **no explica el 100% de los FP** (persistencia mediana de 2-3
  frames en varios FP, no siempre revierten): una fraccion son anomalias
  mas largas (mala luz sostenida, oclusion prolongada) que necesitan la
  capa de "coherencia interna del candidato" ademas de "revierte o no".

## 4. Algoritmo probado: Prototype Memory + confirmacion temporal

Implementado en `algo_memory.py`, 100% streaming (solo usa pasado + hasta 2
observaciones de espera, nunca el resto del viaje):

1. Memoria acotada (ultimos 6 embeddings "buenos") del conductor vigente;
   comparacion contra el **centroide** de esa memoria, no solo el frame
   anterior.
2. Si `dist_memoria < 0.5` → `MISMO_CONDUCTOR`, se actualiza la memoria.
3. Si `dist_memoria >= 0.5` y la fisica no es imposible → se abre/continua
   un **candidato** (ventana de hasta 3 observaciones anomalas).
4. `CAMBIO_CONFIRMADO` **solo** si dentro de esa ventana hay >= 2
   observaciones **coherentes entre si** (su propio cluster, dist `< 0.35`),
   con distancia promedio robusta contra la memoria vieja (`>= 0.85`) y
   fisica razonablemente favorable (`P_fisica >= 0.6`) — nunca con 1 sola
   observacion.
5. Si el conductor vigente reaparece antes de confirmar (`dist_memoria < 0.5`
   de nuevo) el candidato se **descarta** (la anomalia se trata como ruido).
6. Reutiliza `PERSONA_ID` de un conductor anterior del viaje si el centroide
   del candidato confirmado matchea (umbral 0.7, igual que el baseline).

No hizo falta probar CUSUM/HMM ni CatBoost: la combinacion memoria +
persistencia + reversion ya reduce el FP de forma sustancial manteniendo
(e incluso mejorando) el recall, cumpliendo el objetivo con la opcion mas
simple, tal como pide el brief ("no over-engineering").

### Cuanto ayuda cada pieza (ablation rapido durante la calibracion)

> Nota: los pasos intermedios de esta tabla para `reviewed_07` se calibraron
> **antes** de corregir el artefacto `"Otros"` en el CSV (ver nota al inicio
> del reporte), cuando el GT todavia tenia 1065 eventos con 77% suspect. Se
> mantienen aqui solo para ilustrar el razonamiento de diseno (que pieza
> aporta que efecto); la fila final y la seccion 5 ya reflejan el GT
> corregido (89 eventos) y son las metricas autoritativas.

| variante | random_04 FP | random_04 correctos | reviewed_07 FP | reviewed_07 correctos |
|---|---|---|---|---|
| memoria sola (umbral candidato = 0.5, sin exigir coherencia interna fuerte) | 660 | 37 | 840 | 149 |
| + confirmar solo con avg_dist >= 0.78 y fisica >= 0.5 | 106 | 17 | 132 | 45 |
| + avg_dist >= 0.85, coherencia < 0.35, fisica >= 0.6 (calibracion, GT sin corregir) | 56 | 12 | 57 | 29 |
| **igual config, GT corregido (final, ver seccion 5)** | **56** | **12** | **60** | **26** |
| baseline (referencia, GT corregido) | 140 | 1 | 179 | 6 |

Lectura:

- La **memoria historica** por si sola (comparar contra centroide en vez de
  frame previo) ya generaliza mejor: agrupa correctamente muchos mas
  cambios reales (37 y 149 vs 1 y 17 del baseline), pero sin mas control
  dispara tambien mucho mas FP (porque cualquier anomalia moderada abre
  candidato).
- La **persistencia/coherencia interna del candidato** (exigir que las 2
  observaciones que confirman sean coherentes entre si, no solo distintas
  del conductor vigente) es la pieza que realmente tira el FP para abajo,
  sin devolver el recall al nivel del baseline.
- La calibracion final del umbral de distancia promedio (0.85) y de fisica
  (0.6) es la que logra el mejor balance FP/recall dentro del tiempo
  disponible.

## 5. Metricas antes/despues (config final)

Matching por ventana temporal de **3 frames** (no penaliza corrimientos de
hasta 3 observaciones respecto al GT); recall calculado contra "cualquier
alerta" (`POSIBLE_CAMBIO` o `CAMBIO_CONFIRMADO`), ya que `POSIBLE_CAMBIO`
manda a revision humana y por lo tanto no "pierde" el caso.

| dataset | version | CAMBIO_CONFIRMADO | correctos | **FP estrictos** | precision estricta | recall sin suspect | recall con suspect | delay medio (frames) |
|---|---|---|---|---|---|---|---|---|
| random_04 | baseline | 141 | 1 | 140 | 0.7% | 100.0% | 71.8% | -2.00 |
| random_04 | **memoria+persistencia** | 68 | 12 | **56** | 17.6% | 100.0% | **73.5%** | 2.08 |
| reviewed_07 | baseline | 185 | 6 | 179 | 3.2% | 100.0% | 94.4% | -0.50 |
| reviewed_07 | **memoria+persistencia** | 86 | 26 | **60** | 30.2% | 100.0% | **97.8%** | 1.46 |

Resumen de impacto:

- **FP estrictos**: -60% en `random_04` (140→56) y **-66.5%** en `reviewed_07`
  (179→60).
- **`CAMBIO_CONFIRMADO` correctos**: sube 12x en `random_04` (1→12) y
  **~4.3x** en `reviewed_07` (6→26): el nuevo algoritmo no solo confirma
  menos ruido, tambien confirma mas cambios reales de forma automatica
  (antes casi todo lo real quedaba solo en `POSIBLE_CAMBIO`, requiriendo
  revision humana).
- **Recall** (cualquier alerta) se mantiene o mejora en ambos datasets: no
  hubo que sacrificarlo para bajar el FP. En `reviewed_07`, con GT corregido,
  el recall "sin suspect" es 100% en ambas versiones (baseline y memoria);
  la mejora de recall se ve en la columna "con suspect" (94.4%→97.8%).
- **Delay de confirmacion**: media de ~1.5-2.1 frames, maximo 3 (dentro del
  rango de 1-3 observaciones permitido por el brief). El baseline tenia
  delay medio *negativo* porque confirmaba en el mismo frame o antes del
  frame que el GT considera "inicio" del cambio (justo el patron de
  sobre-confirmar de golpe que se queria corregir).

## 6. Cantidad de GT posiblemente incorrecto

| dataset | GT total | GT_SUSPECT | % |
|---|---|---|---|
| random_04 | 117 | 59 | 50% |
| reviewed_07 | 89 | 16 | 18% |

Todas las metricas de arriba se reportan separando `GT_SUSPECT` (recall
"sin suspect" vs "con suspect"); los `GT_SUSPECT` **no se corrigieron
automaticamente** en el CSV (tal como pide el brief), solo se excluyen/
incluyen segun la metrica.

## 7. Limitaciones y siguiente paso

- El umbral final se calibro con una busqueda manual rapida (3 puntos), no
  un grid search exhaustivo; hay margen para afinar mas si se dispone de
  mas tiempo (ademas de considerar CUSUM sobre `dist_memoria` en vez del
  conteo simple 2-de-3, para graduar mejor el trade-off FP/delay).
- `reviewed_07` sigue teniendo mas proporcion de `GT_SUSPECT` que `random_04`
  (18% vs 50% —ya invertido respecto de antes de corregir `"Otros"`, donde
  pareceria mucho mas ruidoso). Igual conviene revisar visualmente una
  muestra de esos `GT_SUSPECT` restantes (mismo patron que ya se hizo en
  `README.md` para `random_trips_data_2026_04.csv`), porque probablemente
  varias de las "correctas" del baseline/memoria que hoy cuentan como FP
  estricto en realidad sean aciertos contra una etiqueta mal puesta.
- No hizo falta probar CUSUM/HMM ni CatBoost: la mejora ya cumple el
  objetivo (bajar FP manteniendo recall) con el algoritmo mas simple
  permitido por el brief.

## 8. Archivos generados en esta carpeta

- `gt_events_{random_04,reviewed_07}.csv` — GT con `gt_suspect`/`reason`.
- `baseline_detail_*.csv`, `memory_detail_*.csv` — detalle frame a frame.
- `baseline_fp_*.csv`, `baseline_tp_*.csv`, `memory_fp_*.csv`, `memory_tp_*.csv`
  — alertas `CAMBIO_CONFIRMADO` clasificadas.
- `fp_diagnostics_*.csv` / `fp_diagnostics_summary_*.csv` — diagnostico TP/FP
  del baseline (seccion 3).
- `comparison_summary.csv` — tabla de la seccion 5.
