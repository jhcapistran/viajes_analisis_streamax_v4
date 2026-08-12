# Handoff: Analisis de Cambios de Conductor con CSV

Fecha de corte: 2026-08-12

Este documento resume todo lo que ya se hizo en este repo para evaluar deteccion de cambios de conductor sobre `random_trips_data_2026_04.csv`, que se intento, que se descarto, que artefactos quedaron y cual es el estado actual para que otra IA pueda continuar sin reconstruir el contexto.

## 1. Objetivo de trabajo

El objetivo fue adaptar y evaluar los tres scripts principales sobre un CSV de viajes para responder estas preguntas:

- Cual de los 3 scripts da mejores resultados.
- Si se detectan los cambios del GT.
- Cuantos cambios hay en los ~1000 viajes y cuantos se detectan.
- Si se puede empujar el recall de V2 a >= 95%.
- En que viajes exactos falla y por que falla.
- Si algunos errores vienen del algoritmo o del propio GT.

## 2. Scripts principales del repo

- `main_analisis_completo.py`
  Versión base V1.
- `main_analisis_completo_v2.py`
  Versión productiva actual usada como baseline principal.
- `main_analisis_completo_v3.py`
  Versión experimental; mete mas alertas/ruido y no debe tomarse como baseline.

Scripts auxiliares creados durante este trabajo:

- `compare_versions_with_csv.py`
  Normaliza el CSV, agrupa por `trip_id`, ejecuta V1/V2/V3 y guarda salidas comparables.
- `grid_search_v2_focus.py`
  Hace grid search sobre un subset de viajes problemáticos para V2.
- `build_miss_review_manifest.py`
  Construye el manifiesto de imágenes a revisar alrededor de cambios GT no detectados.
- `build_miss_review_html.py`
  Construye un HTML secuencial por caso para revisión visual.

## 3. Dataset y cambios aplicados al CSV

Archivo principal:

- `random_trips_data_2026_04.csv`

Cambio manual aplicado al CSV por indicación del usuario:

- Los registros con `identity_id == "Otros"` se trataron como error de etiquetado.
- Se cambió `fqa_valid=False`.
- Se quitó el valor `"Otros"` y la celda quedó vacía.

Subset de enfoque creado para iterar sin correr siempre los ~1000 viajes:

- `random_trips_data_2026_04_v2_missed_gt_focus.csv`
- `random_trips_data_2026_04_v2_missed_gt_focus_meta.csv`

Ese subset contiene solo los viajes donde V2 tenía misses frente al GT proxy actual.

## 4. Cómo se adaptó la ejecución al CSV

`compare_versions_with_csv.py` hace estas transformaciones:

- Lee el CSV por `trip_id`.
- Usa solo filas con embedding no vacío.
- Normaliza nombres esperados por los scripts:
  `trip_id`, `asset_id`, `driver_id`, `image_file`, `source_image`, `timestamp`, `speed`, `embedding`.
- Genera `image_file` como `asset_id + "_" + basename(gs_path)`.
- Ejecuta cada versión con `process_asset_group(...)`.

La evaluación resultante quedó en:

- `outputs_csv_comparison/version_summary.csv`
- `outputs_csv_comparison/trip_metrics.csv`
- `outputs_csv_comparison/comparison_metrics.xlsx`
- `outputs_csv_comparison/v1_main_results.csv`
- `outputs_csv_comparison/v2_prod_results.csv`
- `outputs_csv_comparison/v3_experimental_results.csv`

## 5. Resultado comparativo de V1 / V2 / V3

Archivo principal:

- `outputs_csv_comparison/version_summary.csv`

Resultados:

- `v1_main`
  `998` viajes con salida, `139863` filas de salida, `1872` personas, `5400` alertas.
- `v2_prod`
  Idéntico a `v1_main` en métricas agregadas sobre este CSV.
- `v3_experimental`
  `999` viajes con salida, `158896` filas, `2199` personas, `6547` alertas.

Conclusión:

- V1 y V2 quedaron equivalentes en esta evaluación.
- V3 produce más alertas, más fragmentación y más ruido.
- La base de trabajo se mantuvo en `main_analisis_completo_v2.py`.

## 6. GT proxy y métricas de detección

El GT usado en esta etapa no viene de un archivo de eventos externo; se derivó del cambio de `DRIVER_ID` / identidad entre filas consecutivas del CSV ya normalizado.

Artefactos:

- `outputs_csv_comparison/v2_gt_change_events_detailed.csv`
- `outputs_csv_comparison/v2_gt_change_events_missed.csv`
- `outputs_csv_comparison/v2_gt_change_events_summary.csv`

Totales actuales:

- `96` cambios GT.
- `81` cambios contados como detectados por el CSV detallado actual.
- `15` cambios contados como miss por el CSV detallado actual.
- `76` viajes con al menos un cambio GT.
- `11` viajes con al menos un miss.

Viajes con misses usados para el subset:

- `1264001776343000`
- `1791321775188320`
- `711681775065007`
- `1263951775102100`
- `1214451776357406`
- `1480091776346277`
- `1264281775205639`
- `1648911775324423`
- `1808781776313205`
- `1808801776356704`
- `1866491775117837`

## 7. Inconsistencia importante de métrica

Hay dos juegos de números para el baseline y esto debe quedar explícito:

- `outputs_v2_recall_trials/trial_summary.csv`
  reporta baseline `86/96 = 89.58%`.
- `outputs_csv_comparison/v2_gt_change_events_detailed.csv`
  reporta `81/96`.

La diferencia son `5` casos donde el sistema sí dejó alerta en la misma fila GT, pero el CSV detallado actual no la contó como `matched_alert`.

Esos `5` casos son:

- `1263951775102100`
- `1808801776356704`
- `1791321775188320` fila 81
- `1791321775188320` fila 82
- `1480091776346277`

Interpretación correcta:

- Los `15` misses del CSV detallado actual mezclan fallos reales con fallos de medición.
- De esos `15`, `5` son `alerta_en_misma_fila_no_contada_por_metrica`.
- Los fallos reales del baseline son `10`.

Archivo que separa esas razones:

- `outputs_v2_grid_search_phase2/baseline_miss_reason_analysis.csv`

Conteo actual:

- `6` `visual_match_fuerte`
- `4` `zona_gris_baja_aun_con_fisica_favorable`
- `5` `alerta_en_misma_fila_no_contada_por_metrica`

## 8. Estado actual de `main_analisis_completo_v2.py`

La versión actual del código NO incluye la lógica experimental de zero-boundaries. Esa idea fue probada y descartada.

Puntos importantes del estado actual:

- Se mantiene filtrado top-down por velocidad:
  `speed >= min_stationary_speed`.
- `min_stationary_speed = 7.0`.
- Se sigue saltando el primer frame del viaje después del filtrado:
  `SKIPPING INITIAL FRAME` sigue activo.
- La física usa tiempos dinámicos de maniobra según velocidad:
  `t_maniobra_stationary`, `t_maniobra_slow`, `t_maniobra`.
- El veredicto visual/físico usa:
  `visual_match = 0.5`
  `visual_diff = 1.0`
  `gray_zone_high = 0.73`
  `gray_zone_medium = 0.63`
  `gray_zone_medium_high = 0.78`
  `phys_impossible = 0.1`
  `phys_possible = 0.85`

Valores actuales relevantes:

- `t_maniobra = 90`
- `t_maniobra_stationary = 15`
- `t_maniobra_slow = 45`
- `min_stationary_speed = 7.0`

## 9. Trials y experimentos que ya se intentaron

### 9.1. Comparación completa de V1 / V2 / V3

Hecho con:

- `compare_versions_with_csv.py`

Conclusión:

- V1 = V2 en agregado.
- V3 mete más alertas y más ruido.

### 9.2. Trial sin saltar el primer frame

Artefacto:

- `outputs_v2_recall_trials/trial_summary.csv`

Resultado:

- No mejoró recall.
- Generó más ruido.
- Quedó descartado.

### 9.3. Trial preservando zero-speed boundaries

Artefacto:

- `outputs_v2_recall_trials/trial_summary.csv`

Resultado:

- Recall cayó fuerte.
- Baseline reportado ahí: `86/96`.
- Trial `zero_speed_boundaries`: `63/96 = 65.62%`.
- También generó más alertas y más IDs.
- Quedó descartado y no debe volver a arrastrarse al código.

### 9.4. Grid search sobre subset de viajes problemáticos

Script:

- `grid_search_v2_focus.py`

Salida:

- `outputs_v2_grid_search_phase2/all_results.csv`
- `outputs_v2_grid_search_phase2/top_results.csv`
- `outputs_v2_grid_search_phase2/all_results_partial.csv`

Fase visual fijada:

- `visual_diff = 1.0`
- `gray_zone_high = 0.73`
- `gray_zone_medium = 0.59`
- `gray_zone_medium_high = 0.78`

Malla física barrida:

- `min_stationary_speed`: `3.0`, `5.0`, `7.0`
- `t_maniobra_stationary`: `10`, `15`, `20`, `30`
- `t_maniobra_slow`: `30`, `45`, `60`
- `t_maniobra`: `60`, `90`, `120`
- `phys_impossible`: `0.00`, `0.05`, `0.10`
- `phys_possible`: `0.75`, `0.80`, `0.85`

Mejor resultado del focus set:

- `18/25 = 72%` recall same-row.
- Mejor combo visible en `top_results.csv`:
  `combo_idx=810`
  `visual_diff=1.0`
  `gray_zone_high=0.73`
  `gray_zone_medium=0.59`
  `gray_zone_medium_high=0.78`
  `min_stationary_speed=7.0`
  `t_maniobra_stationary=15`
  `t_maniobra_slow=60`
  `t_maniobra=120`
  `phys_impossible=0.1`
  `phys_possible=0.85`

Conclusión principal del grid search:

- Con física sola NO se llega a `95%`.
- El cuello de botella ya no es la física.
- Los misses residuales son principalmente visuales.

## 10. Dónde falla realmente V2

Archivo principal:

- `outputs_v2_grid_search_phase2/best_combo_focus_miss_reason_analysis.csv`

En el mejor punto del grid search quedan `7` misses:

- `6` son `visual_match_fuerte`
- `1` es `zona_gris_baja_aun_con_fisica_favorable`

Eso significa:

- `6/7` fallan porque la distancia visual cae por debajo de `0.5`, o sea el sistema está convencido de que es el mismo conductor.
- Solo `1/7` parece recuperable tocando un umbral de zona gris.
- Por eso seguir moviendo física no va a resolver el problema completo.

Casos residuales del mejor combo:

- `711681775065007`
- `1866491775117837`
- `1264281775205639`
- `1808781776313205`
- `1264001776343000` evento fila 78
- `1264001776343000` evento fila 106
- `1214451776357406`

Distribución por viaje:

- `1264001776343000`: `2`
- `711681775065007`: `1`
- `1214451776357406`: `1`
- `1264281775205639`: `1`
- `1808781776313205`: `1`
- `1866491775117837`: `1`

## 11. Revisión visual con descarga de imágenes

Para revisar si el error es del algoritmo o del GT, se reutilizó el proyecto externo:

- `/home/capistran/Documents/download_gs_path`

Credenciales usadas:

- `/home/capistran/Documents/download_gs_path/.secrets/creds.json`

No se reescribió el downloader; se reutilizó `download_gs_from_file.py` y desde este repo solo se generó el manifiesto de imágenes a bajar.

Artefactos de revisión:

- `outputs_v2_grid_search_phase2/best_combo_focus_review_manifest.csv`
- `outputs_v2_grid_search_phase2/best_combo_focus_review_cases.csv`
- `outputs_v2_grid_search_phase2/review_images/`
- `outputs_v2_grid_search_phase2/review_images/review_cases.html`

Qué contiene el HTML actual:

- Solo los casos donde el GT dice que hubo cambio y V2 no lo detectó.
- Secuencia temporal única por caso.
- Visualización de ambas identidades:
  `ID_PREV` y `ID_NEW`.
- Si hay `empty_cabin`, se incluyen esos frames y también frames previos.
- Las miniaturas abren la imagen grande al hacer click.

Cantidad descargada:

- `148` imágenes únicas.
- `175` filas de contexto en el manifiesto.
- `0` errores de descarga.

Reportes de descarga:

- `outputs_v2_grid_search_phase2/review_images/descargas_20260812_091903.xlsx`
- `outputs_v2_grid_search_phase2/review_images/descargas_20260812_091906.xlsx`

Resultado de la revisión visual manual:

- Los casos que veníamos tratando como misses residuales del mejor combo resultaron ser el mismo conductor.
- Es decir, en esos casos el problema no era que V2 no detectara un cambio real; el problema era que el GT estaba mal etiquetado.
- Esto aplica a los casos residuales revisados en `review_cases.html`.
- En consecuencia, esos eventos no deben seguir usándose como evidencia de fallo del detector.

## 12. Archivos más importantes para retomar

Si otra IA tiene que continuar, estos son los archivos que debe abrir primero:

- `README.md`
- `main_analisis_completo_v2.py`
- `compare_versions_with_csv.py`
- `grid_search_v2_focus.py`
- `outputs_csv_comparison/version_summary.csv`
- `outputs_csv_comparison/v2_gt_change_events_detailed.csv`
- `outputs_v2_grid_search_phase2/baseline_miss_reason_analysis.csv`
- `outputs_v2_grid_search_phase2/best_combo_focus_miss_reason_analysis.csv`
- `outputs_v2_grid_search_phase2/review_images/review_cases.html`

## 13. Comandos útiles para reproducir

Nota:

- Usar `UV_CACHE_DIR=/tmp/uv-cache` para evitar problemas de lock en cache.

### 13.1. Comparación completa de versiones

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run python compare_versions_with_csv.py
```

### 13.2. Grid search sobre subset problemático

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run python grid_search_v2_focus.py
```

### 13.3. Reconstruir manifiesto de revisión

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run python build_miss_review_manifest.py
```

### 13.4. Reconstruir HTML de revisión

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run python build_miss_review_html.py
```

### 13.5. Volver a descargar imágenes desde el proyecto externo

Se ejecutó desde `/home/capistran/Documents/download_gs_path`.

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run python download_gs_from_file.py \
  hydra.run.dir=/home/capistran/Documents/viajes_analisis_streamax_v4/outputs_v2_grid_search_phase2/hydra_download_review \
  file_input.path=/home/capistran/Documents/viajes_analisis_streamax_v4/outputs_v2_grid_search_phase2/best_combo_focus_review_manifest.csv \
  file_input.output_dir=/home/capistran/Documents/viajes_analisis_streamax_v4/outputs_v2_grid_search_phase2/review_images \
  file_input.filter=null
```

## 14. Qué NO volver a intentar sin una hipótesis nueva

- No volver a empujar solo `phys_impossible`, `phys_possible`, `t_maniobra*` esperando llegar a `95%`.
- No volver a meter `zero_speed_boundaries` al código baseline.
- No volver a tomar V3 como punto de partida para producción.
- No interpretar los `15` misses del CSV detallado como `15` fallos reales del detector; `5` son problema de métrica.

## 15. Hipótesis actual más fuerte

La hipótesis dominante al cierre de este trabajo es:

- La revisión manual más reciente mostró que los casos residuales inspeccionados visualmente eran mismo conductor.
- Por lo tanto, al menos en esos casos, el problema estaba en el GT y no en V2.
- Esto reduce la evidencia de que el detector esté fallando estructuralmente en esos eventos concretos.
- Sigue siendo importante separar en cualquier evaluación futura:
  errores reales del detector vs errores/corrimientos del GT.

La revisión de `review_cases.html` se construyó precisamente para separar esas dos posibilidades, y en esta iteración permitió confirmar que esos residuales eran errores de GT.

## 16. Siguiente paso recomendado

El siguiente paso recomendado no es otro grid search ciego.

El siguiente paso recomendado es:

- depurar el GT para excluir o corregir los casos ya confirmados como mismo conductor;
- recomputar las métricas de recall con ese GT corregido;
- volver a distinguir qué misses restantes son reales y cuáles son problema de etiquetado;
- solo después decidir si hace falta tocar la lógica visual o temporal de V2.

## 17. GT corregido: resultado tras depurar los 7 casos confirmados

Este paso ya se ejecutó. Script usado: `apply_gt_corrections.py`.

Qué hace `apply_gt_corrections.py`:

- Corrige `identity_id` en `random_trips_data_2026_04.csv` para los 6 viajes
  afectados, fusionando la etiqueta incorrecta con la etiqueta previa correcta
  (queda un backup en `random_trips_data_2026_04.pre_gt_fix.bak.csv`).
- Quita los 7 eventos confirmados de `outputs_csv_comparison/v2_gt_change_events_detailed.csv`.
- Actualiza `outputs_csv_comparison/v2_gt_change_events_summary.csv` restando esos eventos por viaje.
- Quita esos mismos eventos de `outputs_v2_grid_search_phase2/best_combo_focus_miss_reason_analysis.csv`.

Fusiones de `identity_id` aplicadas (702 filas corregidas en total):

- `711681775065007`: `Identity_2 -> Identity_1`
- `1866491775117837`: `Identity_2 -> Identity_3`
- `1264281775205639`: `Identity_8 -> Identity_6`
- `1808781776313205`: `Identity_2 -> Identity_1`
- `1264001776343000`: parche acotado (no fusion completa del viaje, ver seccion 18)
- `1214451776357406`: `Identity_2 -> Identity_1`

Nota: no se toco la logica de deteccion (`process_asset_group`), porque `identity_id`
solo se usa como referencia/GT (`DRIVER_ID`), no como insumo de la decision visual/fisica.
Por eso no hace falta re-correr `compare_versions_with_csv.py`; solo se recalculo el
GT de referencia.

### 17.1. GT total tras la depuracion

- Antes: `96` cambios GT.
- Despues: `89` cambios GT (`96 - 7`).

### 17.2. Recall recalculado contra la corrida real de V2 (`v2_prod_results.csv`)

Se volvio a cruzar el GT corregido directamente contra las alertas reales de
`outputs_csv_comparison/v2_prod_results.csv` (match por misma fila `TRIP_ID + ARCHIVO`),
en vez de depender de la columna `matched_alert` ya calculada (esa columna es la que
tenia la inconsistencia de 5 casos descrita en la seccion 7).

Resultado:

- `89` eventos GT.
- `86` matched (misma fila).
- `3` misses reales.
- Recall same-row: `96.63%`.

Esto ya supera el objetivo de `>= 95%` planteado en la seccion 1, una vez que el GT
se depura de los errores de etiquetado ya confirmados visualmente.

### 17.3. Misses reales que quedaron (3, antes de la ronda 2)

- `1648911775324423` en `164891_1775329066000_4.jpeg` (`Identity_1 -> Identity_2`).
- `1264001776343000` en `126400_1776352552000_4.jpeg` (`Identity_1 -> Identity_2`).
- `1264001776343000` en `126400_1776353752000_4.jpeg` (`Identity_2 -> Identity_1`).

Los dos ultimos son en el mismo viaje que ya mostro oscilacion de `Identity_1`/`Identity_2`
a lo largo de todo el recorrido (ver seccion 11).

## 18. Ronda 2 de revision visual: resultado final

Se genero un manifiesto y HTML de revision para esos 3 misses, con las mismas
condiciones que la ronda 1 (`gt-radius=4`, `alert-radius=3`, mismo layout de
`build_miss_review_html.py`):

- `outputs_v2_grid_search_phase2/remaining_miss_reason_analysis_with_alerts.csv`
- `outputs_v2_grid_search_phase2/remaining_miss_review_manifest.csv`
- `outputs_v2_grid_search_phase2/remaining_miss_review_cases.csv`
- `outputs_v2_grid_search_phase2/review_images_remaining/review_cases.html` (60 imagenes unicas)

Resultado de la revision manual (3 casos, `miss_01` a `miss_03`):

- `miss_01` (`1648911775324423`, `164891_1775329066000_4.jpeg`): **error de GT**, mismo conductor.
- `miss_02` (`1264001776343000`, `126400_1776352552000_4.jpeg`, `15:15:52`): **cambio real de conductor**. No se toca.
- `miss_03` (`1264001776343000`, `126400_1776353752000_4.jpeg`, `15:35:52`): **error de GT**, mismo conductor (sigue siendo el conductor nuevo desde `15:15:52`, solo que la etiqueta cruda volvio a `Identity_1` por error).

### 18.1. Correccion aplicada

OJO con `1264001776343000`: no se puede fusionar `Identity_1`/`Identity_2` para
todo el viaje, porque dentro del mismo viaje hay un cambio real (`miss_02`) y
otros dos eventos ya `matched` (fila 68 y fila 119 del GT original) que no
fueron revisados y no deben tocarse. Por eso `apply_gt_corrections.py` ahora
usa `SCOPED_IDENTITY_PATCHES`: parches acotados por timestamp que solo
reemplazan la etiqueta incorrecta hasta el primer valor distinto en la etiqueta
cruda original (frontera con el siguiente evento, revisado o no), en vez de
fusionar toda la etiqueta del viaje.

Para `1648911775324423` (evento unico en el viaje) si se fusiono la etiqueta
completa, igual que en la ronda 1.

### 18.2. GT total y recall final

- GT total: `89 -> 87` (se remueven 2 de los 3 misses revisados; `miss_02` se queda).
- Matched (misma fila, contra `v2_prod_results.csv`): `86`.
- Recall same-row final: `86/87 = 98.85%`.

### 18.3. Unico miss real que queda en todo el dataset

- `1264001776343000` en `126400_1776352552000_4.jpeg` (`15:15:52`, `Identity_1 -> Identity_2`).

Este es un cambio de conductor real que V2 no detecto. Es el unico fallo real
confirmado del detector en las dos rondas de revision visual hechas hasta ahora.
Todo lo demas que parecia falla del detector resulto ser error de etiquetado en
el GT (`identity_id`).

### 18.4. Siguiente paso

- Si se quiere seguir mejorando, el foco ahora es un solo caso real
  (`1264001776343000` a las `15:15:52`), no un problema sistemico de GT ni de
  fisica/umbral visual.
- No tiene sentido correr otro grid search por 1 caso; conviene revisar ese caso
  puntual (ya esta en el HTML de revision, case `miss_02_1264001776343000`) para
  entender si es un patron replicable o un caso aislado antes de tocar codigo.

Si se quiere seguir con código, la siguiente mejora debe apuntar a lógica visual o temporal, no a física pura, y el margen de mejora ya es minimo (1 caso sobre 87).
