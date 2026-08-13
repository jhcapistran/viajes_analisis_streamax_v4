# Precision tuning de CAMBIO_CONFIRMADO (memory v1) — grid search sobre DEV

Objetivo: subir la precision de `CAMBIO_CONFIRMADO` de memory v1
(`algo_memory.py`) sin perder deteccion general (recall_ANY), ajustando
SOLO los umbrales de confirmacion ya existentes (nada de v2, nada de
algoritmo nuevo, `identity_id` nunca usado como feature). Todo el trabajo
se hizo sobre DEV; TEST no se toco.

## Paso 1 — Subset de ajuste

`build_subset.py` arma `subset_trips.csv` (295 viajes de DEV) union de:
- 77 viajes donde memory v1 tuvo FP estricto de CAMBIO_CONFIRMADO (DEV).
- 85 viajes de DEV con >=1 cambio real limpio.
- 150 viajes de DEV sin cambio real, muestreados al azar (seed=42) como control.

## Paso 2/3 — Grid search rapido sobre el subset

`run_grid.py` + `run_grid_extra.py` (extension puntual) probaron 162
configuraciones validas variando `candidate_window` (3-5),
`candidate_min_support` (2-4, respetando support<=window),
`coherence_threshold` (0.25-0.35), `min_avg_dist_confirm` (0.85-0.95) y
`min_phys_confirm` (0.6-0.8). `VISUAL_MATCH` (umbral de match contra
memoria) se dejo FIJO en el valor original (0.5) para no tocar la
sensibilidad de POSIBLE_CAMBIO / recall.

Resultado clave: dentro de este espacio, **recall_ANY se mantuvo en 1.0
en las 162 configuraciones** (el subset no tiene ningun cambio real que
dependa de que el umbral de confirmacion sea mas o menos estricto, porque
lo no confirmado cae a POSIBLE_CAMBIO, que igual cuenta para recall_ANY).
Configuraciones mas extremas (support>=4, coherence<=0.20, min_avg>=1.0)
colapsan CAMBIO_CONFIRMADO a 0 (nunca confirman) y se descartaron.

Resultados completos: `grid_results.csv`. Top 10 (ordenado por precision,
con piso de TP para no elegir configuraciones que "casi nunca confirman"):
`grid_top10.csv`.

## Paso 4 — Top 10 corridas sobre DEV completo (1416 viajes)

`run_top_on_full_dev.py` corrio 10 configuraciones candidatas + memory v1
(valores actuales) sobre TODO DEV. Tabla completa: `full_dev_results.csv`.

| config | precision_confirmado | TP | FP | recall_ANY | recall_confirmado | FP/1000 viajes |
|---|---|---|---|---|---|---|
| **E_w4_s3_c035_d085_p06 (GANADORA)** | **0.297** | **19** | **45** | **1.000** | **0.194** | **31.78** |
| A_w4_s2_c035_d085_p08 | 0.282 | 29 | 74 | 1.000 | 0.296 | 52.26 |
| B_w3_s2_c035_d085_p08 | 0.282 | 29 | 74 | 1.000 | 0.296 | 52.26 |
| F_w5_s3_c035_d085_p06 | 0.268 | 19 | 52 | 1.000 | 0.194 | 36.72 |
| D_w3_s2_c035_d085_p07 | 0.265 | 30 | 83 | 1.000 | 0.306 | 58.62 |
| H_w3_s2_c030_d085_p08 | 0.262 | 22 | 62 | 1.000 | 0.224 | 43.79 |
| G_w4_s2_c030_d085_p08 | 0.261 | 23 | 65 | 1.000 | 0.235 | 45.90 |
| C_w4_s2_c035_d085_p07 | 0.261 | 29 | 82 | 1.000 | 0.296 | 57.91 |
| I_w4_s2_c030_d085_p07 | 0.250 | 24 | 72 | 1.000 | 0.245 | 50.85 |
| **V1_DEFAULT (actual)** | **0.250** | **31** | **93** | **1.000** | **0.316** | **65.68** |
| J_w4_s2_c035_d085_p06 | 0.246 | 30 | 92 | 1.000 | 0.306 | 64.97 |

## Paso 5 — Configuracion ganadora (CONGELADA)

```
candidate_window        = 4   (antes 3)
candidate_min_support   = 3   (antes 2)
coherence_threshold     = 0.35  (sin cambios)
min_avg_dist_confirm    = 0.85  (sin cambios)
min_phys_confirm        = 0.6   (sin cambios)
memory_size             = 6     (sin cambios)
VISUAL_MATCH            = 0.5   (sin cambios, no se toco)
```

El unico cambio real respecto a memory v1 es exigir **3 de 4** observaciones
mutuamente coherentes en vez de **2 de 3** antes de confirmar un cambio de
conductor. Todo lo demas queda igual.

Guardado en `winner_config.json`. **CONGELADO: no se modifica mas.**

## v1 actual vs configuracion ganadora (DEV completo, 1416 viajes)

| Metrica | memory v1 (actual) | Ganadora (E) | Cambio |
|---|---|---|---|
| precision_confirmado | 0.250 | **0.297** | +19% relativo |
| TP_confirmado | 31 | 19 | -39% |
| FP_confirmado | 93 | **45** | -52% |
| recall_ANY | 1.000 | 1.000 | sin cambio |
| recall_confirmado | 0.316 | 0.194 | -39% |
| FP / 1000 viajes | 65.68 | **31.78** | -52% |

## Interpretacion honesta

- **recall_ANY no se toco**: en las 1416 viajes de DEV, la ganadora sigue
  detectando el 100% de los cambios reales limpios (igual que v1 actual),
  porque lo que deja de confirmarse cae a POSIBLE_CAMBIO (sigue yendo a
  revision humana, no se pierde en silencio).
- **CAMBIO_CONFIRMADO es ahora mas confiable**: casi 1 de cada 3 alertas
  confirmadas es correcta (antes 1 de cada 4), y los FP estrictos caen a
  la mitad (93 -> 45, o 65.7 -> 31.8 por cada 1000 viajes).
- **El costo es menos auto-confirmaciones**: 12 cambios reales que antes
  se auto-confirmaban ahora quedan como POSIBLE_CAMBIO (revision manual)
  en vez de CAMBIO_CONFIRMADO. Esto es consistente con el objetivo pedido:
  POSIBLE_CAMBIO sensible / CAMBIO_CONFIRMADO conservador.
- **No se logro una mejora "enorme"**: dentro del espacio de umbrales de
  memory v1 (sin tocar `VISUAL_MATCH` ni la arquitectura), la precision
  tiene un techo real cerca de 0.30 — configuraciones mas agresivas
  (support>=4, coherence mas chico, min_avg mas alto) simplemente dejan
  de confirmar casi nada (TP colapsa a 0), no mejoran la precision.
- Alternativa mas conservadora si se prefiere perder menos TP: **A/B**
  (`candidate_min_support=2`, `min_phys_confirm=0.8`) da precision 0.282
  (+13%) y FP 74 (-20%) manteniendo TP en 29 (solo -6%).

## TEST

No se corrio. Pendiente de aprobacion explicita para la corrida final
(unica) sobre TEST con esta configuracion congelada.

## Archivos generados

- `subset_trips.csv` — 295 viajes de DEV usados para el grid.
- `grid_results.csv` — 162 configuraciones probadas sobre el subset.
- `grid_top10.csv` — top 10 por precision (con piso de TP) sobre el subset.
- `full_dev_results.csv` — 10 configuraciones + v1 actual sobre DEV completo.
- `winner_config.json` — configuracion congelada.
- `algo_v1_param.py` — replica parametrizada de memory v1 (no se toco `algo_memory.py`).
- `precompute.py` — precomputo de embeddings (una sola vez por viaje) para no re-parsear en cada config del grid.
- `build_subset.py`, `run_grid.py`, `run_grid_extra.py`, `run_top_on_full_dev.py` — scripts del pipeline completo.
