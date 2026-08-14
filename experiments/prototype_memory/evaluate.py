"""
Evaluacion generica: separa por viaje (nunca por filas), matchea alertas del
detector contra el GT (con ventana temporal para no penalizar corrimientos de
1-3 observaciones) y calcula:
  - Recall (con y sin GT_SUSPECT)
  - FP estrictos de CAMBIO_CONFIRMADO
  - Precision estricta
  - CAMBIO_CONFIRMADO correctos/incorrectos
  - # POSIBLE_CAMBIO
  - delay de confirmacion (frames de corrimiento vs el frame GT)
  - # GT_SUSPECT

Uso como libreria: evaluate(detail_df, gt_df, window=3) -> dict de metricas
+ dataframe de alertas CAMBIO_CONFIRMADO clasificadas (TP/FP) para el
analisis de causas.
"""
import pandas as pd

from common import match_events_window

WINDOW = 3  # radio de matching en # de observaciones (frames ya filtrados)


def _add_frame_idx(detail_df):
    detail_df = detail_df.copy()
    detail_df['frame_idx'] = detail_df.groupby('ASSET_ID').cumcount()
    return detail_df


def evaluate(detail_df, gt_df, window=WINDOW, label="baseline"):
    """IMPORTANTE: la metrica PRINCIPAL de recall es 'recall_confirmado'
    (cuantos cambios reales limpios terminan en una alerta CAMBIO_CONFIRMADO,
    la unica que actua automaticamente sin revision humana). Esa es la
    linea base a usar para tuning/comparar configs.

    Las metricas 'recall_*_cualquier_alerta' (POSIBLE_CAMBIO o
    CAMBIO_CONFIRMADO) NO son 'el' recall: son una tasa de deteccion mas
    laxa que solo dice si el cambio genero *algun* aviso (aunque haya
    quedado pendiente de revision humana sin auto-confirmarse). Utiles como
    piso de seguridad ("nada se pierde en silencio"), pero no deben
    reportarse ni leerse como si fueran el recall real del sistema.

    FP estricto se mide SOLO sobre CAMBIO_CONFIRMADO (asi lo pide el
    objetivo): una alerta CAMBIO_CONFIRMADO que no cae cerca de ningun
    evento GT (clean o suspect) es un FP estricto."""
    detail_df = _add_frame_idx(detail_df)

    rows_tp = []
    rows_fp = []
    delays = []
    n_gt_clean = 0
    n_gt_suspect = 0
    n_gt_clean_matched_any = 0
    n_gt_all_matched_any = 0
    n_gt_clean_matched_confirm = 0
    n_confirmados = 0
    n_posibles = 0
    n_confirmados_correctos = 0

    trips = set(detail_df['ASSET_ID'].unique()) | set(gt_df['trip_id'].unique() if len(gt_df) else [])

    for trip in trips:
        d = detail_df[detail_df['ASSET_ID'] == trip]
        g = gt_df[gt_df['trip_id'] == trip] if len(gt_df) else gt_df.iloc[0:0]

        confirm_frames = d.loc[d['DECISION_SISTEMA'] == 'CAMBIO_CONFIRMADO', 'frame_idx'].tolist()
        posible_frames = d.loc[d['DECISION_SISTEMA'] == 'POSIBLE_CAMBIO', 'frame_idx'].tolist()
        any_alert_frames = sorted(confirm_frames + posible_frames)
        n_confirmados += len(confirm_frames)
        n_posibles += len(posible_frames)

        gt_all_frames = g['frame_idx'].dropna().astype(int).tolist()
        gt_clean_frames = g.loc[~g['gt_suspect'], 'frame_idx'].dropna().astype(int).tolist()
        gt_suspect_frames = g.loc[g['gt_suspect'], 'frame_idx'].dropna().astype(int).tolist()
        n_gt_clean += len(gt_clean_frames)
        n_gt_suspect += len(gt_suspect_frames)

        # --- FP ESTRICTO: solo CAMBIO_CONFIRMADO contra TODO el GT ---
        matches_confirm_all, _, _ = match_events_window(gt_all_frames, confirm_frames, window)
        matched_confirm_frames = {a for _, a, _ in matches_confirm_all}
        n_confirmados_correctos += len(matches_confirm_all)
        for gf, af, delay in matches_confirm_all:
            delays.append(delay)
            rows_tp.append({'trip_id': trip, 'gt_frame': gf, 'alert_frame': af, 'delay': delay})
        for af in confirm_frames:
            if af not in matched_confirm_frames:
                rows_fp.append({'trip_id': trip, 'alert_frame': af})

        # --- RECALL PRINCIPAL (linea base): cambios reales limpios que
        # terminan en CAMBIO_CONFIRMADO (no cuenta POSIBLE_CAMBIO) ---
        matches_confirm_clean, _, _ = match_events_window(gt_clean_frames, confirm_frames, window)
        n_gt_clean_matched_confirm += len(matches_confirm_clean)

        # --- deteccion laxa (piso de seguridad): cualquier alerta,
        # POSIBLE_CAMBIO o CAMBIO_CONFIRMADO. NO es el recall principal. ---
        matches_any_all, _, _ = match_events_window(gt_all_frames, any_alert_frames, window)
        n_gt_all_matched_any += len(matches_any_all)
        matches_any_clean, _, _ = match_events_window(gt_clean_frames, any_alert_frames, window)
        n_gt_clean_matched_any += len(matches_any_clean)

    tp_df = pd.DataFrame(rows_tp)
    fp_df = pd.DataFrame(rows_fp)

    recall_confirmado = n_gt_clean_matched_confirm / n_gt_clean if n_gt_clean else float('nan')
    recall_clean = n_gt_clean_matched_any / n_gt_clean if n_gt_clean else float('nan')
    recall_all = n_gt_all_matched_any / (n_gt_clean + n_gt_suspect) if (n_gt_clean + n_gt_suspect) else float('nan')
    fp_strict = len(fp_df)
    precision_strict = n_confirmados_correctos / n_confirmados if n_confirmados else float('nan')

    metrics = {
        'label': label,
        'gt_clean': n_gt_clean,
        'gt_suspect': n_gt_suspect,
        'gt_total': n_gt_clean + n_gt_suspect,
        # METRICA PRINCIPAL (linea base): de los cambios reales limpios,
        # cuantos terminaron en CAMBIO_CONFIRMADO (automatico, sin revision).
        'recall_confirmado': recall_confirmado,
        'gt_clean_matched_confirmado': n_gt_clean_matched_confirm,
        'gt_clean_matched_any': n_gt_clean_matched_any,
        'cambio_confirmado_total': n_confirmados,
        'cambio_confirmado_correctos': n_confirmados_correctos,
        'cambio_confirmado_incorrectos_fp_estrictos': fp_strict,
        'posible_cambio_total': n_posibles,
        # Deteccion laxa (POSIBLE_CAMBIO o CAMBIO_CONFIRMADO). NO es 'el'
        # recall: es un piso de seguridad ("nada se pierde en silencio").
        'recall_sin_suspect_cualquier_alerta': recall_clean,
        'recall_con_suspect_cualquier_alerta': recall_all,
        'precision_estricta_cambio_confirmado': precision_strict,
        'delay_medio_frames_confirmado': (sum(delays) / len(delays)) if delays else float('nan'),
        'delay_max_frames_confirmado': max(delays) if delays else float('nan'),
        'window_frames': window,
    }
    return metrics, tp_df, fp_df


if __name__ == "__main__":
    import os
    from common import DATA_DIR

    for name in ["random_04", "reviewed_07"]:
        detail = pd.read_csv(os.path.join(DATA_DIR, f"baseline_detail_{name}.csv"))
        gt = pd.read_csv(os.path.join(DATA_DIR, f"gt_events_{name}.csv"))
        gt['gt_suspect'] = gt['gt_suspect'].astype(bool)
        m, tp, fp = evaluate(detail, gt, label=f"baseline_{name}")
        print(f"\n=== {name} ===")
        print(f"  >>> recall_confirmado (LINEA BASE): {m['recall_confirmado']}")
        for k, v in m.items():
            print(f"  {k}: {v}")
        fp.to_csv(os.path.join(DATA_DIR, f"baseline_fp_{name}.csv"), index=False)
        tp.to_csv(os.path.join(DATA_DIR, f"baseline_tp_{name}.csv"), index=False)
