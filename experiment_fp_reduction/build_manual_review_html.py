"""
HTML DE REVISION VISUAL MANUAL - experiment_fp_reduction
=========================================================

Genera un reporte HTML puramente visual (sin botones de clasificar, sin
formularios, sin almacenamiento de revision) para validar a mano los
resultados ya calculados de experiment_fp_reduction (baseline vs Prototype
Memory vs GT). La revision "correcto/incorrecto" se comunica aparte, por
numero de caso, usando cases.csv para ubicar cada uno.

Grupos generados, en este orden:
  1. FP_MEMORY            - FP estrictos que Prototype Memory TODAVIA genera
                             (memory_fp_<dataset>.csv). reviewed_07 primero
                             (los 57 casos), luego random_04.
  2. MISS_CLEAN            - cambios GT limpios (gt_suspect=False) que
                             Prototype Memory NO detecto (ninguna alerta
                             POSIBLE_CAMBIO/CAMBIO_CONFIRMADO en la ventana).
  3. CONFIRM_GT_SUSPECT     - CAMBIO_CONFIRMADO de Prototype Memory que
                             matchea un evento GT_SUSPECT.
  4. TP_CONTROL             - muestra de control de confirmaciones correctas
                             (GT limpio), 10-20 casos.

Para cada caso se arma una ventana de: hasta 6 observaciones previas, el
frame del evento/GT, la ventana candidata (hasta 3 obs, tomada del
DELAY_CONFIRMACION real de Prototype Memory) y hasta 10 posteriores. Dentro
de ese rango TEMPORAL se incluyen tambien las imagenes crudas intermedias
descartadas por los filtros del preprocesado (velocidad baja, sin
embedding/deteccion, o el primer frame del viaje que siempre se descarta),
para que la revision humana tenga el contexto completo.

Reutiliza integramente la logica de experiment_fp_reduction/common.py
(preprocesado, GT, matching por ventana) y evaluate.py (frame_idx), y el
patron de descarga de imagenes de build_miss_review_html.py (raiz del
repo). No modifica ningun CSV existente, ni algo_memory.py, ni
main_analisis_completo_v2.py. Solo relee los CSV crudos (para timestamps,
gs_path e imagenes intermedias descartadas) porque esa informacion no
estaba guardada en los CSV de detalle del experimento.

Salida (dentro de experiment_fp_reduction/manual_review/):
  - index.html
  - images/<trip_id>/<archivo>.jpeg
  - cases.csv

Uso:
    uv run python build_manual_review_html.py
"""
import os
import sys
import subprocess

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CSV_FILES, EXP_DIR, DATA_DIR, preprocess_trip, match_events_window, cosine_dist, load_emb, CONSTANTS

OUT_DIR = os.path.join(EXP_DIR, "reviews", "manual_review")
IMAGES_DIR = os.path.join(OUT_DIR, "images")
HTML_FILE = os.path.join(OUT_DIR, "index.html")
CASES_CSV = os.path.join(OUT_DIR, "cases.csv")

PRE_N = 6            # observaciones previas
POST_N = 10           # observaciones posteriores
CANDIDATE_MAX = 3      # tope de la ventana candidata
TP_CONTROL_TOTAL = 15  # tamano de la muestra de control (10-20 pedido)
RANDOM_SEED = 42

# Orden de prioridad de datasets (reviewed_07 primero: ahi estan los 57 FP
# estrictos que se piden revisar especialmente).
DATASET_ORDER = ["reviewed_07", "random_04"]

MIN_STATIONARY_SPEED = CONSTANTS['min_stationary_speed']

RAW_COLS = ['trip_id', 'asset_id', 'gs_path', 'timestamp', 'speed', 'identity_id', 'empty_cabin', 'embedding']


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _add_frame_idx(df):
    df = df.copy()
    df['frame_idx'] = df.groupby('ASSET_ID').cumcount()
    return df


def _identity_label(identity_id, empty_cabin):
    raw = str(identity_id)
    if raw not in ('', 'nan', 'None'):
        return raw
    if str(empty_cabin).strip().lower() in ('true', '1'):
        return 'CABINA VACIA'
    return 'SIN DETECCION'


def get_ts_from_gspath(gs_path):
    """basename crudo tiene formato <ts_ms>_<canal>.jpeg (sin prefijo de
    asset_id), a diferencia de 'image_file' en common.py que si lo tiene
    (asset_id_<ts_ms>_<canal>.jpeg); por eso aqui se toma parts[0]."""
    try:
        base = os.path.basename(str(gs_path))
        parts = base.split('_')
        if len(parts) >= 1:
            return int(parts[0]) // 1000
    except Exception:
        pass
    return None


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ''
    if isinstance(v, float):
        return f'{v:.{nd}f}'
    return str(v)


# ---------------------------------------------------------------------------
# Carga de los CSV ya generados por el experimento (sin recalibrar nada)
# ---------------------------------------------------------------------------

def load_dataset_bundle(name):
    gt = pd.read_csv(os.path.join(DATA_DIR, f"gt_events_{name}.csv"))
    gt['gt_suspect'] = gt['gt_suspect'].astype(bool)
    mem_detail = _add_frame_idx(pd.read_csv(os.path.join(DATA_DIR, f"memory_detail_{name}.csv")))
    base_detail = _add_frame_idx(pd.read_csv(os.path.join(DATA_DIR, f"baseline_detail_{name}.csv")))
    mem_fp = pd.read_csv(os.path.join(DATA_DIR, f"memory_fp_{name}.csv"))
    mem_tp = pd.read_csv(os.path.join(DATA_DIR, f"memory_tp_{name}.csv"))
    return {
        'name': name, 'gt': gt, 'mem_detail': mem_detail, 'base_detail': base_detail,
        'mem_fp': mem_fp, 'mem_tp': mem_tp,
    }


def _gt_lookup(gt_df):
    return {(r.trip_id, int(r.frame_idx)): r for r in gt_df.itertuples() if pd.notna(r.frame_idx)}


# ---------------------------------------------------------------------------
# Seleccion de casos (4 grupos, en orden)
# ---------------------------------------------------------------------------

def build_fp_memory_cases(bundles):
    cases = []
    for name in DATASET_ORDER:
        b = bundles[name]
        for r in b['mem_fp'].itertuples():
            cases.append({
                'dataset': name, 'trip_id': r.trip_id, 'case_type': 'FP_MEMORY',
                'gt_frame': None, 'alert_frame': int(r.alert_frame),
                'prev_identity': None, 'new_identity': None,
                'note': 'FP estricto: Prototype Memory confirmo CAMBIO_CONFIRMADO sin GT cercano (ventana=3)',
            })
    return cases


def build_miss_clean_cases(bundles):
    cases = []
    for name in DATASET_ORDER:
        b = bundles[name]
        gt_lu = _gt_lookup(b['gt'])
        for trip, g in b['gt'].groupby('trip_id'):
            clean_frames = g.loc[~g['gt_suspect'], 'frame_idx'].dropna().astype(int).tolist()
            if not clean_frames:
                continue
            d = b['mem_detail'][b['mem_detail']['ASSET_ID'] == trip]
            if d.empty:
                continue
            alert_frames = d.loc[d['DECISION_SISTEMA'].isin(['POSIBLE_CAMBIO', 'CAMBIO_CONFIRMADO']), 'frame_idx'].tolist()
            _, unmatched_gt, _ = match_events_window(clean_frames, alert_frames, window=3)
            for f in unmatched_gt:
                row = gt_lu.get((trip, int(f)))
                cases.append({
                    'dataset': name, 'trip_id': trip, 'case_type': 'MISS_CLEAN',
                    'gt_frame': int(f), 'alert_frame': None,
                    'prev_identity': getattr(row, 'prev_identity', None),
                    'new_identity': getattr(row, 'new_identity', None),
                    'note': 'Cambio GT limpio no detectado por Prototype Memory (ninguna alerta en ventana=3)',
                })
    return cases


def build_confirm_suspect_cases(bundles):
    cases = []
    for name in DATASET_ORDER:
        b = bundles[name]
        gt_lu = _gt_lookup(b['gt'])
        for r in b['mem_tp'].itertuples():
            gtrow = gt_lu.get((r.trip_id, int(r.gt_frame)))
            if gtrow is None or not bool(gtrow.gt_suspect):
                continue
            cases.append({
                'dataset': name, 'trip_id': r.trip_id, 'case_type': 'CONFIRM_GT_SUSPECT',
                'gt_frame': int(r.gt_frame), 'alert_frame': int(r.alert_frame),
                'prev_identity': gtrow.prev_identity, 'new_identity': gtrow.new_identity,
                'note': f'CAMBIO_CONFIRMADO matchea un evento GT_SUSPECT (delay={r.delay}, razon GT: {gtrow.reason})',
            })
    return cases


def build_tp_control_cases(bundles, total=TP_CONTROL_TOTAL, seed=RANDOM_SEED):
    pools = {}
    for name in DATASET_ORDER:
        b = bundles[name]
        gt_lu = _gt_lookup(b['gt'])
        clean_rows = []
        for r in b['mem_tp'].itertuples():
            gtrow = gt_lu.get((r.trip_id, int(r.gt_frame)))
            if gtrow is None or bool(gtrow.gt_suspect):
                continue
            clean_rows.append((r, gtrow))
        pools[name] = clean_rows

    n_avail = {name: len(pools[name]) for name in DATASET_ORDER}
    total_avail = sum(n_avail.values())
    take_total = min(total, total_avail)
    names_with_data = [n for n in DATASET_ORDER if n_avail[n] > 0]

    quotas = {}
    remaining = take_total
    for i, name in enumerate(names_with_data):
        if i == len(names_with_data) - 1:
            quotas[name] = remaining
        else:
            q = max(1, round(take_total * n_avail[name] / total_avail))
            q = min(q, n_avail[name], remaining)
            quotas[name] = q
            remaining -= q

    rng = np.random.default_rng(seed)
    cases = []
    for name in DATASET_ORDER:
        q = quotas.get(name, 0)
        if q <= 0:
            continue
        idxs = sorted(rng.choice(len(pools[name]), size=q, replace=False).tolist())
        for i in idxs:
            r, gtrow = pools[name][i]
            cases.append({
                'dataset': name, 'trip_id': r.trip_id, 'case_type': 'TP_CONTROL',
                'gt_frame': int(r.gt_frame), 'alert_frame': int(r.alert_frame),
                'prev_identity': gtrow.prev_identity, 'new_identity': gtrow.new_identity,
                'note': f'Control: confirmacion correcta vs GT limpio (delay={r.delay})',
            })
    return cases


def select_all_cases(bundles):
    cases = []
    cases += build_fp_memory_cases(bundles)
    cases += build_miss_clean_cases(bundles)
    cases += build_confirm_suspect_cases(bundles)
    cases += build_tp_control_cases(bundles)
    for i, c in enumerate(cases, 1):
        c['case_id'] = i
    return cases


# ---------------------------------------------------------------------------
# Datos crudos por viaje (para timestamps/gs_path y las imagenes crudas
# intermedias descartadas por el preprocesado, requeridas para la revision
# humana aunque el algoritmo no las haya usado).
# ---------------------------------------------------------------------------

def load_raw_rows_for_trips(name, trip_ids):
    """Lee el CSV crudo (grande) en chunks, quedandose solo con las columnas
    necesarias y los trips pedidos, para no cargar el dataset completo en
    memoria (los CSV base pesan >1.5GB)."""
    path = CSV_FILES[name]
    trip_ids = set(trip_ids)
    chunks = []
    print(f"  Leyendo CSV crudo de {name} (filtrando {len(trip_ids)} viajes necesarios)...")
    for chunk in pd.read_csv(path, usecols=RAW_COLS, chunksize=200_000, low_memory=False):
        sub = chunk[chunk['trip_id'].isin(trip_ids)]
        if len(sub):
            chunks.append(sub)
    if not chunks:
        return pd.DataFrame(columns=RAW_COLS)
    return pd.concat(chunks, ignore_index=True)


def build_embed_df(raw_df):
    """Mismo filtro/columna 'image_file' que common.load_raw_csv, pero
    aplicado solo al subconjunto de trips ya filtrado."""
    df = raw_df[raw_df['embedding'].notna() & (raw_df['embedding'].astype(str).str.strip() != '')].copy()
    df['image_file'] = df['asset_id'].astype(str) + "_" + df['gs_path'].apply(lambda p: os.path.basename(str(p)))
    return df


def build_trip_context(raw_df, embed_df, trip_id):
    df_trip_raw = raw_df[raw_df['trip_id'] == trip_id].copy()
    if df_trip_raw.empty:
        return None
    df_trip_raw['ts_seconds'] = df_trip_raw['gs_path'].apply(get_ts_from_gspath)
    df_trip_raw = df_trip_raw.dropna(subset=['ts_seconds']).copy()
    df_trip_raw['ts_seconds'] = df_trip_raw['ts_seconds'].astype(int)
    df_trip_raw['basename'] = df_trip_raw['gs_path'].apply(lambda p: os.path.basename(str(p)))
    df_trip_raw = df_trip_raw.sort_values('ts_seconds').reset_index(drop=True)

    df_trip_embed = embed_df[embed_df['trip_id'] == trip_id]
    df_clean = preprocess_trip(df_trip_embed)
    if df_clean is None:
        return None
    df_clean = df_clean.copy()
    df_clean['basename'] = df_clean['gs_path'].apply(lambda p: os.path.basename(str(p)))
    processed_keys = {b: i for i, b in enumerate(df_clean['basename'].tolist())}

    return {'df_clean': df_clean, 'raw_sorted': df_trip_raw, 'processed_keys': processed_keys}


def compute_dist_vs_memory(df_clean, mem_d, frame_idx):
    """Distancia del embedding del frame vs el centroide de embeddings
    previos del conductor vigente antes de este frame (misma logica que
    analyze_fp.py / algo_memory.py, aplicada puntualmente al frame pedido)."""
    if mem_d is None or frame_idx is None or frame_idx <= 0:
        return None
    if frame_idx >= len(mem_d) or frame_idx >= len(df_clean):
        return None
    prev_person = mem_d.iloc[frame_idx - 1]['PERSONA_ID']
    prev_mask = (mem_d['frame_idx'] < frame_idx) & (mem_d['PERSONA_ID'] == prev_person)
    prev_positions = mem_d.loc[prev_mask, 'frame_idx'].tolist()
    prev_embs = [load_emb(df_clean.iloc[i].get('embedding')) for i in prev_positions if i < len(df_clean)]
    prev_embs = [e for e in prev_embs if e is not None]
    cur_emb = load_emb(df_clean.iloc[frame_idx].get('embedding'))
    if not prev_embs or cur_emb is None:
        return None
    return cosine_dist(np.mean(prev_embs, axis=0), cur_emb)


# ---------------------------------------------------------------------------
# Ventana de frames por caso (procesados + crudos intermedios descartados)
# ---------------------------------------------------------------------------

def build_case_frames(case, trip_ctx, mem_d, base_d):
    df_clean = trip_ctx['df_clean']
    n = len(df_clean)
    gt_frame = case['gt_frame']
    alert_frame = case['alert_frame']

    anchors_raw = [f for f in (gt_frame, alert_frame) if f is not None]
    if not anchors_raw:
        return None
    anchors = [min(max(a, 0), n - 1) for a in anchors_raw]

    candidate_start = candidate_end = None
    if alert_frame is not None and mem_d is not None and alert_frame < len(mem_d):
        delay_raw = mem_d.iloc[alert_frame].get('DELAY_CONFIRMACION', 0)
        try:
            delay = int(delay_raw) if pd.notna(delay_raw) else 0
        except Exception:
            delay = 0
        delay = max(0, min(delay, CANDIDATE_MAX))
        candidate_start = max(0, alert_frame - delay)
        candidate_end = alert_frame

    bound_candidates = list(anchors)
    if candidate_start is not None:
        bound_candidates += [candidate_start, candidate_end]
    bounds_lo = min(bound_candidates)
    bounds_hi = max(bound_candidates)

    pre_start = max(0, bounds_lo - PRE_N)
    post_end = min(n - 1, bounds_hi + POST_N)

    ts_lo = int(df_clean.iloc[pre_start]['ts_seconds'])
    ts_hi = int(df_clean.iloc[post_end]['ts_seconds'])
    ts_bound_lo = int(df_clean.iloc[bounds_lo]['ts_seconds'])
    ts_bound_hi = int(df_clean.iloc[bounds_hi]['ts_seconds'])

    baseline_confirm_positions = set()
    memory_confirm_positions = set()
    if base_d is not None:
        span = base_d[(base_d['frame_idx'] >= pre_start) & (base_d['frame_idx'] <= post_end)]
        baseline_confirm_positions = set(span.loc[span['DECISION_SISTEMA'] == 'CAMBIO_CONFIRMADO', 'frame_idx'].tolist())
    if mem_d is not None:
        span = mem_d[(mem_d['frame_idx'] >= pre_start) & (mem_d['frame_idx'] <= post_end)]
        memory_confirm_positions = set(span.loc[span['DECISION_SISTEMA'] == 'CAMBIO_CONFIRMADO', 'frame_idx'].tolist())

    dist_mem_positions = set(range(candidate_start, candidate_end + 1)) if candidate_start is not None else set()
    if gt_frame is not None:
        dist_mem_positions.add(gt_frame)

    raw_sorted = trip_ctx['raw_sorted']
    processed_keys = trip_ctx['processed_keys']
    window_rows = raw_sorted[(raw_sorted['ts_seconds'] >= ts_lo) & (raw_sorted['ts_seconds'] <= ts_hi)]

    frames = []
    for row in window_rows.itertuples():
        frame_idx = processed_keys.get(row.basename)
        mem_info = {}
        base_info = {}
        dist_mem = None
        if frame_idx is not None:
            status = 'procesado'
            if mem_d is not None and frame_idx < len(mem_d):
                mem_info = mem_d.iloc[frame_idx].to_dict()
            if base_d is not None and frame_idx < len(base_d):
                base_info = base_d.iloc[frame_idx].to_dict()
            if frame_idx in dist_mem_positions:
                dist_mem = compute_dist_vs_memory(df_clean, mem_d, frame_idx)
        else:
            emb = getattr(row, 'embedding', None)
            if emb is None or (isinstance(emb, float) and pd.isna(emb)) or str(emb).strip() == '':
                status = 'descartado_sin_embedding'
            elif float(row.speed) < MIN_STATIONARY_SPEED:
                status = 'descartado_velocidad_baja'
            else:
                status = 'descartado_preprocesado'

        ts_val = row.ts_seconds
        if ts_val < ts_bound_lo:
            role = 'antes'
        elif ts_val > ts_bound_hi:
            role = 'despues'
        else:
            role = 'evento_candidato'

        frames.append({
            'frame_idx': frame_idx,
            'status': status,
            'role': role,
            'gs_path': row.gs_path,
            'basename': row.basename,
            'timestamp': row.timestamp,
            'speed': row.speed,
            'identity_label': _identity_label(row.identity_id, row.empty_cabin),
            'decision_baseline': base_info.get('DECISION_SISTEMA'),
            'decision_memory': mem_info.get('DECISION_SISTEMA'),
            'dist_frame_memory': mem_info.get('DISTANCIA_VISUAL'),
            'dist_frame_baseline': base_info.get('DISTANCIA_VISUAL'),
            'dist_vs_memoria': dist_mem,
            'p_fisica': mem_info.get('PROBABILIDAD_FISICA', base_info.get('PROBABILIDAD_FISICA')),
            'is_gt_frame': (frame_idx is not None and frame_idx == gt_frame),
            'is_baseline_confirm': (frame_idx is not None and frame_idx in baseline_confirm_positions),
            'is_memory_confirm': (frame_idx is not None and frame_idx in memory_confirm_positions),
        })

    baseline_summary = ('CAMBIO_CONFIRMADO en frame #%d' % min(baseline_confirm_positions, key=lambda p: abs(p - bounds_hi))
                         if baseline_confirm_positions else 'Sin CAMBIO_CONFIRMADO en la ventana')
    memory_summary = ('CAMBIO_CONFIRMADO en frame #%d' % min(memory_confirm_positions, key=lambda p: abs(p - bounds_hi))
                       if memory_confirm_positions else 'Sin CAMBIO_CONFIRMADO en la ventana')
    coincide = bool(baseline_confirm_positions) and bool(memory_confirm_positions) and any(
        abs(a - b) <= 3 for a in baseline_confirm_positions for b in memory_confirm_positions)

    return {
        'frames': frames,
        'baseline_summary': baseline_summary,
        'memory_summary': memory_summary,
        'coincide_baseline_memory': coincide,
        'candidate_start': candidate_start,
        'candidate_end': candidate_end,
    }


# ---------------------------------------------------------------------------
# Descarga de imagenes (mismo patron que build_miss_review_html.py)
# ---------------------------------------------------------------------------

def download_images(all_cases_data):
    by_trip = {}
    for cd in all_cases_data:
        trip_dir_name = str(cd['case']['trip_id'])
        for frame in cd['built']['frames']:
            gs_path = str(frame['gs_path'])
            if not gs_path.startswith('gs://'):
                continue
            filename = os.path.basename(gs_path)
            local_path = os.path.join(IMAGES_DIR, trip_dir_name, filename)
            frame['local_path'] = local_path
            if not os.path.exists(local_path):
                by_trip.setdefault(trip_dir_name, set()).add(gs_path)

    total = len(by_trip)
    for n, (trip_dir_name, gs_paths) in enumerate(by_trip.items(), 1):
        trip_dir = os.path.join(IMAGES_DIR, trip_dir_name)
        os.makedirs(trip_dir, exist_ok=True)
        cmd = ["gcloud", "storage", "cp", *sorted(gs_paths), trip_dir + "/"]
        print(f"   Descargando [{n}/{total}] {len(gs_paths)} imagenes (trip {trip_dir_name})...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"   AVISO: error descargando imagenes de {trip_dir_name}: {result.stderr[:300]}")


# ---------------------------------------------------------------------------
# Render HTML
# ---------------------------------------------------------------------------

CASE_TYPE_LABELS = {
    'FP_MEMORY': 'FP estricto (Prototype Memory)',
    'MISS_CLEAN': 'Cambio GT limpio no detectado',
    'CONFIRM_GT_SUSPECT': 'CAMBIO_CONFIRMADO vs GT_SUSPECT',
    'TP_CONTROL': 'Control (confirmacion correcta)',
}


def _frame_card_html(frame, out_dir):
    local_path = frame.get('local_path')
    rel_path = os.path.relpath(local_path, out_dir) if local_path else None
    if rel_path and os.path.exists(os.path.join(out_dir, rel_path)):
        img_tag = f'<img src="{rel_path}" loading="lazy" onclick="openLightbox(this.src)">'
    else:
        img_tag = '<div class="no-img">Sin imagen</div>'

    marks = []
    if frame['is_gt_frame']:
        marks.append('<span class="chip chip-gt">GT</span>')
    if frame['is_baseline_confirm']:
        marks.append('<span class="chip chip-base">BASELINE CONFIRMO</span>')
    if frame['is_memory_confirm']:
        marks.append('<span class="chip chip-mem">MEMORY CONFIRMO</span>')
    if len(marks) >= 2:
        marks.append('<span class="chip chip-match">COINCIDEN</span>')

    highlight_classes = ''
    if frame['is_gt_frame']:
        highlight_classes += ' mark-gt'
    if frame['is_baseline_confirm']:
        highlight_classes += ' mark-base'
    if frame['is_memory_confirm']:
        highlight_classes += ' mark-mem'

    pos_label = f"#{frame['frame_idx']}" if frame['frame_idx'] is not None else 'RAW'
    status_label = '' if frame['status'] == 'procesado' else f" ({frame['status'].replace('descartado_', 'descartado: ')})"

    return f'''
    <div class="frame-card role-{frame['role']}{highlight_classes}">
        {img_tag}
        <div class="chips">{''.join(marks)}</div>
        <div class="frame-meta">
            <div><b>{pos_label}</b>{status_label}</div>
            <div>Ts: {frame['timestamp']}</div>
            <div>Identidad: {frame['identity_label']}</div>
            <div>Decision base: {frame['decision_baseline'] or '-'}</div>
            <div>Decision mem: {frame['decision_memory'] or '-'}</div>
            <div>Dist frame-frame: {_fmt(frame['dist_frame_memory'] if frame['dist_frame_memory'] is not None else frame['dist_frame_baseline'])}</div>
            <div>Dist vs memoria: {_fmt(frame['dist_vs_memoria'])}</div>
            <div>P_fisica: {_fmt(frame['p_fisica'], 2)}</div>
            <div>Speed: {_fmt(frame['speed'], 1)}</div>
        </div>
    </div>'''


def render_case_html(cd, out_dir):
    case = cd['case']
    built = cd['built']
    frames = built['frames']
    cols = {'antes': [], 'evento_candidato': [], 'despues': []}
    for f in frames:
        cols[f['role']].append(_frame_card_html(f, out_dir))

    expected_change = ''
    if case['prev_identity'] is not None and case['new_identity'] is not None:
        expected_change = f"{case['prev_identity']} &rarr; {case['new_identity']}"
    else:
        expected_change = 'N/A'

    coincide_html = ''
    if built['coincide_baseline_memory']:
        coincide_html = '<span class="badge badge-ok">Baseline y Memory coinciden en la ventana</span>'

    return f'''
    <section class="case" data-case-id="{case['case_id']}" data-case-type="{case['case_type']}">
        <h2>Caso #{case['case_id']} <span class="case-type-badge type-{case['case_type']}">{CASE_TYPE_LABELS[case['case_type']]}</span></h2>
        <div class="case-header">
            <div>Trip ID: <b>{case['trip_id']}</b> ({case['dataset']})</div>
            <div>Cambio identity_id esperado: <b>{expected_change}</b></div>
            <div>Frame GT: <b>{case['gt_frame'] if case['gt_frame'] is not None else 'N/A'}</b></div>
            <div>Frame de alerta (Memory): <b>{case['alert_frame'] if case['alert_frame'] is not None else 'N/A'}</b></div>
            <div>Decision baseline: <b>{built['baseline_summary']}</b></div>
            <div>Decision Prototype Memory: <b>{built['memory_summary']}</b></div>
            {coincide_html}
        </div>
        <p class="case-note">{case['note']}</p>
        <div class="tramo">
            <div class="tramo-col"><h3>ANTES</h3><div class="frames-row">{''.join(cols['antes'])}</div></div>
            <div class="tramo-col highlight-col"><h3>EVENTO / CANDIDATO</h3><div class="frames-row">{''.join(cols['evento_candidato'])}</div></div>
            <div class="tramo-col"><h3>DESPUES</h3><div class="frames-row">{''.join(cols['despues'])}</div></div>
        </div>
    </section>'''


def render_html(all_cases_data, out_dir):
    case_sections = []
    index_items = []
    for cd in all_cases_data:
        case = cd['case']
        case_sections.append(render_case_html(cd, out_dir))
        index_items.append(
            f'<li data-case-id="{case["case_id"]}" data-case-type="{case["case_type"]}" '
            f'onclick="jumpToCase({case["case_id"]})">#{case["case_id"]} - {case["case_type"]} - trip {case["trip_id"]}</li>')

    counts = {}
    for cd in all_cases_data:
        t = cd['case']['case_type']
        counts[t] = counts.get(t, 0) + 1
    counts_html = ' | '.join(f'{CASE_TYPE_LABELS[t]}: <b>{n}</b>' for t, n in counts.items())

    filter_buttons = ['<button class="filter-btn active" data-type="ALL" onclick="filterCases(\'ALL\')">Todos</button>']
    for t in CASE_TYPE_LABELS:
        filter_buttons.append(f'<button class="filter-btn" data-type="{t}" onclick="filterCases(\'{t}\')">{CASE_TYPE_LABELS[t]}</button>')

    total = len(all_cases_data)

    html = f'''<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Revision Manual - experiment_fp_reduction</title>
<style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: sans-serif; background: #111; color: #eee; margin: 0; padding: 0; }}
    #topbar {{ position: sticky; top: 0; z-index: 500; background: #1b1b1b; border-bottom: 1px solid #333; padding: 10px 18px; display: flex; flex-wrap: wrap; align-items: center; gap: 12px; }}
    #topbar h1 {{ font-size: 16px; margin: 0; color: #fff; }}
    .filter-btn {{ background: #222; color: #ccc; border: 1px solid #444; border-radius: 14px; padding: 4px 12px; cursor: pointer; font-size: 12px; }}
    .filter-btn.active {{ background: #4a90d9; color: #fff; border-color: #4a90d9; }}
    #nav-controls {{ margin-left: auto; display: flex; align-items: center; gap: 8px; }}
    #nav-controls button {{ background: #333; color: #fff; border: none; border-radius: 6px; padding: 6px 14px; cursor: pointer; }}
    #counter {{ font-size: 13px; color: #ffb454; min-width: 110px; text-align: center; }}
    #layout {{ display: flex; }}
    #sidebar {{ width: 260px; flex-shrink: 0; background: #181818; border-right: 1px solid #333; height: calc(100vh - 54px); overflow-y: auto; position: sticky; top: 54px; padding: 8px; }}
    #sidebar ul {{ list-style: none; margin: 0; padding: 0; }}
    #sidebar li {{ padding: 6px 8px; font-size: 11px; border-bottom: 1px solid #262626; cursor: pointer; color: #aaa; }}
    #sidebar li:hover {{ background: #262626; color: #fff; }}
    #main {{ flex: 1; padding: 16px 24px; min-width: 0; }}
    .summary {{ background: #222; padding: 12px 18px; border-radius: 8px; margin-bottom: 20px; font-size: 13px; }}
    .case {{ background: #1b1b1b; border: 1px solid #333; border-radius: 8px; padding: 16px; margin-bottom: 28px; scroll-margin-top: 60px; }}
    .case h2 {{ margin-top: 0; color: #ffb454; font-size: 18px; }}
    .case-type-badge {{ font-size: 11px; padding: 3px 8px; border-radius: 10px; margin-left: 10px; background: #333; }}
    .type-FP_MEMORY {{ background: #6b2b2b; }}
    .type-MISS_CLEAN {{ background: #6b4d1f; }}
    .type-CONFIRM_GT_SUSPECT {{ background: #2b4d6b; }}
    .type-TP_CONTROL {{ background: #2b6b3d; }}
    .case-header {{ display: flex; flex-wrap: wrap; gap: 16px; font-size: 13px; background: #141414; padding: 8px 12px; border-radius: 6px; margin-bottom: 6px; }}
    .case-note {{ font-size: 12px; color: #999; margin: 4px 0 14px; }}
    .badge-ok {{ background: #2b6b3d; color: #d6ffe0; padding: 3px 10px; border-radius: 10px; font-size: 12px; }}
    .tramo {{ display: flex; gap: 14px; align-items: flex-start; }}
    .tramo-col {{ flex: 1; min-width: 0; }}
    .tramo-col h3 {{ font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid #333; padding-bottom: 4px; }}
    .highlight-col h3 {{ color: #ffb454; }}
    .frames-row {{ display: flex; flex-wrap: wrap; gap: 10px; }}
    .frame-card {{ width: 250px; border: 2px solid #444; border-radius: 6px; padding: 6px; background: #000; }}
    .frame-card.role-antes {{ border-color: #4a90d9; }}
    .frame-card.role-evento_candidato {{ border-color: #f5a623; }}
    .frame-card.role-despues {{ border-color: #7ed321; }}
    .frame-card.mark-gt {{ box-shadow: 0 0 0 3px #ff3b30; }}
    .frame-card.mark-base.mark-gt {{ box-shadow: 0 0 0 3px #ff3b30, 0 0 0 6px #ff9d0a; }}
    .frame-card.mark-mem.mark-gt {{ box-shadow: 0 0 0 3px #ff3b30, 0 0 0 6px #2ee66b; }}
    .frame-card.mark-base.mark-mem.mark-gt {{ box-shadow: 0 0 0 3px #ff3b30, 0 0 0 6px #ff9d0a, 0 0 0 9px #2ee66b; }}
    .frame-card.mark-base:not(.mark-gt) {{ box-shadow: 0 0 0 3px #ff9d0a; }}
    .frame-card.mark-mem:not(.mark-gt) {{ box-shadow: 0 0 0 3px #2ee66b; }}
    .frame-card.mark-base.mark-mem:not(.mark-gt) {{ box-shadow: 0 0 0 3px #ff9d0a, 0 0 0 6px #2ee66b; }}
    .frame-card img {{ width: 100%; height: 220px; object-fit: cover; border-radius: 4px; cursor: zoom-in; display: block; }}
    .no-img {{ width: 100%; height: 220px; display: flex; align-items: center; justify-content: center; background: #222; border-radius: 4px; font-size: 12px; color: #888; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 3px; margin-top: 4px; }}
    .chip {{ font-size: 9px; padding: 2px 5px; border-radius: 4px; color: #111; font-weight: bold; }}
    .chip-gt {{ background: #ff3b30; color: #fff; }}
    .chip-base {{ background: #ff9d0a; }}
    .chip-mem {{ background: #2ee66b; }}
    .chip-match {{ background: #fff; }}
    .frame-meta {{ font-size: 11px; margin-top: 6px; line-height: 1.4; }}
    #lightbox {{ display: none; position: fixed; z-index: 1000; inset: 0; background: rgba(0,0,0,0.94); align-items: center; justify-content: center; cursor: zoom-out; }}
    #lightbox.open {{ display: flex; }}
    #lightbox img {{ max-width: 96vw; max-height: 96vh; border-radius: 6px; box-shadow: 0 0 40px rgba(0,0,0,0.8); }}
</style>
</head>
<body>
    <div id="topbar">
        <h1>Revision Manual - experiment_fp_reduction</h1>
        {''.join(filter_buttons)}
        <div id="nav-controls">
            <button onclick="navigate(-1)">&larr; Anterior</button>
            <span id="counter">Caso 1 de {total}</span>
            <button onclick="navigate(1)">Siguiente &rarr;</button>
        </div>
    </div>
    <div id="layout">
        <div id="sidebar"><ul>{''.join(index_items)}</ul></div>
        <div id="main">
            <div class="summary">
                <p>Total de casos: <b>{total}</b> &nbsp;|&nbsp; {counts_html}</p>
                <p>Borde de color = rol (azul=antes, naranja=evento/candidato, verde=despues). Anillo rojo = frame GT,
                anillo naranja = donde el baseline confirmo, anillo verde = donde Prototype Memory confirmo. Si aparecen
                varios anillos en el mismo frame, coinciden (chip blanco "COINCIDEN").</p>
                <p><i>Solo visualizacion: no hay botones de clasificar. Anota los numeros de caso aparte usando cases.csv.
                Click en cualquier imagen para verla en grande (lightbox); click fuera, sobre la imagen/fondo, o ESC para cerrar.</i></p>
            </div>
            {''.join(case_sections)}
        </div>
    </div>
    <div id="lightbox" onclick="closeLightbox()">
        <img id="lightbox-img" src="">
    </div>
    <script>
        var currentType = 'ALL';

        function visibleCases() {{
            return Array.prototype.filter.call(document.querySelectorAll('.case'), function(el) {{
                return el.style.display !== 'none';
            }});
        }}

        function updateCounter() {{
            var vis = visibleCases();
            var idx = vis.findIndex(function(el) {{
                var r = el.getBoundingClientRect();
                return r.top >= -20;
            }});
            if (idx < 0) idx = 0;
            document.getElementById('counter').innerText = 'Caso ' + (idx + 1) + ' de ' + vis.length;
        }}

        function filterCases(type) {{
            currentType = type;
            document.querySelectorAll('.filter-btn').forEach(function(b) {{
                b.classList.toggle('active', b.getAttribute('data-type') === type);
            }});
            document.querySelectorAll('.case').forEach(function(el) {{
                var show = (type === 'ALL' || el.getAttribute('data-case-type') === type);
                el.style.display = show ? '' : 'none';
            }});
            document.querySelectorAll('#sidebar li').forEach(function(li) {{
                var show = (type === 'ALL' || li.getAttribute('data-case-type') === type);
                li.style.display = show ? '' : 'none';
            }});
            window.scrollTo(0, 0);
            updateCounter();
        }}

        function navigate(delta) {{
            var vis = visibleCases();
            if (!vis.length) return;
            var idx = vis.findIndex(function(el) {{ return el.getBoundingClientRect().top >= -20; }});
            if (idx < 0) idx = 0;
            var next = Math.min(vis.length - 1, Math.max(0, idx + delta));
            vis[next].scrollIntoView({{ behavior: 'smooth', block: 'start' }});
            setTimeout(updateCounter, 300);
        }}

        function jumpToCase(caseId) {{
            var el = document.querySelector('.case[data-case-id="' + caseId + '"]');
            if (el) {{
                el.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                setTimeout(updateCounter, 300);
            }}
        }}

        function openLightbox(src) {{
            document.getElementById('lightbox-img').src = src;
            document.getElementById('lightbox').classList.add('open');
        }}
        function closeLightbox() {{
            document.getElementById('lightbox').classList.remove('open');
        }}
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'Escape') closeLightbox();
        }});
        document.addEventListener('scroll', function() {{
            clearTimeout(window._scrollTimer);
            window._scrollTimer = setTimeout(updateCounter, 150);
        }});
        updateCounter();
    </script>
</body>
</html>'''

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html)


def write_cases_csv(all_cases_data):
    rows = []
    for cd in all_cases_data:
        c = cd['case']
        built = cd['built']
        rows.append({
            'case_id': c['case_id'],
            'dataset': c['dataset'],
            'trip_id': c['trip_id'],
            'case_type': c['case_type'],
            'gt_frame': c['gt_frame'],
            'alert_frame': c['alert_frame'],
            'prev_identity': c['prev_identity'],
            'new_identity': c['new_identity'],
            'decision_baseline': built['baseline_summary'],
            'decision_memory': built['memory_summary'],
            'coincide_baseline_memory': built['coincide_baseline_memory'],
            'n_frames_mostrados': len(built['frames']),
            'note': c['note'],
        })
    pd.DataFrame(rows).to_csv(CASES_CSV, index=False)


def verify_images(all_cases_data, out_dir):
    missing = []
    total = 0
    for cd in all_cases_data:
        for f in cd['built']['frames']:
            lp = f.get('local_path')
            if not lp:
                continue
            total += 1
            if not os.path.exists(lp):
                missing.append(lp)
    print(f"Verificacion de imagenes: {total - len(missing)}/{total} existen localmente.")
    if missing:
        print(f"AVISO: {len(missing)} imagenes referenciadas no se pudieron descargar/encontrar. Ejemplos:")
        for m in missing[:10]:
            print("   -", m)
    return missing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Cargando CSV ya generados por el experimento...")
    bundles = {name: load_dataset_bundle(name) for name in DATASET_ORDER}

    print("Seleccionando casos (4 grupos)...")
    cases = select_all_cases(bundles)
    print(f"  Total de casos seleccionados: {len(cases)}")
    for t in CASE_TYPE_LABELS:
        n = sum(1 for c in cases if c['case_type'] == t)
        print(f"    {t}: {n}")

    trip_ids_by_dataset = {name: set() for name in DATASET_ORDER}
    for c in cases:
        trip_ids_by_dataset[c['dataset']].add(c['trip_id'])

    trip_ctx_cache = {}
    mem_d_cache = {}
    base_d_cache = {}
    for name in DATASET_ORDER:
        needed = trip_ids_by_dataset[name]
        if not needed:
            continue
        raw_df = load_raw_rows_for_trips(name, needed)
        embed_df = build_embed_df(raw_df)
        b = bundles[name]
        for trip in needed:
            ctx = build_trip_context(raw_df, embed_df, trip)
            trip_ctx_cache[(name, trip)] = ctx
            mem_d_cache[(name, trip)] = b['mem_detail'][b['mem_detail']['ASSET_ID'] == trip].reset_index(drop=True)
            base_d_cache[(name, trip)] = b['base_detail'][b['base_detail']['ASSET_ID'] == trip].reset_index(drop=True)

    print("Armando ventanas de frames por caso...")
    all_cases_data = []
    skipped = 0
    for c in cases:
        key = (c['dataset'], c['trip_id'])
        ctx = trip_ctx_cache.get(key)
        if ctx is None:
            skipped += 1
            continue
        mem_d = mem_d_cache.get(key)
        base_d = base_d_cache.get(key)
        built = build_case_frames(c, ctx, mem_d, base_d)
        if built is None:
            skipped += 1
            continue
        all_cases_data.append({'case': c, 'built': built})
    if skipped:
        print(f"  AVISO: {skipped} casos se saltaron por falta de datos de contexto.")

    print("Descargando imagenes necesarias desde GCS...")
    os.makedirs(IMAGES_DIR, exist_ok=True)
    download_images(all_cases_data)

    print("Verificando imagenes locales...")
    verify_images(all_cases_data, OUT_DIR)

    print("Generando HTML...")
    render_html(all_cases_data, OUT_DIR)
    write_cases_csv(all_cases_data)
    print(f"Reporte generado en: {HTML_FILE}")
    print(f"Tabla de casos: {CASES_CSV}")


if __name__ == "__main__":
    main()
