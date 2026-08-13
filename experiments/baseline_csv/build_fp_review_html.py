"""
REPORTE HTML DE FALSOS POSITIVOS (ALERTAS SIN CAMBIO REAL)
============================================================

Corre el pipeline de main_analisis_completo_v2.py, calcula los cambios reales
(GT = identity_id sostenido >= 2 frames) y busca las alertas del sistema
que NO coinciden con un cambio real (los falsos positivos que bajan la
precisión). Por defecto solo mira los CAMBIO_CONFIRMADO (el compromiso firme
del sistema, el error más grave); los POSIBLE_CAMBIO son solo advertencias
para revisión y no se incluyen a menos que se agreguen a FILTER_DECISIONS.
Descarga de GCS las imágenes relevantes y genera un HTML con la secuencia de
frames alrededor de cada alerta para revisar a mano si es realmente un
falso positivo del algoritmo o si el GT tiene un cambio real no etiquetado.

Uso:
    python build_fp_review_html.py
"""
import os
import subprocess

from main_analisis_completo_v2 import load_and_process, compute_gt_changes, INPUT_FILE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "outputs_fp_confirmado_review")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
HTML_FILE = os.path.join(OUTPUT_DIR, "index.html")

PRE_FRAMES = 2   # frames de contexto antes de la alerta
POST_FRAMES = 2  # frames de contexto después de la alerta

# Qué decisiones del sistema se consideran falso positivo a revisar.
# Solo CAMBIO_CONFIRMADO: es el compromiso firme incorrecto (el error grave).
# Agregar 'POSIBLE_CAMBIO' si se quiere volver a revisar las advertencias.
FILTER_DECISIONS = ('CAMBIO_CONFIRMADO',)


def compute_false_positives(out_df, gt_changes):
    """Encuentra, por asset, las posiciones donde el sistema alertó
    (según FILTER_DECISIONS) pero no corresponden a un cambio real del GT
    (mismo criterio que la tabla de métricas en main())."""
    tp_positions_by_asset = {}
    for c in gt_changes:
        if c['detected']:
            tp_positions_by_asset.setdefault(c['asset_id'], set()).add(c['cur_start'])

    fps = []
    for asset_id, group in out_df.groupby('ASSET_ID', sort=False):
        decision = group['DECISION_SISTEMA'].tolist()
        tp_positions = tp_positions_by_asset.get(asset_id, set())
        for pos, dec in enumerate(decision):
            if dec in FILTER_DECISIONS and pos not in tp_positions:
                fps.append({'asset_id': asset_id, 'position': pos, 'decision': dec})
    return fps


def build_case_windows(out_df, fps):
    """Arma, para cada falso positivo, la ventana de frames a revisar
    (contexto antes + la alerta + contexto después)."""
    cases = []
    for fp in fps:
        asset_id = fp['asset_id']
        group = out_df[out_df['ASSET_ID'] == asset_id].reset_index(drop=True)
        n = len(group)

        pos = fp['position']
        win_start = max(0, pos - PRE_FRAMES)
        win_end = min(n - 1, pos + POST_FRAMES)

        frames = []
        for idx in range(win_start, win_end + 1):
            row = group.iloc[idx]
            role = 'alerta' if idx == pos else ('previo' if idx < pos else 'posterior')
            frames.append({
                'position': idx,
                'role': role,
                'identity_id': row.get('IDENTITY_ID', ''),
                'empty_cabin': row.get('EMPTY_CABIN', ''),
                'decision': row.get('DECISION_SISTEMA', ''),
                'timestamp': row.get('TIMESTAMP', ''),
                'speed': row.get('VELOCIDAD', ''),
                'gs_path': row.get('GS_PATH', ''),
                'is_alert_point': idx == pos,
            })

        cases.append({
            'asset_id': asset_id,
            'alert_position': pos,
            'decision': fp['decision'],
            'frames': frames,
        })
    return cases


def download_images(cases):
    """Descarga de GCS (una vez por asset) las imágenes únicas necesarias para el reporte."""
    by_asset = {}
    for case in cases:
        asset_dir = os.path.join(IMAGES_DIR, str(case['asset_id']))
        for frame in case['frames']:
            gs_path = str(frame['gs_path'])
            if not gs_path.startswith('gs://'):
                continue
            filename = os.path.basename(gs_path)
            local_path = os.path.join(asset_dir, filename)
            frame['local_path'] = local_path
            if not os.path.exists(local_path):
                by_asset.setdefault(str(case['asset_id']), set()).add(gs_path)

    total_assets = len(by_asset)
    for n, (asset_id, gs_paths) in enumerate(by_asset.items(), 1):
        asset_dir = os.path.join(IMAGES_DIR, asset_id)
        os.makedirs(asset_dir, exist_ok=True)
        cmd = ["gcloud", "storage", "cp", *sorted(gs_paths), asset_dir + "/"]
        print(f"   📥 [{n}/{total_assets}] Descargando {len(gs_paths)} imágenes (asset {asset_id})...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"   ⚠️ Error descargando imágenes de {asset_id}: {result.stderr[:300]}")


def render_html(cases, gt_total, gt_detected, fp_count):
    def esc(v):
        return str(v) if v is not None else ''

    rows_html = []
    for i, case in enumerate(cases, 1):
        frame_cards = []
        for frame in case['frames']:
            local_path = frame.get('local_path')
            rel_path = os.path.relpath(local_path, OUTPUT_DIR) if local_path else None
            if rel_path and os.path.exists(os.path.join(OUTPUT_DIR, rel_path)):
                img_tag = f'<img src="{rel_path}" loading="lazy" onclick="openLightbox(this.src)">'
            else:
                img_tag = '<div class="no-img">Sin imagen</div>'

            raw_identity = str(frame['identity_id'])
            if raw_identity not in ('', 'nan'):
                identity_label = raw_identity
            elif str(frame['empty_cabin']).strip().lower() in ('true', '1'):
                identity_label = 'CABINA VACÍA'
            else:
                identity_label = 'SIN DETECCIÓN'

            highlight = ' alert-point' if frame['is_alert_point'] else ''
            frame_cards.append(f'''
                <div class="frame-card {frame['role']}{highlight}">
                    {img_tag}
                    <div class="frame-meta">
                        <div><b>#{frame['position']}</b> ({frame['role']})</div>
                        <div>Identidad: {esc(identity_label)}</div>
                        <div>Decisión: {esc(frame['decision'])}</div>
                        <div>Ts: {esc(frame['timestamp'])} | Vel: {esc(frame['speed'])}</div>
                    </div>
                </div>''')

        rows_html.append(f'''
        <section class="case">
            <h2>FP #{i} — Asset {esc(case['asset_id'])}</h2>
            <p>Alerta del sistema sin cambio real de GT: <b>{esc(case['decision'])}</b> en frame #{case['alert_position']}</p>
            <div class="frames-row">{''.join(frame_cards)}</div>
        </section>''')

    html = f'''<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Revisión de Falsos Positivos (CAMBIO_CONFIRMADO) - Cambios de Conductor</title>
<style>
    body {{ font-family: sans-serif; background: #111; color: #eee; margin: 0; padding: 20px; }}
    h1 {{ color: #fff; }}
    .summary {{ background: #222; padding: 12px 18px; border-radius: 8px; margin-bottom: 24px; }}
    .case {{ background: #1b1b1b; border: 1px solid #333; border-radius: 8px; padding: 16px; margin-bottom: 24px; }}
    .case h2 {{ margin-top: 0; color: #ffb454; }}
    .frames-row {{ display: flex; flex-wrap: wrap; gap: 12px; }}
    .frame-card {{ width: 200px; border: 2px solid #444; border-radius: 6px; padding: 6px; background: #000; }}
    .frame-card.previo {{ border-color: #4a90d9; }}
    .frame-card.alerta {{ border-color: #f5a623; }}
    .frame-card.posterior {{ border-color: #7ed321; }}
    .frame-card.alert-point {{ border-color: #ff3b30; border-width: 3px; }}
    .frame-card img {{ width: 100%; height: 140px; object-fit: cover; border-radius: 4px; cursor: zoom-in; }}
    .no-img {{ width: 100%; height: 140px; display: flex; align-items: center; justify-content: center; background: #222; border-radius: 4px; font-size: 12px; color: #888; }}
    .frame-meta {{ font-size: 11px; margin-top: 6px; line-height: 1.4; }}
    #lightbox {{ display: none; position: fixed; z-index: 1000; inset: 0; background: rgba(0,0,0,0.9); align-items: center; justify-content: center; cursor: zoom-out; }}
    #lightbox.open {{ display: flex; }}
    #lightbox img {{ max-width: 90vw; max-height: 90vh; border-radius: 6px; box-shadow: 0 0 30px rgba(0,0,0,0.7); }}
</style>
</head>
<body>
    <h1>🔍 Revisión de Falsos Positivos (CAMBIO_CONFIRMADO sin cambio real)</h1>
    <div class="summary">
        <p>Cambios reales (GT): <b>{gt_total}</b> | Detectados: <b>{gt_detected}</b> | <b style="color:#ff6b6b">Falsos Positivos: {fp_count}</b></p>
        <p>Colores: <span style="color:#4a90d9">azul</span> = contexto previo, <span style="color:#f5a623">naranja</span> = frame de la alerta, <span style="color:#7ed321">verde</span> = contexto posterior. Borde <span style="color:#ff3b30">rojo</span> = frame exacto donde el sistema alertó sin que el GT tenga un cambio real.</p>
        <p><i>Haz click en cualquier imagen para verla en grande. Click de nuevo (o ESC) para cerrar.</i></p>
    </div>
    {''.join(rows_html)}
    <div id="lightbox" onclick="closeLightbox()">
        <img id="lightbox-img" src="">
    </div>
    <script>
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
    </script>
</body>
</html>'''

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(HTML_FILE, 'w', encoding='utf-8') as f:
        f.write(html)


def main():
    print("🚀 Corriendo pipeline para obtener el detalle frame-a-frame...")
    out_df = load_and_process(INPUT_FILE)
    if out_df is None:
        print("❌ No se pudo generar el detalle. Abortando.")
        return

    gt_changes = compute_gt_changes(out_df)
    gt_total = len(gt_changes)
    gt_detected = sum(1 for c in gt_changes if c['detected'])

    fps = compute_false_positives(out_df, gt_changes)
    print(f"ℹ️ Cambios reales (GT): {gt_total} | Detectados: {gt_detected} | Falsos Positivos: {len(fps)}")

    cases = build_case_windows(out_df, fps)

    print("📥 Descargando imágenes necesarias desde GCS...")
    download_images(cases)

    print("🖨️ Generando HTML...")
    render_html(cases, gt_total, gt_detected, len(fps))
    print(f"✅ Reporte generado en: {HTML_FILE}")


if __name__ == "__main__":
    main()
