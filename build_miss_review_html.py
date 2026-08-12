"""
REPORTE HTML DE CAMBIOS DE CONDUCTOR NO DETECTADOS (MISSES)
============================================================

Corre el pipeline de main_analisis_completo_v2.py, calcula los cambios reales
(GT = identity_id sostenido >= 2 frames) que el sistema NO marcó como
POSIBLE_CAMBIO/CAMBIO_CONFIRMADO, descarga de GCS las imágenes relevantes y
genera un HTML con la secuencia de frames alrededor de cada miss:

- Al menos 2 frames previos (identidad anterior), para saber quién manejaba antes.
- Todos los frames de transición entre ambas identidades (incluye cabina vacía
  si la hay).
- 3 frames posteriores (nueva identidad confirmada).

Esto sirve para revisar a mano si el miss es una falla real del algoritmo o un
problema del propio ground truth (identity_id mal etiquetado/parpadeante).

Uso:
    python build_miss_review_html.py
"""
import os
import subprocess

from main_analisis_completo_v2 import load_and_process, compute_gt_changes, INPUT_FILE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_miss_review")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
HTML_FILE = os.path.join(OUTPUT_DIR, "index.html")

PRE_FRAMES = 2   # mínimo de frames previos de la identidad anterior
POST_FRAMES = 3  # frames posteriores al cambio (nueva identidad)


def build_case_windows(out_df, misses):
    """Arma, para cada miss, la ventana de frames a revisar (previos + transición + posteriores)."""
    cases = []
    for miss in misses:
        asset_id = miss['asset_id']
        group = out_df[out_df['ASSET_ID'] == asset_id].reset_index(drop=True)
        n = len(group)

        prev_end = miss['prev_run_end']
        cur_start = miss['cur_start']

        pre_start = max(0, prev_end - (PRE_FRAMES - 1))
        post_end = min(n - 1, cur_start + POST_FRAMES - 1)

        frames = []
        for idx in range(pre_start, post_end + 1):
            row = group.iloc[idx]
            if idx <= prev_end:
                role = 'previo'
            elif idx < cur_start:
                role = 'transicion'
            else:
                role = 'nuevo'
            frames.append({
                'position': idx,
                'role': role,
                'identity_id': row.get('IDENTITY_ID', ''),
                'empty_cabin': row.get('EMPTY_CABIN', ''),
                'decision': row.get('DECISION_SISTEMA', ''),
                'timestamp': row.get('TIMESTAMP', ''),
                'speed': row.get('VELOCIDAD', ''),
                'gs_path': row.get('GS_PATH', ''),
                'is_change_point': idx == cur_start,
            })

        cases.append({
            'asset_id': asset_id,
            'prev_identity': miss['prev_identity'],
            'new_identity': miss['new_identity'],
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


def render_html(cases, gt_total, gt_detected, misses_count):
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

            highlight = ' change-point' if frame['is_change_point'] else ''
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
            <h2>Miss #{i} — Asset {esc(case['asset_id'])}</h2>
            <p>Cambio real no detectado: <b>{esc(case['prev_identity'])}</b> &rarr; <b>{esc(case['new_identity'])}</b></p>
            <div class="frames-row">{''.join(frame_cards)}</div>
        </section>''')

    html = f'''<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Revisión de Misses - Cambios de Conductor</title>
<style>
    body {{ font-family: sans-serif; background: #111; color: #eee; margin: 0; padding: 20px; }}
    h1 {{ color: #fff; }}
    .summary {{ background: #222; padding: 12px 18px; border-radius: 8px; margin-bottom: 24px; }}
    .case {{ background: #1b1b1b; border: 1px solid #333; border-radius: 8px; padding: 16px; margin-bottom: 24px; }}
    .case h2 {{ margin-top: 0; color: #ffb454; }}
    .frames-row {{ display: flex; flex-wrap: wrap; gap: 12px; }}
    .frame-card {{ width: 200px; border: 2px solid #444; border-radius: 6px; padding: 6px; background: #000; }}
    .frame-card.previo {{ border-color: #4a90d9; }}
    .frame-card.transicion {{ border-color: #f5a623; }}
    .frame-card.nuevo {{ border-color: #7ed321; }}
    .frame-card.change-point {{ border-color: #ff3b30; border-width: 3px; }}
    .frame-card img {{ width: 100%; height: 140px; object-fit: cover; border-radius: 4px; cursor: zoom-in; }}
    .no-img {{ width: 100%; height: 140px; display: flex; align-items: center; justify-content: center; background: #222; border-radius: 4px; font-size: 12px; color: #888; }}
    .frame-meta {{ font-size: 11px; margin-top: 6px; line-height: 1.4; }}
    #lightbox {{ display: none; position: fixed; z-index: 1000; inset: 0; background: rgba(0,0,0,0.9); align-items: center; justify-content: center; cursor: zoom-out; }}
    #lightbox.open {{ display: flex; }}
    #lightbox img {{ max-width: 90vw; max-height: 90vh; border-radius: 6px; box-shadow: 0 0 30px rgba(0,0,0,0.7); }}
</style>
</head>
<body>
    <h1>🔍 Revisión de Cambios de Conductor No Detectados</h1>
    <div class="summary">
        <p>Cambios reales (GT): <b>{gt_total}</b> | Detectados: <b>{gt_detected}</b> | <b style="color:#ff6b6b">No detectados (misses): {misses_count}</b></p>
        <p>Colores: <span style="color:#4a90d9">azul</span> = identidad previa, <span style="color:#f5a623">naranja</span> = transición, <span style="color:#7ed321">verde</span> = nueva identidad. Borde <span style="color:#ff3b30">rojo</span> = frame donde el GT dice que ya cambió pero el sistema no alertó.</p>
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
    misses = [c for c in gt_changes if not c['detected']]
    print(f"ℹ️ Cambios reales (GT): {len(gt_changes)} | Detectados: {len(gt_changes) - len(misses)} | Misses: {len(misses)}")

    cases = build_case_windows(out_df, misses)

    print("📥 Descargando imágenes necesarias desde GCS...")
    download_images(cases)

    print("🖨️ Generando HTML...")
    render_html(cases, len(gt_changes), len(gt_changes) - len(misses), len(misses))
    print(f"✅ Reporte generado en: {HTML_FILE}")


if __name__ == "__main__":
    main()
