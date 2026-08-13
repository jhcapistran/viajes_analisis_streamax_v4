"""
HTML SIMPLE DE REVISION VISUAL - SOLO FP_MEMORY de reviewed_07
================================================================

Objetivo unico: revisar visualmente los 60 FP estrictos restantes de
Prototype Memory en `reviewed_07` (memory_fp_reviewed_07.csv) y decidir a
mano si son FP reales o el GT esta mal. Un caso a la vez, solo las
observaciones faciales validas (las que ve el algoritmo, sin frames
crudos/descartados), maximo 9 imagenes por caso (4 antes + alerta + 4
despues).

No recalibra ni corre de nuevo el algoritmo: solo relee
memory_fp_reviewed_07.csv (ya generado) + el CSV crudo de reviewed_07 (para
imagenes, timestamps e identity_id), reutilizando preprocess_trip de
common.py para reproducir EXACTAMENTE la misma secuencia de frames que vio
el algoritmo (mismo indice de frame que alert_frame).

Salida (dentro de experiment_fp_reduction/fp_simple_review/):
  - index.html (un solo archivo, con los 60 casos embebidos en JS)
  - images/<trip_id>/<archivo>.jpeg

Uso:
    uv run python build_fp_simple_review.py
"""
import os
import sys
import json
import subprocess

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CSV_FILES, EXP_DIR, DATA_DIR, load_raw_csv, preprocess_trip

DATASET = "reviewed_07"
OUT_DIR = os.path.join(EXP_DIR, "reviews", "fp_simple_review")
IMAGES_DIR = os.path.join(OUT_DIR, "images")
HTML_FILE = os.path.join(OUT_DIR, "index.html")
N_BEFORE = 4
N_AFTER = 4


def identity_label(identity_id, empty_cabin):
    raw = str(identity_id)
    if raw not in ('', 'nan', 'None'):
        return raw
    if str(empty_cabin).strip().lower() in ('true', '1'):
        return 'CABINA VACIA'
    return 'SIN DETECCION'


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)

    fp_path = os.path.join(DATA_DIR, f"memory_fp_{DATASET}.csv")
    fp_df = pd.read_csv(fp_path)
    print(f"Casos FP_MEMORY leidos de {fp_path}: {len(fp_df)}")
    assert len(fp_df) == 60, f"Se esperaban 60 casos, hay {len(fp_df)}"

    print(f"Cargando CSV crudo de {DATASET}...")
    raw = load_raw_csv(CSV_FILES[DATASET])

    cases = []
    for case_id, row in enumerate(fp_df.itertuples(), start=1):
        trip_id = row.trip_id
        alert_frame = int(row.alert_frame)

        df_trip = raw[raw['trip_id'] == trip_id]
        df_clean = preprocess_trip(df_trip)
        if df_clean is None:
            print(f"  AVISO: caso {case_id} (trip {trip_id}) sin df_clean, se omite")
            continue
        df_clean = df_clean.reset_index(drop=True)
        df_clean['basename'] = df_clean['gs_path'].apply(lambda p: os.path.basename(str(p)))

        lo = max(0, alert_frame - N_BEFORE)
        hi = min(len(df_clean) - 1, alert_frame + N_AFTER)

        frames = []
        for idx in range(lo, hi + 1):
            r = df_clean.iloc[idx]
            frames.append({
                'pos': idx - alert_frame,
                'is_alert': idx == alert_frame,
                'identity_label': identity_label(r['identity_id'], r.get('empty_cabin')),
                'gs_path': str(r['gs_path']),
                'basename': r['basename'],
            })

        assert len(frames) <= 9, f"Caso {case_id} tiene {len(frames)} frames (> 9)"

        cases.append({
            'case_id': case_id,
            'trip_id': str(trip_id),
            'alert_frame': alert_frame,
            'frames': frames,
        })

    print(f"Total de casos armados: {len(cases)}")
    assert len(cases) == 60, f"Se esperaban 60 casos armados, hay {len(cases)}"
    assert all(len(c['frames']) <= 9 for c in cases), "Algun caso tiene mas de 9 imagenes"

    # -----------------------------------------------------------------
    # Descarga de imagenes (agrupada por trip, mismo patron que antes)
    # -----------------------------------------------------------------
    print("Descargando imagenes necesarias desde GCS...")
    by_trip = {}
    for c in cases:
        trip_dir_name = c['trip_id']
        for fr in c['frames']:
            gs_path = fr['gs_path']
            if not gs_path.startswith('gs://'):
                continue
            local_path = os.path.join(IMAGES_DIR, trip_dir_name, fr['basename'])
            fr['local_path'] = os.path.relpath(local_path, OUT_DIR)
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

    missing = 0
    for c in cases:
        for fr in c['frames']:
            lp = fr.get('local_path')
            if lp and not os.path.exists(os.path.join(OUT_DIR, lp)):
                missing += 1
    print(f"Verificacion de imagenes: {'todas existen' if missing == 0 else f'{missing} faltantes'}")

    # -----------------------------------------------------------------
    # Render HTML (un solo archivo, todo embebido)
    # -----------------------------------------------------------------
    cases_json = json.dumps(cases, ensure_ascii=False)
    html = HTML_TEMPLATE.replace("__CASES_JSON__", cases_json).replace("__N_CASES__", str(len(cases)))

    with open(HTML_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nReporte generado en: {HTML_FILE}")

    # -----------------------------------------------------------------
    # Verificaciones finales explicitas
    # -----------------------------------------------------------------
    print("\n=== Verificaciones ===")
    print(f"1. Cantidad de casos == 60: {len(cases) == 60} ({len(cases)})")
    print(f"2. Max frames por caso <= 9: {all(len(c['frames']) <= 9 for c in cases)}")
    print("3. Click para ampliar: implementado via <img onclick=openLightbox> + lightbox JS (ver index.html)")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Revision FP estrictos - Prototype Memory (reviewed_07)</title>
<style>
    * { box-sizing: border-box; }
    body { font-family: sans-serif; background: #111; color: #eee; margin: 0; padding: 0; }
    #topbar { background: #1b1b1b; border-bottom: 1px solid #333; padding: 14px 20px; text-align: center; }
    #topbar .counter { font-size: 20px; font-weight: bold; color: #ffb454; }
    #topbar .meta { font-size: 14px; color: #ccc; margin-top: 6px; }
    #topbar .instruction { font-size: 15px; color: #fff; margin-top: 10px; font-weight: bold; }
    #main { padding: 24px; max-width: 1500px; margin: 0 auto; }
    .columns { display: flex; gap: 24px; justify-content: center; align-items: flex-start; flex-wrap: wrap; }
    .col { flex: 1; min-width: 260px; }
    .col h3 { text-align: center; font-size: 13px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .frames-row { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; }
    .frame-card { width: 220px; border: 3px solid #444; border-radius: 8px; padding: 8px; background: #000; text-align: center; }
    .frame-card.alert { border: 6px solid #ff3b30; }
    .frame-card img { width: 100%; height: 220px; object-fit: cover; border-radius: 4px; cursor: zoom-in; display: block; }
    .alert-tag { color: #ff3b30; font-weight: bold; font-size: 13px; margin-top: 6px; }
    .frame-meta { font-size: 13px; margin-top: 8px; line-height: 1.4; }
    .frame-meta .pos { color: #888; }
    .frame-meta .id { color: #fff; font-weight: bold; }
    #nav { display: flex; justify-content: center; gap: 20px; margin-top: 30px; }
    #nav button { background: #333; color: #fff; border: none; border-radius: 8px; padding: 16px 32px; font-size: 18px; cursor: pointer; }
    #nav button:hover { background: #4a90d9; }
    #nav button:disabled { opacity: 0.3; cursor: default; }
    #lightbox { display: none; position: fixed; z-index: 1000; inset: 0; background: rgba(0,0,0,0.95); align-items: center; justify-content: center; cursor: zoom-out; }
    #lightbox.open { display: flex; }
    #lightbox img { max-width: 96vw; max-height: 96vh; border-radius: 6px; box-shadow: 0 0 40px rgba(0,0,0,0.8); }
    .no-img { width: 100%; height: 220px; display: flex; align-items: center; justify-content: center; background: #222; border-radius: 4px; font-size: 12px; color: #888; }
</style>
</head>
<body>
    <div id="topbar">
        <div class="counter" id="counter">Caso 1 / __N_CASES__</div>
        <div class="meta" id="meta"></div>
        <div class="instruction">Prototype Memory indico CAMBIO DE CONDUCTOR en la imagen marcada en rojo.</div>
    </div>
    <div id="main">
        <div id="case-content"></div>
        <div id="nav">
            <button id="prev-btn" onclick="go(-1)">&larr; Caso anterior</button>
            <button id="next-btn" onclick="go(1)">Caso siguiente &rarr;</button>
        </div>
    </div>
    <div id="lightbox" onclick="closeLightbox()">
        <img id="lightbox-img" src="">
    </div>

<script>
const CASES = __CASES_JSON__;
let current = 0;

function frameCardHtml(fr) {
    const cls = fr.is_alert ? "frame-card alert" : "frame-card";
    const imgTag = fr.local_path
        ? `<img src="${fr.local_path}" onclick="openLightbox('${fr.local_path}')">`
        : `<div class="no-img">sin imagen</div>`;
    const alertTag = fr.is_alert ? `<div class="alert-tag">CAMBIO_CONFIRMADO</div>` : "";
    return `
    <div class="${cls}">
        ${imgTag}
        ${alertTag}
        <div class="frame-meta">
            <div class="pos">pos ${fr.pos > 0 ? "+" + fr.pos : fr.pos}</div>
            <div class="id">${fr.identity_label}</div>
        </div>
    </div>`;
}

function render() {
    const c = CASES[current];
    document.getElementById("counter").textContent = `Caso ${current + 1} / ${CASES.length}`;
    document.getElementById("meta").textContent = `trip_id: ${c.trip_id}  |  alert_frame: ${c.alert_frame}`;

    const antes = c.frames.filter(f => f.pos < 0);
    const alerta = c.frames.filter(f => f.pos === 0);
    const despues = c.frames.filter(f => f.pos > 0);

    document.getElementById("case-content").innerHTML = `
        <div class="columns">
            <div class="col"><h3>Antes</h3><div class="frames-row">${antes.map(frameCardHtml).join("")}</div></div>
            <div class="col"><h3>Alerta</h3><div class="frames-row">${alerta.map(frameCardHtml).join("")}</div></div>
            <div class="col"><h3>Despues</h3><div class="frames-row">${despues.map(frameCardHtml).join("")}</div></div>
        </div>`;

    document.getElementById("prev-btn").disabled = current === 0;
    document.getElementById("next-btn").disabled = current === CASES.length - 1;
}

function go(delta) {
    const next = current + delta;
    if (next < 0 || next >= CASES.length) return;
    current = next;
    render();
}

function openLightbox(src) {
    document.getElementById("lightbox-img").src = src;
    document.getElementById("lightbox").classList.add("open");
}
function closeLightbox() {
    document.getElementById("lightbox").classList.remove("open");
}

document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { closeLightbox(); return; }
    if (document.getElementById("lightbox").classList.contains("open")) return;
    if (e.key === "ArrowLeft") go(-1);
    if (e.key === "ArrowRight") go(1);
});
document.getElementById("lightbox-img").addEventListener("click", (e) => e.stopPropagation());

render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
