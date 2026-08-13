"""
Revision manual visual de los 45 FP estrictos de CAMBIO_CONFIRMADO de la
config congelada "3-de-4" (E_w4_s3_c035_d085_p06, ver winner_config.json)
sobre DEV.

Esto NO es un experimento nuevo: no cambia el algoritmo (algo_v1_param.py
sigue intacto; se usa una copia instrumentada -- algo_v1_instrumented.py --
que agrega SOLO campos de diagnostico, misma logica de decision) y NO toca
TEST ni splits.csv.

Pasos:
  1. Correr la config congelada sobre TODO DEV (memoria: emb ya
     precomputados via precompute.py) y extraer, con evaluate.py, SOLO los
     CAMBIO_CONFIRMADO que son FP estrictos (no matchean ningun evento GT).
  2. Verificar que sean exactamente 45 (igual a full_dev_results.csv).
  3. Para cada FP: reconstruir frames antes / candidatos / confirmacion /
     despues (misma logica de preprocesado que ve el algoritmo), con
     distancia a memoria, P_fisica, velocidad y decision por frame.
     identity_id NUNCA se muestra (para no sesgar la revision).
  4. Descargar imagenes (gcloud storage cp) y generar un HTML standalone
     con controles de revision (misma persona / cambio real /
     indeterminado + causa) que guarda respuestas en localStorage y permite
     exportar CSV.

Uso:
    uv run python build_frozen_fp_review.py
"""
import os
import sys
import json
import subprocess

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prototype_memory"))

from common import CSV_FILES, EXP_DIR, DATA_DIR, load_raw_csv, preprocess_trip  # noqa: E402
from evaluate import evaluate  # noqa: E402
from precompute import precompute_trips  # noqa: E402
from algo_v1_instrumented import run_v1_instrumented  # noqa: E402

TUNING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_NAME = "E_w4_s3_c035_d085_p06"
FROZEN_PARAMS = dict(
    candidate_window=4,
    candidate_min_support=3,
    coherence_threshold=0.35,
    min_avg_dist_confirm=0.85,
    min_phys_confirm=0.6,
)
EXPECTED_FP = 45

OUT_DIR = os.path.join(EXP_DIR, "reviews", "frozen_fp_review_45")
IMAGES_DIR = os.path.join(OUT_DIR, "images")
HTML_FILE = os.path.join(OUT_DIR, "index.html")

N_BEFORE = 3
N_AFTER = 3

CAUSAS = ["pose", "lateral", "lentes", "IR/iluminacion", "oclusion", "blur", "distancia a camara", "otro"]


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # -----------------------------------------------------------------
    # 1. Correr la config congelada sobre TODO DEV
    # -----------------------------------------------------------------
    splits = pd.read_csv(os.path.join(EXP_DIR, "splits", "splits.csv"))
    dev = splits[splits["split"] == "DEV"].copy()
    trip_ids_by_dataset = {
        name: dev.loc[dev["dataset"] == name, "trip_id"].tolist()
        for name in dev["dataset"].unique()
    }
    trip_to_dataset = dict(zip(dev["trip_id"], dev["dataset"]))
    n_dev = len(dev)
    print(f"Precomputando DEV ({n_dev} viajes)...")
    cache = precompute_trips(trip_ids_by_dataset)
    print(f"  {len(cache)} viajes precomputados")

    print(f"Corriendo config congelada {CONFIG_NAME} sobre DEV...")
    records_by_trip = {}
    all_records = []
    for trip_id, frames in cache.items():
        recs = run_v1_instrumented(frames, FROZEN_PARAMS, trip_id=trip_id)
        records_by_trip[trip_id] = recs
        all_records.extend(recs)
    detail_df = pd.DataFrame(all_records)

    gt_parts = []
    for name in dev["dataset"].unique():
        gt = pd.read_csv(os.path.join(DATA_DIR, f"gt_events_{name}.csv"))
        gt_parts.append(gt[gt["trip_id"].isin(set(cache.keys()))])
    gt_df = pd.concat(gt_parts, ignore_index=True)
    gt_df["gt_suspect"] = gt_df["gt_suspect"].astype(bool)

    m, tp_df, fp_df = evaluate(detail_df, gt_df, window=3, label=CONFIG_NAME)
    print(f"CAMBIO_CONFIRMADO totales: {m['cambio_confirmado_total']}")
    print(f"CAMBIO_CONFIRMADO correctos: {m['cambio_confirmado_correctos']}")
    print(f"FP estrictos: {m['cambio_confirmado_incorrectos_fp_estrictos']}")

    # -----------------------------------------------------------------
    # 2. Verificar que sean exactamente 45
    # -----------------------------------------------------------------
    assert len(fp_df) == EXPECTED_FP, (
        f"Se esperaban {EXPECTED_FP} FP estrictos, se obtuvieron {len(fp_df)}. "
        "No se continua: revisar que la config y el DEV no hayan cambiado."
    )
    print(f"Verificado: {len(fp_df)} FP estrictos (esperado {EXPECTED_FP}).")

    # -----------------------------------------------------------------
    # 3. Armar casos (frames antes/candidatos/confirmacion/despues)
    # -----------------------------------------------------------------
    raw_cache = {}

    def get_df_clean(trip_id):
        dataset = trip_to_dataset[trip_id]
        if dataset not in raw_cache:
            print(f"Cargando CSV crudo de {dataset}...")
            raw_cache[dataset] = load_raw_csv(CSV_FILES[dataset])
        raw = raw_cache[dataset]
        df_trip = raw[raw["trip_id"] == trip_id]
        df_clean = preprocess_trip(df_trip)
        if df_clean is None:
            return None
        df_clean = df_clean.reset_index(drop=True)
        df_clean["basename"] = df_clean["gs_path"].apply(lambda p: os.path.basename(str(p)))
        return df_clean

    cases = []
    for case_id, row in enumerate(fp_df.itertuples(), start=1):
        trip_id = row.trip_id
        alert_frame = int(row.alert_frame)
        dataset = trip_to_dataset[trip_id]

        recs = records_by_trip[trip_id]
        confirm_rec = recs[alert_frame]
        assert confirm_rec["DECISION_SISTEMA"] == "CAMBIO_CONFIRMADO", (
            f"Caso {case_id} (trip {trip_id}): frame {alert_frame} no es CAMBIO_CONFIRMADO"
        )
        candidate_frames = confirm_rec["CANDIDATE_FRAMES"] or [alert_frame]
        coherent_frames = set(confirm_rec["COHERENT_FRAMES"] or [])

        df_clean = get_df_clean(trip_id)
        if df_clean is None:
            print(f"  AVISO: caso {case_id} (trip {trip_id}) sin df_clean, se omite")
            continue

        lo = max(0, min(candidate_frames) - N_BEFORE)
        hi = min(len(df_clean) - 1, alert_frame + N_AFTER)

        frames_out = []
        for idx in range(lo, hi + 1):
            r = df_clean.iloc[idx]
            rec = recs[idx]
            if idx == alert_frame:
                role = "confirmado"
            elif idx in candidate_frames:
                role = "candidato"
            elif idx < min(candidate_frames):
                role = "antes"
            else:
                role = "despues"
            frames_out.append({
                "pos": idx - alert_frame,
                "role": role,
                "is_coherent": idx in coherent_frames,
                "ts_seconds": int(rec["TS_SECONDS"]) if rec.get("TS_SECONDS") is not None else None,
                "dist_mem": round(float(rec["DIST_MEM"]), 3) if rec.get("DIST_MEM") is not None else None,
                "p_fisica": round(float(rec["P_FISICA"]), 3) if rec.get("P_FISICA") is not None else None,
                "speed": round(float(rec["SPEED"]), 1) if rec.get("SPEED") is not None else None,
                "decision": rec["DECISION_SISTEMA"],
                "gs_path": str(r["gs_path"]),
                "basename": r["basename"],
            })

        cases.append({
            "case_id": case_id,
            "trip_id": str(trip_id),
            "dataset": dataset,
            "alert_frame": alert_frame,
            "frames": frames_out,
        })

    print(f"Total de casos armados: {len(cases)}")
    assert len(cases) == EXPECTED_FP, f"Se esperaban {EXPECTED_FP} casos armados, hay {len(cases)}"

    # -----------------------------------------------------------------
    # Descarga de imagenes
    # -----------------------------------------------------------------
    print("Descargando imagenes necesarias desde GCS...")
    by_trip = {}
    for c in cases:
        trip_dir_name = c["trip_id"]
        for fr in c["frames"]:
            gs_path = fr["gs_path"]
            if not gs_path.startswith("gs://"):
                continue
            local_path = os.path.join(IMAGES_DIR, trip_dir_name, fr["basename"])
            fr["local_path"] = os.path.relpath(local_path, OUT_DIR)
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
        for fr in c["frames"]:
            lp = fr.get("local_path")
            if lp and not os.path.exists(os.path.join(OUT_DIR, lp)):
                missing += 1
    print(f"Verificacion de imagenes: {'todas existen' if missing == 0 else f'{missing} faltantes'}")

    # -----------------------------------------------------------------
    # Render HTML
    # -----------------------------------------------------------------
    for c in cases:
        for fr in c["frames"]:
            fr.pop("gs_path", None)

    cases_json = json.dumps(cases, ensure_ascii=False)
    causas_json = json.dumps(CAUSAS, ensure_ascii=False)
    html = (HTML_TEMPLATE
            .replace("__CASES_JSON__", cases_json)
            .replace("__CAUSAS_JSON__", causas_json)
            .replace("__N_CASES__", str(len(cases)))
            .replace("__CONFIG_NAME__", CONFIG_NAME))

    with open(HTML_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nReporte generado en: {HTML_FILE}")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Revision FP congelados 3-de-4 (__CONFIG_NAME__)</title>
<style>
    * { box-sizing: border-box; }
    body { font-family: sans-serif; background: #111; color: #eee; margin: 0; padding: 0; }
    #topbar { background: #1b1b1b; border-bottom: 1px solid #333; padding: 14px 20px; text-align: center; }
    #topbar .counter { font-size: 20px; font-weight: bold; color: #ffb454; }
    #topbar .meta { font-size: 14px; color: #ccc; margin-top: 6px; }
    #topbar .instruction { font-size: 14px; color: #fff; margin-top: 8px; }
    #topbar .progress { font-size: 13px; color: #8ecf8e; margin-top: 6px; }
    #export-btn { margin-top: 10px; background: #2c6b2f; color: #fff; border: none; border-radius: 6px; padding: 8px 18px; font-size: 14px; cursor: pointer; }
    #export-btn:hover { background: #3a8a3e; }
    #main { padding: 20px; max-width: 1600px; margin: 0 auto; }
    .frames-row { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }
    .frame-card { width: 190px; border: 3px solid #444; border-radius: 8px; padding: 6px; background: #000; text-align: center; }
    .frame-card.role-candidato { border-color: #d9a441; }
    .frame-card.role-confirmado { border: 5px solid #ff3b30; }
    .frame-card.coherent { box-shadow: 0 0 0 2px #ffdd55 inset; }
    .frame-card img { width: 100%; height: 190px; object-fit: cover; border-radius: 4px; cursor: zoom-in; display: block; }
    .role-tag { font-weight: bold; font-size: 12px; margin-top: 6px; text-transform: uppercase; }
    .role-tag.antes { color: #888; }
    .role-tag.candidato { color: #d9a441; }
    .role-tag.confirmado { color: #ff3b30; }
    .role-tag.despues { color: #888; }
    .frame-meta { font-size: 11.5px; margin-top: 6px; line-height: 1.5; text-align: left; padding: 0 4px; }
    .frame-meta div { display: flex; justify-content: space-between; }
    .frame-meta .lbl { color: #888; }
    .no-img { width: 100%; height: 190px; display: flex; align-items: center; justify-content: center; background: #222; border-radius: 4px; font-size: 12px; color: #888; }
    #review-panel { margin-top: 24px; background: #1b1b1b; border: 1px solid #333; border-radius: 10px; padding: 18px; max-width: 700px; margin-left: auto; margin-right: auto; }
    #review-panel h3 { margin-top: 0; font-size: 14px; color: #ccc; text-transform: uppercase; letter-spacing: 1px; }
    .radio-row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 10px; }
    .radio-row label { display: flex; align-items: center; gap: 6px; font-size: 15px; cursor: pointer; }
    #causa-row { display: none; margin-top: 10px; }
    #causa-row select { background: #222; color: #eee; border: 1px solid #444; border-radius: 6px; padding: 6px 10px; font-size: 14px; }
    #saved-tag { font-size: 13px; color: #8ecf8e; margin-top: 8px; height: 16px; }
    #nav { display: flex; justify-content: center; gap: 20px; margin-top: 24px; }
    #nav button { background: #333; color: #fff; border: none; border-radius: 8px; padding: 14px 28px; font-size: 17px; cursor: pointer; }
    #nav button:hover { background: #4a90d9; }
    #nav button:disabled { opacity: 0.3; cursor: default; }
    #lightbox { display: none; position: fixed; z-index: 1000; inset: 0; background: rgba(0,0,0,0.95); align-items: center; justify-content: center; cursor: zoom-out; }
    #lightbox.open { display: flex; }
    #lightbox img { max-width: 96vw; max-height: 96vh; border-radius: 6px; box-shadow: 0 0 40px rgba(0,0,0,0.8); }
</style>
</head>
<body>
    <div id="topbar">
        <div class="counter" id="counter">Caso 1 / __N_CASES__</div>
        <div class="meta" id="meta"></div>
        <div class="instruction">Config congelada __CONFIG_NAME__ (3-de-4). CAMBIO_CONFIRMADO marcado en rojo es un FP estricto (no matchea GT).</div>
        <div class="progress" id="progress"></div>
        <button id="export-btn" onclick="exportCsv()">Exportar CSV de respuestas</button>
    </div>
    <div id="main">
        <div id="case-content"></div>
        <div id="review-panel">
            <h3>Revision</h3>
            <div class="radio-row" id="respuesta-row"></div>
            <div id="causa-row">
                <label for="causa-select">Causa (misma persona):</label>
                <select id="causa-select" onchange="setCausa(this.value)"></select>
            </div>
            <div id="saved-tag"></div>
        </div>
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
const CAUSAS = __CAUSAS_JSON__;
const CONFIG_NAME = "__CONFIG_NAME__";
const STORAGE_KEY = "frozen_fp_review_" + CONFIG_NAME;
let current = 0;

function loadAnswers() {
    try {
        return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {};
    } catch (e) {
        return {};
    }
}
function saveAnswers(answers) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(answers));
}
let ANSWERS = loadAnswers();

function fmtTs(ts) {
    if (ts === null || ts === undefined) return "N/A";
    const d = new Date(ts * 1000);
    return d.toISOString().substr(11, 8);
}
function fmtNum(v) {
    return (v === null || v === undefined) ? "N/A" : v;
}

function frameCardHtml(fr) {
    const cls = "frame-card role-" + fr.role + (fr.is_coherent ? " coherent" : "");
    const imgTag = fr.local_path
        ? `<img src="${fr.local_path}" onclick="openLightbox('${fr.local_path}')">`
        : `<div class="no-img">sin imagen</div>`;
    return `
    <div class="${cls}">
        ${imgTag}
        <div class="role-tag ${fr.role}">${fr.role}${fr.is_coherent ? " (coherente)" : ""}</div>
        <div class="frame-meta">
            <div><span class="lbl">pos</span><span>${fr.pos > 0 ? "+" + fr.pos : fr.pos}</span></div>
            <div><span class="lbl">hora</span><span>${fmtTs(fr.ts_seconds)}</span></div>
            <div><span class="lbl">dist_mem</span><span>${fmtNum(fr.dist_mem)}</span></div>
            <div><span class="lbl">P_fisica</span><span>${fmtNum(fr.p_fisica)}</span></div>
            <div><span class="lbl">velocidad</span><span>${fmtNum(fr.speed)}</span></div>
            <div><span class="lbl">decision</span><span>${fr.decision}</span></div>
        </div>
    </div>`;
}

function renderRespuestaRow(caseId) {
    const ans = ANSWERS[caseId] || {};
    const opciones = [
        ["misma_persona", "Misma persona"],
        ["cambio_real", "Cambio real"],
        ["indeterminado", "Indeterminado"],
    ];
    document.getElementById("respuesta-row").innerHTML = opciones.map(([val, label]) => `
        <label>
            <input type="radio" name="respuesta" value="${val}" ${ans.respuesta === val ? "checked" : ""}
                onchange="setRespuesta('${val}')">
            ${label}
        </label>`).join("");

    const causaSelect = document.getElementById("causa-select");
    causaSelect.innerHTML = `<option value="">-- elegir causa --</option>` +
        CAUSAS.map(c => `<option value="${c}" ${ans.causa === c ? "selected" : ""}>${c}</option>`).join("");

    document.getElementById("causa-row").style.display = (ans.respuesta === "misma_persona") ? "block" : "none";
}

function setRespuesta(val) {
    const c = CASES[current];
    ANSWERS[c.case_id] = ANSWERS[c.case_id] || {};
    ANSWERS[c.case_id].respuesta = val;
    if (val !== "misma_persona") {
        delete ANSWERS[c.case_id].causa;
    }
    saveAnswers(ANSWERS);
    renderRespuestaRow(c.case_id);
    showSaved();
    updateProgress();
}

function setCausa(val) {
    const c = CASES[current];
    ANSWERS[c.case_id] = ANSWERS[c.case_id] || {};
    ANSWERS[c.case_id].causa = val;
    saveAnswers(ANSWERS);
    showSaved();
}

function showSaved() {
    const tag = document.getElementById("saved-tag");
    tag.textContent = "Guardado.";
    setTimeout(() => { tag.textContent = ""; }, 1200);
}

function updateProgress() {
    const total = CASES.length;
    const done = CASES.filter(c => ANSWERS[c.case_id] && ANSWERS[c.case_id].respuesta).length;
    document.getElementById("progress").textContent = `Revisados: ${done} / ${total}`;
}

function render() {
    const c = CASES[current];
    document.getElementById("counter").textContent = `Caso ${current + 1} / ${CASES.length}`;
    document.getElementById("meta").textContent = `trip_id: ${c.trip_id}  |  dataset: ${c.dataset}  |  alert_frame: ${c.alert_frame}`;

    document.getElementById("case-content").innerHTML = `
        <div class="frames-row">${c.frames.map(frameCardHtml).join("")}</div>`;

    renderRespuestaRow(c.case_id);
    updateProgress();

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

function exportCsv() {
    const rows = [["case_id", "trip_id", "dataset", "alert_frame", "respuesta", "causa"]];
    for (const c of CASES) {
        const a = ANSWERS[c.case_id] || {};
        rows.push([c.case_id, c.trip_id, c.dataset, c.alert_frame, a.respuesta || "", a.causa || ""]);
    }
    const csv = rows.map(r => r.map(v => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "frozen_fp_review_" + CONFIG_NAME + ".csv";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { closeLightbox(); return; }
    if (document.getElementById("lightbox").classList.contains("open")) return;
    if (e.target.tagName === "SELECT") return;
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
