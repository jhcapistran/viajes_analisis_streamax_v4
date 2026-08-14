"""
Genera el HTML standalone (sin servidor) del explicador visual/interactivo
de la config congelada "3-de-4" (E_w4_s3_c035_d085_p06). Lee
output/cases.json (generado por build_cases_data.py, que a su vez usa
algo_v1_explainer.py, copia de solo-lectura de la logica de decision real).
No cambia el algoritmo, no corre experimentos nuevos, no toca TEST.

Uso: uv run python generate_html.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")
CASES_JSON_PATH = os.path.join(OUT_DIR, "cases.json")
HTML_PATH = os.path.join(OUT_DIR, "index.html")

with open(CASES_JSON_PATH, "r", encoding="utf-8") as f:
    CASES = json.load(f)

CASES_JSON = json.dumps(CASES, ensure_ascii=False)

HTML = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Como funciona Memory v1 + 3-de-4</title>
<style>
:root {
  --bg: #0d1117; --panel: #161b22; --panel2: #1c2330; --border: #2a313d;
  --text: #e6edf3; --muted: #8b96a5; --accent: #4a90d9; --green: #3fb950;
  --yellow: #d9a441; --red: #f85149; --blue: #58a6ff; --purple: #bc8cff;
}
* { box-sizing: border-box; }
body { margin: 0; font-family: 'Segoe UI', Roboto, Arial, sans-serif; background: var(--bg); color: var(--text); }
h1,h2,h3,h4 { font-weight: 600; }
.wrap { max-width: 1500px; margin: 0 auto; padding: 24px; }

/* ---------- Hero ---------- */
#hero { background: linear-gradient(135deg,#0d1117 0%,#141c2b 60%,#0d1117 100%); border-bottom: 1px solid var(--border); padding: 40px 24px 30px; text-align: center; }
#hero h1 { font-size: 28px; margin: 0 0 8px; }
#hero .subtitle { color: var(--muted); font-size: 15px; max-width: 820px; margin: 0 auto; line-height: 1.5; }
.tagline { display:inline-block; margin-top:14px; padding:6px 16px; border-radius: 20px; background: rgba(74,144,217,0.15); border:1px solid var(--accent); color: var(--blue); font-size: 13px; }

/* ---------- Flow diagram ---------- */
#flow { display: flex; align-items: center; justify-content: center; gap: 6px; flex-wrap: wrap; margin: 28px auto 8px; max-width: 1300px; }
.flow-step { background: var(--panel2); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; font-size: 13px; text-align: center; min-width: 108px;
  opacity: 0; animation: fadeInUp .55s ease forwards; }
.flow-step b { display:block; font-size: 14px; margin-bottom: 3px; color: var(--blue); }
.flow-arrow { color: var(--muted); font-size: 20px; opacity: 0; animation: fadeInUp .55s ease forwards; }
@keyframes fadeInUp { from { opacity: 0; transform: translateY(10px);} to { opacity: 1; transform: translateY(0);} }

/* ---------- Rule box ---------- */
#rulebox { max-width: 980px; margin: 26px auto 10px; background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 18px 22px; }
#rulebox h3 { margin-top:0; color: var(--yellow); font-size: 15px; text-transform: uppercase; letter-spacing: .5px;}
#rulebox ul { margin: 0; padding-left: 20px; font-size: 13.5px; line-height: 1.85; color: #cfd8e3; }
#rulebox code { background: #0006; padding: 1px 6px; border-radius: 4px; color: var(--yellow); }

/* ---------- Case tabs ---------- */
#tabs-section { margin-top: 34px; }
.tab-group-label { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin: 14px 0 6px; }
#tabs { display: flex; flex-wrap: wrap; gap: 8px; }
.tab-btn { background: var(--panel2); border: 1px solid var(--border); color: var(--text); padding: 9px 14px; border-radius: 8px; cursor: pointer; font-size: 13px; transition: all .15s; }
.tab-btn:hover { border-color: var(--accent); }
.tab-btn.active { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; }
.tab-btn.kind-fp.active { background: var(--red); border-color: var(--red); }
.tab-btn.kind-vuelve.active, .tab-btn.kind-fp_dificil.active { background: var(--purple); border-color: var(--purple); }

/* ---------- Case view ---------- */
#case-view { margin-top: 22px; }
.case-header { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 18px 22px; margin-bottom: 16px; }
.case-header h2 { margin: 0 0 6px; font-size: 19px; }
.case-meta { color: var(--muted); font-size: 13px; }
.gt-box { margin-top: 12px; border: 1px dashed #665; border-radius: 8px; padding: 10px 14px; background: rgba(217,164,65,0.08); font-size: 13px; }
.gt-box .gt-tag { display:inline-block; font-weight:700; font-size: 11px; letter-spacing: .5px; color: var(--yellow); border: 1px solid var(--yellow); border-radius: 5px; padding: 2px 8px; margin-right: 8px; }
.gt-box .gt-note { display:block; margin-top: 6px; color: var(--muted); font-size: 11.5px; font-style: italic; }

/* ---------- Timeline ---------- */
#timeline-wrap { overflow-x: auto; padding: 6px 2px 14px; }
#timeline { display: flex; gap: 8px; align-items: flex-end; position: relative; }
.tl-frame { flex: 0 0 auto; width: 78px; text-align: center; cursor: pointer; }
.tl-frame .thumb { width: 78px; height: 78px; object-fit: cover; border-radius: 6px; border: 3px solid #333; display:block; }
.tl-frame .thumb.dec-INICIO_VIAJE { border-color: var(--blue); }
.tl-frame .thumb.dec-MISMO_CONDUCTOR { border-color: var(--green); }
.tl-frame .thumb.dec-POSIBLE_CAMBIO { border-color: var(--yellow); }
.tl-frame .thumb.dec-CAMBIO_CONFIRMADO { border-color: var(--red); box-shadow: 0 0 10px rgba(248,81,73,.6); }
.tl-frame .thumb.dec-INDETERMINADO { border-color: #555; }
.tl-frame.selected .thumb { outline: 3px solid #fff; }
.tl-frame .idx { font-size: 10.5px; color: var(--muted); margin-top: 3px; }
.tl-connector { position:absolute; height: 2px; background: var(--border); z-index: -1; }

/* ---------- Detail grid ---------- */
#detail { display: grid; grid-template-columns: 380px 1fr; gap: 18px; }
@media (max-width: 980px) { #detail { grid-template-columns: 1fr; } }
.panel { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px; }
.panel h4 { margin: 0 0 10px; font-size: 12.5px; text-transform: uppercase; letter-spacing: .6px; color: var(--muted); }
#big-img-wrap { text-align: center; }
#big-img-wrap img { max-width: 100%; max-height: 340px; border-radius: 8px; border: 4px solid #333; }
#decision-badge { display:inline-block; margin-top: 12px; padding: 8px 22px; border-radius: 30px; font-weight: 700; font-size: 15px; letter-spacing: .5px; }
.badge-INICIO_VIAJE { background: rgba(88,166,255,0.18); color: var(--blue); border:1px solid var(--blue); }
.badge-MISMO_CONDUCTOR { background: rgba(63,185,80,0.18); color: var(--green); border:1px solid var(--green); }
.badge-POSIBLE_CAMBIO { background: rgba(217,164,65,0.18); color: var(--yellow); border:1px solid var(--yellow); }
.badge-CAMBIO_CONFIRMADO { background: rgba(248,81,73,0.2); color: var(--red); border:1px solid var(--red); }
.badge-INDETERMINADO { background: rgba(139,148,165,0.18); color: var(--muted); border:1px solid var(--muted); }

.stat-row { display:flex; justify-content: space-between; font-size: 13px; padding: 3px 0; border-bottom: 1px dotted #2a313d; }
.stat-row .k { color: var(--muted); }
.stat-row .v { font-weight: 600; }

.right-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
@media (max-width: 720px) { .right-grid { grid-template-columns: 1fr; } }

.formula { font-family: 'Cambria Math', 'Latin Modern Math', Georgia, serif; background: #0006; border-radius: 8px; padding: 10px 12px; margin: 8px 0; font-size: 14.5px; line-height: 1.6; }
.formula .num { color: var(--blue); font-weight: 700; font-family: 'Segoe UI', sans-serif; }
.formula .res { color: var(--yellow); font-weight: 700; font-family: 'Segoe UI', sans-serif; }

.mem-row { display: flex; gap: 6px; flex-wrap: wrap; margin: 8px 0; }
.mem-chip { width: 52px; text-align: center; }
.mem-chip img { width: 52px; height: 52px; object-fit: cover; border-radius: 6px; border: 2px solid var(--accent); }
.mem-chip .noimg { width: 52px; height: 52px; border-radius: 6px; border: 2px dashed #555; display:flex; align-items:center; justify-content:center; font-size: 10px; color: var(--muted); }
.mem-chip .lbl { font-size: 9.5px; color: var(--muted); margin-top: 2px; }

.window-row { display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0; }
.win-slot { width: 66px; text-align: center; border-radius: 8px; padding: 4px; border: 2px solid #444; }
.win-slot.coherent { border-color: var(--yellow); box-shadow: 0 0 8px rgba(217,164,65,.5); }
.win-slot img { width: 58px; height: 58px; object-fit: cover; border-radius: 5px; }
.win-slot .noimg { width: 58px; height: 58px; border-radius: 5px; border: 1px dashed #555; display:flex; align-items:center; justify-content:center; font-size: 9px; color: var(--muted); }
.win-slot .d { font-size: 10.5px; margin-top: 3px; color: var(--muted); }

.check-row { display:flex; justify-content: space-between; align-items:center; font-size: 13.5px; padding: 6px 8px; margin: 4px 0; border-radius: 6px; background: #0004; }
.check-row .ok { color: var(--green); font-weight: 700; }
.check-row .bad { color: var(--red); font-weight: 700; }
.final-verdict { margin-top: 10px; text-align:center; padding: 10px; border-radius: 8px; font-weight: 700; font-size: 14px; }
.final-verdict.yes { background: rgba(248,81,73,.18); color: var(--red); border: 1px solid var(--red); }
.final-verdict.no { background: rgba(63,185,80,.12); color: var(--green); border: 1px solid var(--green); }

.stage-flow { display:flex; align-items:center; justify-content:center; gap: 10px; margin: 10px 0 4px; flex-wrap: wrap;}
.stage { padding: 8px 14px; border-radius: 8px; border: 1px solid #444; font-size: 12.5px; color: var(--muted); }
.stage.active { border-color: var(--accent); color: var(--text); background: rgba(74,144,217,.15); font-weight:700; }
.stage-arrow { color: #555; }

/* ---------- Metrics chart (evolucion en el tiempo) ---------- */
.chart-wrap { overflow-x: auto; }
.chart-legend { display:flex; gap: 16px; flex-wrap: wrap; font-size: 12px; color: var(--muted); margin-top: 6px; }
.chart-legend span { display:inline-flex; align-items:center; gap:5px; }
.chart-legend i { width: 14px; height: 3px; display:inline-block; border-radius: 2px; }
.chart-col-hit { cursor: pointer; }
.chart-col-hit:hover { fill: rgba(255,255,255,0.06); }

/* ---------- Ventana: evolucion / solapamiento ---------- */
.panel.wide { grid-column: 1 / -1; }
.wgrid { display:flex; flex-direction:column; gap:3px; overflow-x:auto; padding-bottom: 4px; }
.wgrid-row { display:flex; align-items:center; gap:4px; }
.wgrid-label { width:112px; flex:0 0 auto; font-size:11px; color:var(--muted); white-space:nowrap; }
.wgrid-cells { display:flex; gap:3px; }
.wgrid-collabel { width:24px; text-align:center; font-size:10px; color:var(--muted); }
.wcell { width:24px; height:20px; border-radius:4px; background:#20242c; border:1px solid #2a313d; cursor:pointer; }
.wc-anom { background: rgba(88,166,255,.35); border-color: var(--blue); }
.wc-coherent { background: rgba(217,164,65,.6); border-color: var(--yellow); }
.wc-current { outline: 2px solid #fff; outline-offset: -1px; }

/* ---------- Comparacion (ahora es un caso interactivo mas) ---------- */
.cmp-intro { font-size: 13.5px; color: var(--muted); line-height: 1.6; margin-bottom: 14px; }
.cmp-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }
@media (max-width: 1100px) { .cmp-grid { grid-template-columns: 1fr; } }
.cmp-col { border-radius: 12px; padding: 4px; border: 2px solid; }
.cmp-col.ok { border-color: var(--green); }
.cmp-col.fp { border-color: var(--red); }
.cmp-col-title { text-align:center; font-weight: 700; padding: 8px; font-size: 13.5px; letter-spacing:.4px; }
.cmp-col.ok .cmp-col-title { color: var(--green); }
.cmp-col.fp .cmp-col-title { color: var(--red); }
.cmp-col .case-header, .cmp-col .panel { margin: 8px; }
.compare-note { text-align:center; max-width: 900px; margin: 22px auto 0; color: var(--muted); font-size: 13.5px; line-height:1.6; }

/* ---------- Lightbox ---------- */
#lightbox { display:none; position: fixed; inset:0; background: rgba(0,0,0,.92); z-index: 999; align-items:center; justify-content:center; cursor: zoom-out; }
#lightbox.open { display:flex; }
#lightbox img { max-width: 94vw; max-height: 94vh; border-radius: 8px; }

footer { text-align:center; color: var(--muted); font-size: 12px; padding: 30px 10px 50px; }
</style>
</head>
<body>

<div id="hero">
  <h1>Memory v1 + 3-de-4: como decide el algoritmo</h1>
  <div class="subtitle">
    Explicador visual e interactivo del sistema de deteccion de cambio de conductor por memoria de embeddings faciales,
    coherencia temporal y fisica del vehiculo. Todos los numeros que se muestran a continuacion son calculos reales
    sobre viajes reales del repositorio (no son ejemplos inventados).
  </div>
  <div class="tagline">E_w4_s3_c035_d085_p06 &nbsp;|&nbsp; ventana=4 &nbsp;|&nbsp; soporte=3 &nbsp;|&nbsp; coherencia&lt;0.35 &nbsp;|&nbsp; dist&ge;0.85 &nbsp;|&nbsp; fisica&ge;0.60</div>
</div>

<div class="wrap">

  <div id="flow"></div>

  <div id="rulebox">
    <h3>La regla de confirmacion (no es una simple suma de umbrales)</h3>
    <ul>
      <li><code>dist_mem &ge; 0.50</code> &rarr; el frame es candidato a anomalia (no hace match directo con la memoria del conductor vigente).</li>
      <li>Se abre/actualiza una <b>ventana candidata</b> de hasta <b>4</b> observaciones anomalas recientes.</li>
      <li>Dentro de esa ventana se busca el <b>grupo mas coherente entre si</b> (distancia par-a-par <code>&lt; 0.35</code>).</li>
      <li>Ese grupo debe tener <b>soporte &ge; 3</b> (al menos 3 de las 4 observaciones deben ser coherentes entre si).</li>
      <li>La <b>distancia promedio del grupo coherente contra la memoria vieja</b> debe ser <code>&ge; 0.85</code>.</li>
      <li>La <b>fisica del ultimo salto</b> (tiempo disponible vs. tiempo minimo de maniobra) debe dar <code>P_fisica &ge; 0.60</code>.</li>
      <li>Si <b>las 3 condiciones se cumplen a la vez</b> &rarr; <code>CAMBIO_CONFIRMADO</code>. Si el conductor anterior reaparece antes (dist_mem &lt; 0.5), el candidato se descarta por completo.</li>
    </ul>
  </div>

  <div id="tabs-section">
    <div class="tab-group-label">Cambios reales detectados correctamente</div>
    <div id="tabs-ok" class="tab-list"></div>
    <div class="tab-group-label">Falsos positivos confirmados (revisados manualmente como misma persona)</div>
    <div id="tabs-fp" class="tab-list"></div>
    <div class="tab-group-label">Caso especial: anomalia que no confirma</div>
    <div id="tabs-special" class="tab-list"></div>
    <div class="tab-group-label">Comparacion directa</div>
    <div id="tabs-cmp" class="tab-list"></div>
  </div>

  <div id="case-view"></div>

</div>

<div id="lightbox" onclick="closeLightbox()"><img id="lightbox-img" src=""></div>

<footer>
  Generado a partir de datos reales de random_trips_data_2026_04.csv / all_reviewed_trips_data_2026_07.csv &middot;
  E_w4_s3_c035_d085_p06 (algo_v1_param.py) &middot; Solo visualizacion, ningun umbral fue modificado para este reporte.
</footer>

<script>
const CASES = __CASES_JSON__;
// Caso sintetico de comparacion directa: no trae frames propios, apunta a
// dos casos ya existentes (mismos datos reales, sin duplicar). Se agrega
// como un caso mas seleccionable en las pestañas, no como seccion fija.
CASES.push({ case_id: "cmp", kind: "comparacion", titulo: "Comparacion directa: caso correcto vs falso positivo",
  leftId: "tp2", rightId: "fp1" });

const TH = { dist_mem: 0.50, window: 4, support: 3, coherence: 0.35, avg_dist: 0.85, p_fisica: 0.60 };
let currentCaseIdx = 0;
let currentFrameLocalIdx = null;
let cmpLeftIdx = null;
let cmpRightIdx = null;

// ---------------------------------------------------------------
// Flow diagram
// ---------------------------------------------------------------
function renderFlow() {
  const steps = ["Frame", "Embedding", "Memoria (centroide)", "Distancia coseno", "Candidato (ventana)", "Coherencia + Persistencia + Fisica", "Decision"];
  const el = document.getElementById("flow");
  let html = "";
  let delay = 0;
  steps.forEach((s, i) => {
    html += `<div class="flow-step" style="animation-delay:${delay}s"><b>${i+1}</b>${s}</div>`;
    delay += 0.12;
    if (i < steps.length - 1) {
      html += `<div class="flow-arrow" style="animation-delay:${delay}s">&rarr;</div>`;
      delay += 0.05;
    }
  });
  el.innerHTML = html;
}

// ---------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------
function renderTabs() {
  const groups = { ok: [], fp: [], special: [], cmp: [] };
  CASES.forEach((c, i) => {
    if (c.kind === "correcto") groups.ok.push(i);
    else if (c.kind === "fp" || c.kind === "fp_dificil") groups.fp.push(i);
    else if (c.kind === "comparacion") groups.cmp.push(i);
    else groups.special.push(i);
  });
  const mk = (idxs) => idxs.map(i => {
    const c = CASES[i];
    return `<button class="tab-btn kind-${c.kind}" data-idx="${i}" onclick="selectCase(${i})">${c.titulo}</button>`;
  }).join("");
  document.getElementById("tabs-ok").innerHTML = mk(groups.ok);
  document.getElementById("tabs-fp").innerHTML = mk(groups.fp);
  document.getElementById("tabs-special").innerHTML = mk(groups.special);
  document.getElementById("tabs-cmp").innerHTML = mk(groups.cmp);
}

function markActiveTab() {
  document.querySelectorAll(".tab-btn").forEach(b => {
    b.classList.toggle("active", parseInt(b.dataset.idx) === currentCaseIdx);
  });
}

// ---------------------------------------------------------------
// Case view
// ---------------------------------------------------------------
function selectCase(i) {
  currentCaseIdx = i;
  currentFrameLocalIdx = null;
  cmpLeftIdx = null;
  cmpRightIdx = null;
  markActiveTab();
  renderCase();
}

function fmt(v, d) {
  if (v === null || v === undefined || Number.isNaN(v)) return "N/A";
  return Number(v).toFixed(d === undefined ? 3 : d);
}

function findCase(id) { return CASES.find(c => c.case_id === id); }

function frameByIdx(caseObj, idx) {
  return caseObj.frames.find(f => f.frame_idx === idx);
}

function defaultFrameIdx(c) {
  let target = c.alert_frame;
  let found = target !== null && target !== undefined ? c.frames.findIndex(f => f.frame_idx === target) : -1;
  if (found === -1) found = Math.floor(c.frames.length / 2);
  return found;
}

function renderCase() {
  const c = CASES[currentCaseIdx];
  if (c.kind === "comparacion") {
    renderComparisonCase(c);
    return;
  }
  if (currentFrameLocalIdx === null) currentFrameLocalIdx = defaultFrameIdx(c);

  let gtTagLabel = c.kind === "correcto" ? "RESULTADO HUMANO / GT" : (c.kind.startsWith("fp") ? "RESULTADO HUMANO / GT" : "GT (dato crudo, sin revision manual dedicada)");

  const html = `
  ${buildCaseHeaderHtml(c, gtTagLabel)}
  <div class="panel chart-wrap">
    <h4>Evolucion de las metricas clave a lo largo del tramo (clickeable)</h4>
    <div id="chart-${c.case_id}"></div>
  </div>
  <div id="timeline-wrap"><div id="timeline"></div></div>
  <div id="detail"></div>`;

  document.getElementById("case-view").innerHTML = html;
  document.getElementById(`chart-${c.case_id}`).innerHTML = buildMetricsChartHtml(c, currentFrameLocalIdx, "selectFrame");
  renderTimeline(c);
  renderDetail(c);
}

function buildCaseHeaderHtml(c, gtTagLabel) {
  return `
  <div class="case-header">
    <h2>${c.titulo}</h2>
    <div class="case-meta">Dataset: <b>${c.dataset}</b> &middot; trip_id: <b>${c.trip_id}</b> &middot; ${c.n_total_frames} frames validos en el viaje completo</div>
    <div class="gt-box">
      <span class="gt-tag">${gtTagLabel}</span>${c.gt_label}
      <span class="gt-note">Esta etiqueta es SOLO para contexto humano. El algoritmo nunca ve identity_id ni revision manual: decide unicamente con embeddings, velocidad y tiempo.</span>
    </div>
  </div>`;
}

// ---------------------------------------------------------------
// Metrics chart: como evolucionan dist_mem / distancia promedio / P_fisica
// frame a frame, para poder ver de un vistazo por que cambia la decision.
// ---------------------------------------------------------------
function buildMetricsChartHtml(c, selectedLocalIdx, clickFn) {
  const frames = c.frames;
  const n = frames.length;
  const colW = 46;
  const width = Math.max(360, n * colW + 40);
  const height = 170;
  const top = 12, bottom = 26, left = 6, right = 6;
  const plotH = height - top - bottom;
  const plotW = width - left - right;
  const stepX = plotW / Math.max(1, n - 1);
  const yMax = 1.15;
  const yOf = (v) => top + (1 - Math.min(v, yMax) / yMax) * plotH;
  const xOf = (i) => left + i * stepX;

  const decColors = {
    INICIO_VIAJE: "rgba(88,166,255,0.16)", MISMO_CONDUCTOR: "rgba(63,185,80,0.14)",
    POSIBLE_CAMBIO: "rgba(217,164,65,0.20)", CAMBIO_CONFIRMADO: "rgba(248,81,73,0.28)",
    INDETERMINADO: "rgba(139,148,165,0.14)",
  };

  const colW2 = stepX;
  let bg = frames.map((f, i) => {
    const x0 = xOf(i) - colW2 / 2;
    return `<rect class="chart-col-hit" x="${x0}" y="${top}" width="${colW2}" height="${plotH}" fill="${decColors[f.decision] || 'transparent'}" onclick="${clickFn}(${i})"></rect>`;
  }).join("");

  function lineFor(getter) {
    const pts = [];
    frames.forEach((f, i) => {
      const v = getter(f);
      if (v === null || v === undefined) return;
      pts.push(`${xOf(i)},${yOf(v)}`);
    });
    return pts;
  }
  function pathFromPts(pts, color) {
    if (pts.length < 2) return "";
    return `<polyline points="${pts.join(' ')}" fill="none" stroke="${color}" stroke-width="2"></polyline>`;
  }
  function dotsFromPts(pts, color, getter) {
    return frames.map((f, i) => {
      const v = getter(f);
      if (v === null || v === undefined) return "";
      return `<circle cx="${xOf(i)}" cy="${yOf(v)}" r="3" fill="${color}"></circle>`;
    }).join("");
  }

  const distMemPts = lineFor(f => f.dist_mem);
  const avgDistPts = lineFor(f => f.avg_dist_vs_old);
  const pFisicaPts = lineFor(f => f.p_fisica);

  const thLine = (v, color) => `<line x1="${left}" y1="${yOf(v)}" x2="${width-right}" y2="${yOf(v)}" stroke="${color}" stroke-width="1" stroke-dasharray="4,3" opacity="0.55"></line>`;

  const cursorX = xOf(selectedLocalIdx);
  const cursor = `<line x1="${cursorX}" y1="${top}" x2="${cursorX}" y2="${top+plotH}" stroke="#fff" stroke-width="2" opacity="0.85"></line>`;

  const labels = frames.map((f, i) => `<text x="${xOf(i)}" y="${height-8}" font-size="9.5" fill="#8b96a5" text-anchor="middle">${f.frame_idx}</text>`).join("");

  const svg = `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
    ${bg}
    ${thLine(TH.dist_mem, '#58a6ff')}
    ${thLine(TH.avg_dist, '#d9a441')}
    ${thLine(TH.p_fisica, '#3fb950')}
    ${pathFromPts(distMemPts, '#58a6ff')}
    ${pathFromPts(avgDistPts, '#d9a441')}
    ${pathFromPts(pFisicaPts, '#3fb950')}
    ${dotsFromPts(distMemPts, '#58a6ff', f => f.dist_mem)}
    ${dotsFromPts(avgDistPts, '#d9a441', f => f.avg_dist_vs_old)}
    ${dotsFromPts(pFisicaPts, '#3fb950', f => f.p_fisica)}
    ${cursor}
    ${labels}
  </svg>`;

  const legend = `<div class="chart-legend">
    <span><i style="background:#58a6ff"></i> dist_mem (umbral 0.50)</span>
    <span><i style="background:#d9a441"></i> distancia promedio vs memoria vieja (umbral 0.85)</span>
    <span><i style="background:#3fb950"></i> P_fisica (umbral 0.60)</span>
    <span style="color:#fff;">| linea blanca = frame seleccionado. El fondo de cada columna refleja la decision del algoritmo en ese frame.</span>
  </div>`;

  return svg + legend;
}

function renderTimeline(c) {
  const el = document.getElementById("timeline");
  el.innerHTML = buildTimelineHtml(c, currentFrameLocalIdx, "selectFrame");
}

function buildTimelineHtml(c, selectedLocalIdx, clickFn) {
  return c.frames.map((f, li) => {
    const img = f.local_path ? `<img class="thumb dec-${f.decision}" src="${f.local_path}">` : `<div class="thumb dec-${f.decision}" style="display:flex;align-items:center;justify-content:center;font-size:9px;color:#888;">sin img</div>`;
    return `<div class="tl-frame ${li === selectedLocalIdx ? 'selected' : ''}" onclick="${clickFn}(${li})">
      ${img}<div class="idx">#${f.frame_idx}</div></div>`;
  }).join("");
}

function selectFrame(li) {
  currentFrameLocalIdx = li;
  const c = CASES[currentCaseIdx];
  document.getElementById(`chart-${c.case_id}`).innerHTML = buildMetricsChartHtml(c, currentFrameLocalIdx, "selectFrame");
  renderTimeline(c);
  renderDetail(c);
}

function memChip(c, idx) {
  const f = frameByIdx(c, idx);
  if (f && f.local_path) {
    return `<div class="mem-chip"><img src="${f.local_path}"><div class="lbl">#${idx}</div></div>`;
  }
  return `<div class="mem-chip"><div class="noimg">#${idx}</div><div class="lbl">fuera de rango</div></div>`;
}

function winSlot(c, idx, dist, isCoherent) {
  const f = frameByIdx(c, idx);
  const img = (f && f.local_path) ? `<img src="${f.local_path}">` : `<div class="noimg">#${idx}</div>`;
  return `<div class="win-slot ${isCoherent ? 'coherent' : ''}">${img}<div class="d">#${idx}<br>d=${fmt(dist,3)}</div></div>`;
}

function checkRow(label, value, threshold, op) {
  let ok;
  if (op === ">=") ok = value >= threshold;
  else ok = value < threshold;
  const cls = ok ? "ok" : "bad";
  const sym = ok ? "&#10003;" : "&#10007;";
  return `<div class="check-row"><span>${label}</span><span class="${cls}">${sym} ${value === null || value === undefined ? "N/A" : fmt(value,3)} ${op} ${threshold}</span></div>`;
}

// ---------------------------------------------------------------
// Panel: como se desliza la ventana candidata (solapamiento entre pasos)
// ---------------------------------------------------------------
function buildWindowEvolutionHtml(c, li) {
  const f = c.frames[li];
  if (!(f.decision === "POSIBLE_CAMBIO" || f.decision === "CAMBIO_CONFIRMADO")) {
    return `<div class="panel wide">
      <h4>Como se desliza la ventana (solapamiento entre pasos)</h4>
      <div style="color:#888;font-size:12.5px;">No hay racha de candidato activa en este frame (dist_mem &lt; 0.50 o inicio/indeterminado): la ventana esta vacia.</div>
    </div>`;
  }
  // Encontrar el inicio de la racha continua de POSIBLE_CAMBIO/CAMBIO_CONFIRMADO
  let start = li;
  while (start > 0 && c.frames[start - 1].decision === "POSIBLE_CAMBIO") start--;
  const steps = c.frames.slice(start, li + 1).filter(s => s.decision === "POSIBLE_CAMBIO" || s.decision === "CAMBIO_CONFIRMADO");

  const allIdx = new Set();
  steps.forEach(s => {
    const arr = s.pending_frames || s.candidate_frames || [];
    arr.forEach(x => allIdx.add(x));
  });
  if (allIdx.size === 0) {
    return `<div class="panel wide"><h4>Como se desliza la ventana (solapamiento entre pasos)</h4><div style="color:#888;font-size:12.5px;">Sin datos de ventana.</div></div>`;
  }
  const minCol = Math.min(...allIdx), maxCol = Math.max(...allIdx);
  const cols = [];
  for (let i = minCol; i <= maxCol; i++) cols.push(i);

  const colLabels = cols.map(i => `<div class="wgrid-collabel">${i}</div>`).join("");

  const rows = steps.map(s => {
    const winArr = s.pending_frames || s.candidate_frames || [];
    const coherentSet = new Set();
    if (s.pending_frames) {
      (s.coherent_local || []).forEach(localI => coherentSet.add(s.pending_frames[localI]));
    } else if (s.candidate_frames) {
      (s.coherent_frames || []).forEach(x => coherentSet.add(x));
    }
    const cells = cols.map(colIdx => {
      const inWindow = winArr.includes(colIdx);
      let cls = "wcell";
      if (inWindow) cls += coherentSet.has(colIdx) ? " wc-coherent" : " wc-anom";
      if (colIdx === s.frame_idx) cls += " wc-current";
      const targetLi = c.frames.findIndex(x => x.frame_idx === s.frame_idx);
      return `<div class="${cls}" title="frame #${colIdx}" onclick="selectFrame(${targetLi})"></div>`;
    }).join("");
    const tag = s.decision === "CAMBIO_CONFIRMADO" ? " &rarr; CONFIRMA" : "";
    return `<div class="wgrid-row"><div class="wgrid-label">frame #${s.frame_idx}${tag}</div><div class="wgrid-cells">${cells}</div></div>`;
  }).join("");

  return `<div class="panel wide">
    <h4>Como se desliza la ventana (solapamiento entre pasos)</h4>
    <div class="wgrid">
      <div class="wgrid-row"><div class="wgrid-label">frame # &rarr; columna = idx</div><div class="wgrid-cells">${colLabels}</div></div>
      ${rows}
    </div>
    <div style="font-size:11.5px;color:var(--muted);margin-top:6px;">
      Cada fila es un paso de la racha (ventana FIFO de hasta ${TH.window} observaciones anomalas). Celda amarilla = parte del
      grupo coherente en ese paso; celda azul = estaba en la ventana pero no coherente. Se ve como la ventana se desplaza
      1 frame por paso y se solapa 3-de-4 con el paso anterior (no es un promedio acumulado sin memoria del orden).
    </div>
  </div>`;
}

function renderDetail(c) {
  document.getElementById("detail").innerHTML = buildDetailHtml(c, currentFrameLocalIdx);
}

function buildDetailHtml(c, li) {
  const f = c.frames[li];

  // --- left panel: image + decision ---
  const bigImg = f.local_path ? `<img src="${f.local_path}" onclick="openLightbox('${f.local_path}')">` : `<div style="padding:60px 0;color:#888;">sin imagen</div>`;
  const left = `
  <div class="panel">
    <h4>Frame seleccionado #${f.frame_idx}</h4>
    <div id="big-img-wrap">${bigImg}</div>
    <div style="text-align:center;"><div id="decision-badge" class="badge-${f.decision}">${f.decision.replace(/_/g,' ')}</div></div>
    <div style="margin-top:14px;">
      <div class="stat-row"><span class="k">Hora (ts)</span><span class="v">${f.ts_seconds ?? 'N/A'}</span></div>
      <div class="stat-row"><span class="k">Velocidad</span><span class="v">${fmt(f.speed,1)} km/h</span></div>
      <div class="stat-row"><span class="k">&Delta;t vs frame anterior</span><span class="v">${f.delta_t ?? 'N/A'} s</span></div>
      <div class="stat-row"><span class="k">Distancia vs frame anterior</span><span class="v">${fmt(f.prev_frame_dist,3)}</span></div>
      <div class="stat-row"><span class="k">dist_mem (vs memoria)</span><span class="v">${fmt(f.dist_mem,3)}</span></div>
      <div class="stat-row"><span class="k">P_fisica</span><span class="v">${fmt(f.p_fisica,3)}</span></div>
      <div class="stat-row"><span class="k">Tamano memoria actual</span><span class="v">${f.memory_size ?? 'N/A'}</span></div>
    </div>
    <div class="stage-flow">
      <div class="stage ${f.decision==='MISMO_CONDUCTOR'||f.decision==='INICIO_VIAJE' ? 'active':''}">Memoria A</div>
      <div class="stage-arrow">&rarr;</div>
      <div class="stage ${f.decision==='POSIBLE_CAMBIO' ? 'active':''}">Candidato B</div>
      <div class="stage-arrow">&rarr;</div>
      <div class="stage ${f.decision==='CAMBIO_CONFIRMADO' ? 'active':''}">Confirmacion</div>
    </div>
  </div>`;

  // --- memory panel ---
  const memIdxList = f.memory_frame_idx || [];
  const memChips = memIdxList.map(idx => memChip(c, idx)).join("");
  const mb = f.dist_mem_breakdown;
  const K = f.memory_size ?? memIdxList.length;
  const sumTerms = memIdxList.map(idx => `e<sub>${idx}</sub>`).join(" + ");
  const allKs = c.frames.map(x => x.memory_size).filter(v => v !== null && v !== undefined);
  const minK = allKs.length ? Math.min(...allKs) : null;
  const maxK = allKs.length ? Math.max(...allKs) : null;
  const kNote = (minK !== null && minK !== maxK)
    ? `K no es fijo: en este tramo vario entre <b style="color:#fff;">${minK}</b> y <b style="color:#fff;">${maxK}</b>. Crece de a 1 al inicio del viaje hasta el maximo (6); si hay un CAMBIO_CONFIRMADO, la memoria se reinicia solo con las observaciones coherentes que confirmaron el cambio (puede ser menos de 6), y vuelve a crecer desde ahi.`
    : `En este tramo K se mantuvo constante en ${maxK ?? '?'} (memoria ya llena, ventana FIFO de tamano fijo).`;
  const muFormula = memIdxList.length ? `
    <div class="formula">
      &mu;<sub>A</sub> = (1/K) &middot; (${sumTerms}) &nbsp;&nbsp;con&nbsp;&nbsp; K = <span class="res">${K}</span>
    </div>` : `
    <div style="color:#888;font-size:12.5px;">Memoria vacia: no hay embeddings previos para promediar.</div>`;
  const memFormula = mb ? `
    <div class="formula">
      &mu;<sub>A</sub>&middot;e<sub>t</sub> (producto punto) = <span class="res">${fmt(mb.dot,3)}</span>
    </div>
    <div class="formula">
      ||&mu;<sub>A</sub>|| = <span class="num">${fmt(mb.norm_a,3)}</span> &nbsp;&nbsp; ||e<sub>t</sub>|| = <span class="num">${fmt(mb.norm_b,3)}</span>
    </div>
    <div class="formula">
      cos_sim = (&mu;<sub>A</sub>&middot;e<sub>t</sub>) / (||&mu;<sub>A</sub>|| &middot; ||e<sub>t</sub>||) = ${fmt(mb.dot,3)} / (${fmt(mb.norm_a,3)} &middot; ${fmt(mb.norm_b,3)}) = <span class="res">${fmt(mb.cos_sim,3)}</span>
    </div>
    <div class="formula">
      dist_mem = 1 &minus; cos_sim = 1 &minus; ${fmt(mb.cos_sim,3)} = <span class="res">${fmt(f.dist_mem,3)}</span>
    </div>` : `
    <div style="color:#888;font-size:12.5px;">Sin memoria previa: no hay centroide contra el cual comparar.</div>`;
  const memPanel = `
  <div class="panel">
    <h4>Memoria del conductor vigente (en este instante: K=${K ?? '?'} embeddings)</h4>
    <div class="mem-row">${memChips || '<span style="color:#888;font-size:12px;">sin memoria (inicio de viaje)</span>'}</div>
    ${muFormula}
    <div style="font-size:11.5px;color:var(--muted);margin:6px 0 2px;">${kNote}</div>
    ${memFormula}
  </div>`;

  // --- candidate window panel (fotos del instante actual) ---
  let winPanel;
  if (f.pending_frames && f.pending_frames.length) {
    const coherentSet = new Set(f.coherent_local || []);
    const slots = f.pending_frames.map((idx, pli) => winSlot(c, idx, f.pending_dists ? f.pending_dists[pli] : null, coherentSet.has(pli))).join("");
    const support = f.support ?? 0;
    winPanel = `
    <div class="panel">
      <h4>Ventana candidata en este instante (max ${TH.window} observaciones anomalas)</h4>
      <div class="window-row">${slots}</div>
      <div style="font-size:12.5px;color:var(--muted);margin-top:4px;">Recuadro amarillo = grupo coherente entre si (distancia par-a-par &lt; ${TH.coherence}). Soporte actual: <b style="color:#fff;">${support}/${TH.support}</b></div>
    </div>`;
  } else if (f.decision === 'CAMBIO_CONFIRMADO' && f.candidate_frames) {
    const coherentFrameSet = new Set(f.coherent_frames || []);
    const slots = f.candidate_frames.map(idx => winSlot(c, idx, null, coherentFrameSet.has(idx))).join("");
    winPanel = `
    <div class="panel">
      <h4>Ventana candidata que confirmo el cambio</h4>
      <div class="window-row">${slots}</div>
      <div style="font-size:12.5px;color:var(--muted);margin-top:4px;">Recuadro amarillo = subgrupo coherente usado para confirmar (${(f.coherent_frames||[]).length}/${TH.support} minimo).</div>
    </div>`;
  } else {
    winPanel = `
    <div class="panel">
      <h4>Ventana candidata en este instante</h4>
      <div style="color:#888;font-size:12.5px;">No hay ventana candidata abierta en este frame (dist_mem &lt; 0.50, sigue matcheando la memoria vigente).</div>
    </div>`;
  }

  const winEvolutionPanel = buildWindowEvolutionHtml(c, li);

  // --- physics panel ---
  const pb = f.physics_breakdown;
  let physPanel;
  if (pb) {
    physPanel = `
    <div class="panel">
      <h4>Fisica del vehiculo</h4>
      <div class="formula">v<sub>prev</sub> = ${fmt(f.speed,1)} km/h &divide; 3.6 = <span class="res">${fmt(pb.v_prev,2)} m/s</span></div>
      <div class="formula">t<sub>frenado</sub> = v<sub>prev</sub> / a<sub>decel</sub> = ${fmt(pb.v_prev,2)} / ${fmt(pb.a_decel,2)} = <span class="res">${fmt(pb.t_frenado,2)} s</span></div>
      <div class="formula">v<sub>curr</sub> = ${fmt(f.speed,1)} km/h &divide; 3.6 = <span class="res">${fmt(pb.v_curr,2)} m/s</span></div>
      <div class="formula">t<sub>arranque</sub> = v<sub>curr</sub> / a<sub>accel</sub> = ${fmt(pb.v_curr,2)} / ${fmt(pb.a_accel,2)} = <span class="res">${fmt(pb.t_arranque,2)} s</span></div>
      <div class="formula">t<sub>maniobra</sub> = <span class="res">${fmt(pb.t_maniobra,1)} s</span> <span style="color:var(--muted);font-size:11px;">(${pb.t_maniobra_reason || ''}, v_max=${fmt(pb.v_max,2)} m/s)</span></div>
      <div class="formula">t<sub>req</sub> = t<sub>frenado</sub> + t<sub>maniobra</sub> + t<sub>arranque</sub><br>
        = <span class="num">${fmt(pb.t_frenado,1)}</span> + <span class="num">${fmt(pb.t_maniobra,1)}</span> + <span class="num">${fmt(pb.t_arranque,1)}</span>
        = <span class="res">${fmt(pb.t_req,1)} s</span></div>
      <div class="formula">t<sub>sobra</sub> = &Delta;t &minus; t<sub>req</sub> = <span class="num">${f.delta_t}</span> &minus; <span class="num">${fmt(pb.t_req,1)}</span>
        = <span class="res">${fmt(pb.t_sobra,1)} s</span></div>
      <div class="formula">P<sub>fisica</sub> = 1 / (1 + e<sup>&minus;0.1&middot;t_sobra</sup>) = 1 / (1 + e<sup>&minus;0.1&middot;${fmt(pb.t_sobra,1)}</sup>) = <span class="res">${fmt(f.p_fisica,3)}</span></div>
    </div>`;
  } else {
    physPanel = `<div class="panel"><h4>Fisica del vehiculo</h4><div style="color:#888;font-size:12.5px;">Primer frame del viaje: no hay frame anterior para calcular &Delta;t.</div></div>`;
  }

  // --- rule checklist panel ---
  let rulePanel = "";
  if (f.pending_frames && f.pending_frames.length) {
    const support = f.support ?? 0;
    const avgd = f.avg_dist_vs_old;
    const rows = [
      checkRow("dist_mem &ge; 0.50 (candidato)", f.dist_mem, TH.dist_mem, ">="),
      checkRow(`soporte &ge; ${TH.support} (de ventana ${TH.window})`, support, TH.support, ">="),
      checkRow("dist. promedio vs memoria vieja &ge; 0.85", avgd, TH.avg_dist, ">="),
      checkRow("P_fisica &ge; 0.60", f.p_fisica, TH.p_fisica, ">="),
    ].join("");
    const allOk = f.decision === 'CAMBIO_CONFIRMADO';
    rulePanel = `
    <div class="panel">
      <h4>Regla evaluada en este frame</h4>
      ${rows}
      <div class="final-verdict ${allOk ? 'yes' : 'no'}">${allOk ? 'CAMBIO_CONFIRMADO &#10003;' : 'Aun no confirma &rarr; POSIBLE_CAMBIO'}</div>
    </div>`;
  } else if (f.decision === 'CAMBIO_CONFIRMADO') {
    rulePanel = `
    <div class="panel">
      <h4>Regla evaluada en este frame</h4>
      ${checkRow("dist_mem &ge; 0.50 (candidato)", f.dist_mem, TH.dist_mem, ">=")}
      ${checkRow(`soporte &ge; ${TH.support} (de ventana ${TH.window})`, (f.coherent_frames||[]).length, TH.support, ">=")}
      ${checkRow("dist. promedio vs memoria vieja &ge; 0.85", f.avg_dist_vs_old, TH.avg_dist, ">=")}
      ${checkRow("P_fisica &ge; 0.60", f.p_fisica, TH.p_fisica, ">=")}
      <div class="final-verdict yes">CAMBIO_CONFIRMADO &#10003;</div>
    </div>`;
  } else {
    rulePanel = `
    <div class="panel">
      <h4>Regla evaluada en este frame</h4>
      <div style="color:#888;font-size:12.5px;">dist_mem = ${fmt(f.dist_mem,3)} &lt; 0.50 &rarr; match directo con la memoria, no se evalua la regla de confirmacion.</div>
    </div>`;
  }

  return left + `<div class="right-grid">${memPanel}${winPanel}${winEvolutionPanel}${physPanel}${rulePanel}</div>`;
}

function openLightbox(src) {
  document.getElementById("lightbox-img").src = src;
  document.getElementById("lightbox").classList.add("open");
}
function closeLightbox() {
  document.getElementById("lightbox").classList.remove("open");
}

// ---------------------------------------------------------------
// Comparacion directa: ahora es un caso interactivo mas (no una seccion
// fija al final). Dos columnas independientes, cada una navegable igual
// que un caso normal.
// ---------------------------------------------------------------
function renderComparisonCase(c) {
  const okCase = findCase(c.leftId);
  const fpCase = findCase(c.rightId);
  if (cmpLeftIdx === null) cmpLeftIdx = defaultFrameIdx(okCase);
  if (cmpRightIdx === null) cmpRightIdx = defaultFrameIdx(fpCase);

  const html = `
  <div class="case-header">
    <h2>${c.titulo}</h2>
    <div class="cmp-intro">
      Ambos casos cumplen la regla con numeros muy parecidos (dist_mem alto, distancia promedio &ge; 0.85,
      fisica favorable, 3+ observaciones coherentes). El algoritmo no tiene forma de distinguirlos solo con la cara:
      la diferencia real esta en pose / iluminacion IR / lentes / angulo de camara, informacion que el embedding
      facial no logra separar de un cambio de identidad genuino en estos casos limite. Navega cada lado por
      separado (timeline y frame propios) para comparar el detalle completo.
    </div>
  </div>
  <div class="cmp-grid">
    <div class="cmp-col ok">
      <div class="cmp-col-title">CASO CORRECTO (cambio real)</div>
      ${buildCaseHeaderHtml(okCase, "RESULTADO HUMANO / GT")}
      <div class="panel chart-wrap"><h4>Evolucion de metricas</h4><div id="cmp-chart-left"></div></div>
      <div id="cmp-timeline-left" style="overflow-x:auto; padding: 6px 10px;"></div>
      <div id="cmp-detail-left" class="right-grid" style="margin:8px;"></div>
    </div>
    <div class="cmp-col fp">
      <div class="cmp-col-title">FALSO POSITIVO (misma persona)</div>
      ${buildCaseHeaderHtml(fpCase, "RESULTADO HUMANO / GT")}
      <div class="panel chart-wrap"><h4>Evolucion de metricas</h4><div id="cmp-chart-right"></div></div>
      <div id="cmp-timeline-right" style="overflow-x:auto; padding: 6px 10px;"></div>
      <div id="cmp-detail-right" class="right-grid" style="margin:8px;"></div>
    </div>
  </div>`;

  document.getElementById("case-view").innerHTML = html;
  renderCmpSide("left", okCase, cmpLeftIdx);
  renderCmpSide("right", fpCase, cmpRightIdx);
}

function selectCmpFrame(side, li) {
  const c = CASES[currentCaseIdx];
  if (side === "left") { cmpLeftIdx = li; renderCmpSide("left", findCase(c.leftId), li); }
  else { cmpRightIdx = li; renderCmpSide("right", findCase(c.rightId), li); }
}

function renderCmpSide(side, caseObj, li) {
  document.getElementById(`cmp-chart-${side}`).innerHTML = buildMetricsChartHtml(caseObj, li, `selectCmpFrame_${side}`);
  document.getElementById(`cmp-timeline-${side}`).innerHTML = `<div id="timeline" style="display:flex;gap:8px;">` +
    buildTimelineHtml(caseObj, li, `selectCmpFrame_${side}`) + `</div>`;
  document.getElementById(`cmp-detail-${side}`).innerHTML = buildDetailHtml(caseObj, li);
}
function selectCmpFrame_left(li) { selectCmpFrame("left", li); }
function selectCmpFrame_right(li) { selectCmpFrame("right", li); }

renderFlow();
renderTabs();
selectCase(0);
markActiveTab();
</script>
</body>
</html>
"""

html_out = HTML.replace("__CASES_JSON__", CASES_JSON)
with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(html_out)
print(f"HTML generado en: {HTML_PATH}")
