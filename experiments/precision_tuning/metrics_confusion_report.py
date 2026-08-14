"""
Metricas + matriz de confusion de la config congelada "3-de-4"
(E_w4_s3_c035_d085_p06, winner_config.json) sobre TODO DEV (1416 viajes).

Por que existe este script (y no se reusa full_dev_results.csv): el GT fue
corregido despues de la revision manual de los 45 FP (ver
precision_tuning/candidate_fix/correction_log_real_changes.csv, 9 viajes
reclasificados de "FP" a "cambio real"). full_dev_results.csv quedo con el
GT VIEJO. Este script recalcula el GT desde los CSV ya corregidos
(build_gt_for_trip) y corre memory v1 + config congelada sobre DEV para que
las metricas reflejen la correccion.

NO es un experimento nuevo, NO cambia el algoritmo ni sus umbrales
(reusa algo_v1_param.run_v1_param con winner_config.json tal cual), NO toca
TEST (solo lee splits.csv y filtra split=="DEV").

Salida (experiments/precision_tuning/outputs_metrics_confusion/):
  - metrics_dashboard.png  (panel elegante, paleta apta para daltonismo:
    Okabe-Ito / colormap cividis, sin depender de rojo/verde para
    distinguir "bien" de "mal")
  - metrics_summary.json  (todas las metricas + frecuencia de error, para
    poder citarlas en texto sin tener que releer el PNG)

Uso: uv run python metrics_confusion_report.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prototype_memory"))
from common import CSV_FILES, EXP_DIR, DATA_DIR, load_raw_csv, preprocess_trip, build_gt_for_trip, match_events_window
from evaluate import evaluate
from precompute import precompute_trips
from algo_v1_param import run_v1_param

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs_metrics_confusion")
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(HERE, "winner_config.json"), "r", encoding="utf-8") as f:
    WINNER_RAW = json.load(f)
FROZEN_PARAMS = {
    k: WINNER_RAW[k]
    for k in ("candidate_window", "candidate_min_support", "coherence_threshold",
              "min_avg_dist_confirm", "min_phys_confirm", "memory_size")
}
CONFIG_NAME = WINNER_RAW["config_name"]
WINDOW = 3  # mismo radio de matching que evaluate.py

# ---------- Paleta apta para daltonismo (Okabe-Ito) ----------
OI_BLUE = "#0072B2"
OI_ORANGE = "#E69F00"
OI_SKY = "#56B4E9"
OI_GREEN = "#009E73"
OI_YELLOW = "#F0E442"
OI_VERMILLION = "#D55E00"
OI_PURPLE = "#CC79A7"
OI_GREY = "#8B8B8B"
OI_BLACK = "#000000"


def build_gt_fresh(dev_trip_ids_by_dataset):
    """Recalcula GT desde los CSV crudos ya corregidos (NO lee
    gt_events_<dataset>.csv en disco, que puede estar desactualizado)."""
    gt_rows = []
    for name, trip_ids in dev_trip_ids_by_dataset.items():
        if not trip_ids:
            continue
        trip_ids = set(trip_ids)
        print(f"  recalculando GT fresco de {name} para {len(trip_ids)} viajes DEV...")
        df = load_raw_csv(CSV_FILES[name])
        df = df[df["trip_id"].isin(trip_ids)]
        for t in trip_ids:
            df_trip = df[df["trip_id"] == t]
            if df_trip.empty:
                continue
            df_clean = preprocess_trip(df_trip)
            if df_clean is None:
                continue
            for e in build_gt_for_trip(df_clean):
                e["trip_id"] = t
                gt_rows.append(e)
    gt_df = pd.DataFrame(gt_rows)
    if len(gt_df):
        gt_df["gt_suspect"] = gt_df["gt_suspect"].astype(bool)
    return gt_df


def build_detail_df(cache, params):
    all_records = []
    for trip_id, frames in cache.items():
        recs = run_v1_param(frames, params, trip_id=trip_id)
        all_records.extend(recs)
    return pd.DataFrame(all_records)


def trip_level_confusion(detail_df, gt_df, cache, window=WINDOW):
    """Matriz de confusion POR VIAJE (no por frame ni por evento suelto):
      TP: el viaje tuvo >=1 cambio real Y el sistema lo confirmo correctamente.
      FN: el viaje tuvo >=1 cambio real y el sistema NO lo confirmo (se
          quedo en POSIBLE_CAMBIO o ni eso).
      FP: el viaje NO tuvo ningun cambio real pero el sistema confirmo algo
          igual (por definicion, incorrecto).
      TN: el viaje no tuvo cambio real y el sistema no confirmo nada.
    Nota: un viaje con cambio real que ADEMAS tiene una confirmacion
    adicional que no matchea su propio evento cuenta como TP en esta matriz
    (se prioriza si el cambio real fue detectado o no); esa confirmacion
    extra sigue sumando al conteo global de FP_estricto (fp_df de
    evaluate()), solo no genera una fila propia en esta matriz de 4 celdas."""
    detail_df = detail_df.copy()
    detail_df["frame_idx"] = detail_df.groupby("ASSET_ID").cumcount()

    all_trip_ids = set(cache.keys())
    gt_by_trip = {t: g for t, g in gt_df.groupby("trip_id")} if len(gt_df) else {}

    trips_with_change, trips_without_change = set(), set()
    tp_trips, fn_trips, fp_trips, tn_trips = set(), set(), set(), set()

    for t in all_trip_ids:
        d = detail_df[detail_df["ASSET_ID"] == t]
        confirm_frames = d.loc[d["DECISION_SISTEMA"] == "CAMBIO_CONFIRMADO", "frame_idx"].tolist()
        g = gt_by_trip.get(t)
        gt_frames = g["frame_idx"].dropna().astype(int).tolist() if g is not None else []

        if gt_frames:
            trips_with_change.add(t)
            matches, _, _ = match_events_window(gt_frames, confirm_frames, window)
            if matches:
                tp_trips.add(t)
            else:
                fn_trips.add(t)
        else:
            trips_without_change.add(t)
            if confirm_frames:
                fp_trips.add(t)
            else:
                tn_trips.add(t)

    return {
        "n_dev_trips": len(all_trip_ids),
        "trips_with_change": len(trips_with_change),
        "trips_without_change": len(trips_without_change),
        "TP_trips": len(tp_trips),
        "FN_trips": len(fn_trips),
        "FP_trips": len(fp_trips),
        "TN_trips": len(tn_trips),
    }


def total_hours_analyzed(cache):
    total_seconds = 0.0
    for frames in cache.values():
        if len(frames) < 2:
            continue
        ts = [f["ts_seconds"] for f in frames]
        total_seconds += max(ts) - min(ts)
    return total_seconds / 3600.0


def make_dashboard(metrics, conf, freq, out_path):
    fig = plt.figure(figsize=(14, 9), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.42, wspace=0.5)
    

    # ---------- Panel A: matriz de confusion por viaje ----------
    ax0 = fig.add_subplot(gs[0, 0])
    mat = np.array([[conf["TP_trips"], conf["FN_trips"]],
                     [conf["FP_trips"], conf["TN_trips"]]])
    im = ax0.imshow(mat, cmap="cividis", vmin=0, vmax=mat.max() * 1.15)
    ax0.set_xticks([0, 1])
    ax0.set_yticks([0, 1])
    ax0.set_xticklabels(["Confirmo\ncorrectamente", "NO confirmo\n(perdido/posible)"], fontsize=9.5)
    ax0.set_yticklabels(["Viaje CON\ncambio real", "Viaje SIN\ncambio real"], fontsize=9.5)
    labels = np.array([["TP", "FN"], ["FP", "TN"]])
    for i in range(2):
        for j in range(2):
            val = mat[i, j]
            luminance = im.cmap(im.norm(val))[:3]
            txt_color = "white" if sum(luminance) < 1.5 else "black"
            ax0.text(j, i, f"{labels[i,j]}\n{val}", ha="center", va="center",
                     fontsize=13, fontweight="bold", color=txt_color)
    ax0.set_title("Matriz de confusion (por viaje)", fontsize=11, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax0, fraction=0.045, pad=0.12, shrink=0.85)
    cbar.ax.tick_params(labelsize=8)

    # ---------- Panel B: KPIs principales (barras horizontales) ----------
    ax1 = fig.add_subplot(gs[0, 1])
    kpi_names = ["Precision\n(confirmados)", "Recall\n(confirmado)", "Recall\n(cualquier alerta)"]
    kpi_vals = [metrics["precision_estricta_cambio_confirmado"],
                metrics["cambio_confirmado_correctos"] / metrics["gt_clean"] if metrics["gt_clean"] else 0.0,
                metrics["recall_sin_suspect_cualquier_alerta"]]
    colors = [OI_ORANGE, OI_BLUE, OI_SKY]
    y_pos = np.arange(len(kpi_names))
    ax1.barh(y_pos, kpi_vals, color=colors, edgecolor=OI_BLACK, linewidth=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(kpi_names, fontsize=9.5)
    ax1.set_xlim(0, 1.05)
    ax1.set_xlabel("proporcion")
    ax1.invert_yaxis()
    for i, v in enumerate(kpi_vals):
        ax1.text(v + 0.02, i, f"{v*100:.1f}%", va="center", fontsize=9.5, fontweight="bold")
    ax1.set_title("Indicadores clave", fontsize=11, fontweight="bold")
    ax1.spines[["top", "right"]].set_visible(False)

    # ---------- Panel C: conteos crudos (barras) ----------
    ax2 = fig.add_subplot(gs[1, 0])
    cat_names = ["Cambios reales\n(GT limpio)", "Confirmados\ncorrectos (TP)",
                 "Confirmados\nincorrectos (FP)", "Cambios reales\nperdidos"]
    cat_vals = [metrics["gt_clean"], metrics["cambio_confirmado_correctos"],
                metrics["cambio_confirmado_incorrectos_fp_estrictos"],
                metrics["gt_clean"] - round(metrics["recall_sin_suspect_cualquier_alerta"] * metrics["gt_clean"])]
    cat_colors = [OI_GREY, OI_GREEN, OI_VERMILLION, OI_PURPLE]
    bars = ax2.bar(cat_names, cat_vals, color=cat_colors, edgecolor=OI_BLACK, linewidth=0.6)
    for b, v in zip(bars, cat_vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + max(cat_vals) * 0.015, str(v),
                  ha="center", fontsize=9.5, fontweight="bold")
    ax2.set_title("Conteos absolutos (DEV completo)", fontsize=11, fontweight="bold")
    ax2.tick_params(axis="x", labelsize=8.5)
    ax2.spines[["top", "right"]].set_visible(False)

    # ---------- Panel D: frecuencia de error (texto) ----------
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    lines = [
        ("¿Cada cuanto nos equivocamos?", True),
        (f"1 falso positivo (CAMBIO_CONFIRMADO incorrecto) cada "
         f"{freq['trips_per_fp']:.0f} viajes analizados.", False),
        (f"1 falso positivo cada {freq['hours_per_fp']:.1f} horas de video analizado.", False),
        (f"{freq['fp_per_1000_trips']:.1f} falsos positivos por cada 1000 viajes.", False),
        (f"1 cambio real perdido (ni siquiera quedo como posible) cada "
         f"{freq['changes_per_miss']:.1f} cambios reales." if freq["changes_per_miss"] else
         "Ningun cambio real fue perdido en silencio (recall cualquier-alerta = 100%).", False),
        (f"Delay medio de confirmacion: {metrics['delay_medio_frames_confirmado']:.1f} frames "
         f"(max {metrics['delay_max_frames_confirmado']:.0f}).", False),
       
    ]
    y = 0.95
    for text, is_title in lines:
        ax3.text(0.0, y, text, fontsize=11.5 if is_title else 10, fontweight="bold" if is_title else "normal",
                  color=OI_BLACK, va="top", wrap=True, transform=ax3.transAxes)
        y -= 0.14 if is_title else 0.135
    ax3.set_title("Frecuencia de error", fontsize=11, fontweight="bold", loc="left")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    splits = pd.read_csv(os.path.join(EXP_DIR, "splits", "splits.csv"))
    dev = splits[splits["split"] == "DEV"].copy()
    trip_ids_by_dataset = {
        name: dev.loc[dev["dataset"] == name, "trip_id"].tolist()
        for name in dev["dataset"].unique()
    }
    n_dev = len(dev)
    print(f"Config: {CONFIG_NAME} -> {FROZEN_PARAMS}")
    print(f"Precomputando TODO DEV ({n_dev} viajes) -- puede tardar varios minutos...")
    cache = precompute_trips(trip_ids_by_dataset)
    print(f"  {len(cache)} viajes precomputados, {sum(len(v) for v in cache.values())} frames totales")

    print("Recalculando GT desde los CSV corregidos (identity_id ya con el fix de cambios reales)...")
    gt_df = build_gt_fresh(trip_ids_by_dataset)
    gt_df = gt_df[gt_df["trip_id"].isin(set(cache.keys()))] if len(gt_df) else gt_df

    print(f"  corriendo {CONFIG_NAME} sobre {len(cache)} viajes...")
    detail_df = build_detail_df(cache, FROZEN_PARAMS)
    metrics, tp_df, fp_df = evaluate(detail_df, gt_df, window=WINDOW, label=CONFIG_NAME)

    conf = trip_level_confusion(detail_df, gt_df, cache, window=WINDOW)

    total_hours = total_hours_analyzed(cache)
    fp_total = metrics["cambio_confirmado_incorrectos_fp_estrictos"]
    misses = metrics["gt_clean"] - round(metrics["recall_sin_suspect_cualquier_alerta"] * metrics["gt_clean"])
    freq = {
        "trips_per_fp": (n_dev / fp_total) if fp_total else float("inf"),
        "hours_per_fp": (total_hours / fp_total) if fp_total else float("inf"),
        "fp_per_1000_trips": 1000 * fp_total / n_dev,
        "changes_per_miss": (metrics["gt_clean"] / misses) if misses else 0.0,
        "posible_por_1000_trips": 1000 * metrics["posible_cambio_total"] / n_dev,
        "total_hours_analyzed": total_hours,
    }

    print("\n=== METRICAS ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("\n=== MATRIZ DE CONFUSION (por viaje) ===")
    for k, v in conf.items():
        print(f"  {k}: {v}")
    print("\n=== FRECUENCIA DE ERROR ===")
    for k, v in freq.items():
        print(f"  {k}: {v}")

    png_path = os.path.join(OUT_DIR, "metrics_dashboard.png")
    make_dashboard(metrics, conf, freq, png_path)
    print(f"\nDashboard guardado en: {png_path}")

    summary = {"config_name": CONFIG_NAME, "params": FROZEN_PARAMS, "n_dev_trips": n_dev,
               "metrics": metrics, "confusion_por_viaje": conf, "frecuencia_de_error": freq}
    json_path = os.path.join(OUT_DIR, "metrics_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=lambda o: None if isinstance(o, float) and (o != o) else o)
    print(f"Resumen JSON guardado en: {json_path}")

    fp_df.to_csv(os.path.join(OUT_DIR, "fp_confirmado_dev.csv"), index=False)
    tp_df.to_csv(os.path.join(OUT_DIR, "tp_confirmado_dev.csv"), index=False)


if __name__ == "__main__":
    main()
