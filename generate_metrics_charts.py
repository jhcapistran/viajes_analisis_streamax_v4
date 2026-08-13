"""
GRÁFICAS DE MÉTRICAS DE DETECCIÓN DE CAMBIO DE CONDUCTOR
==========================================================

Corre el pipeline de main_analisis_completo_v2.py, calcula las mismas
métricas que se imprimen en la tabla "🎯 MÉTRICAS DE DETECCIÓN" (Recall,
Precisión, Precisión Estricta, F1) y genera gráficas PNG en
outputs_metrics_charts/.

Uso:
    python generate_metrics_charts.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from main_analisis_completo_v2 import load_and_process, compute_gt_changes, INPUT_FILE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "outputs_metrics_charts")


def compute_metrics(out_df):
    gt_changes = compute_gt_changes(out_df)
    gt_total = len(gt_changes)
    gt_detected = sum(1 for c in gt_changes if c['detected'])

    tp_positions_by_asset = {}
    for c in gt_changes:
        if c['detected']:
            tp_positions_by_asset.setdefault(c['asset_id'], set()).add(c['cur_start'])

    fp_confirmado = 0
    fp_posible = 0
    for asset_id, group in out_df.groupby('ASSET_ID', sort=False):
        decision = group['DECISION_SISTEMA'].tolist()
        tp_positions = tp_positions_by_asset.get(asset_id, set())
        for pos, dec in enumerate(decision):
            if dec in ('POSIBLE_CAMBIO', 'CAMBIO_CONFIRMADO') and pos not in tp_positions:
                if dec == 'CAMBIO_CONFIRMADO':
                    fp_confirmado += 1
                else:
                    fp_posible += 1
    fp_total = fp_confirmado + fp_posible

    recall = (gt_detected / gt_total) if gt_total else 0.0
    precision_total = (gt_detected / (gt_detected + fp_total)) if (gt_detected + fp_total) else 0.0
    precision_estricta = (gt_detected / (gt_detected + fp_confirmado)) if (gt_detected + fp_confirmado) else 0.0

    def f1(p, r):
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    return {
        'gt_total': gt_total,
        'gt_detected': gt_detected,
        'fp_confirmado': fp_confirmado,
        'fp_posible': fp_posible,
        'fp_total': fp_total,
        'recall': recall,
        'precision_total': precision_total,
        'precision_estricta': precision_estricta,
        'f1_total': f1(precision_total, recall),
        'f1_estricta': f1(precision_estricta, recall),
    }


def plot_metrics(m):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Recall / Precisión (total y estricta) / F1 (total y estricto)
    labels = ['Recall', 'Precisión\n(total)', 'Precisión\nEstricta', 'F1\n(total)', 'F1\nEstricto']
    values = [m['recall'], m['precision_total'], m['precision_estricta'], m['f1_total'], m['f1_estricta']]
    colors = ['#4a90d9', '#e07b39', '#f5a623', '#e07b39', '#7ed321']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, [v * 100 for v in values], color=colors)
    ax.set_ylim(0, 110)
    ax.set_ylabel('%')
    ax.set_title('Métricas de Detección de Cambio de Conductor')
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{v * 100:.1f}%", ha='center', fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'metricas_resumen.png'), dpi=150)
    plt.close(fig)

    # 2) Desglose de Falsos Positivos por tipo de decisión
    fig, ax = plt.subplots(figsize=(6, 5))
    labels_fp = ['POSIBLE_CAMBIO\n(advertencia)', 'CAMBIO_CONFIRMADO\n(compromiso firme)']
    values_fp = [m['fp_posible'], m['fp_confirmado']]
    bars = ax.bar(labels_fp, values_fp, color=['#f5a623', '#ff3b30'])
    ax.set_ylabel('Cantidad de Falsos Positivos')
    ax.set_title(f"Falsos Positivos por Tipo de Alerta (total: {m['fp_total']})")
    for bar, v in zip(bars, values_fp):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(v), ha='center', fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'falsos_positivos_por_tipo.png'), dpi=150)
    plt.close(fig)

    # 3) Cambios Reales (GT) vs Detectados
    fig, ax = plt.subplots(figsize=(6, 5))
    labels_gt = ['Cambios Reales (GT)', 'Detectados']
    values_gt = [m['gt_total'], m['gt_detected']]
    bars = ax.bar(labels_gt, values_gt, color=['#4a90d9', '#7ed321'])
    ax.set_ylabel('Cantidad')
    ax.set_title('Cambios Reales vs Detectados')
    for bar, v in zip(bars, values_gt):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, str(v), ha='center', fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'gt_vs_detectados.png'), dpi=150)
    plt.close(fig)


def main():
    print("🚀 Corriendo pipeline para calcular métricas...")
    out_df = load_and_process(INPUT_FILE)
    if out_df is None:
        print("❌ No se pudo generar el detalle. Abortando.")
        return

    m = compute_metrics(out_df)
    print(f"ℹ️ GT: {m['gt_total']} | Detectados: {m['gt_detected']} | Recall: {m['recall']*100:.2f}%")
    print(f"ℹ️ FP total: {m['fp_total']} (POSIBLE_CAMBIO: {m['fp_posible']}, CAMBIO_CONFIRMADO: {m['fp_confirmado']})")
    print(f"ℹ️ Precisión total: {m['precision_total']*100:.2f}% | Precisión Estricta: {m['precision_estricta']*100:.2f}%")
    print(f"ℹ️ F1 total: {m['f1_total']*100:.2f}% | F1 Estricto: {m['f1_estricta']*100:.2f}%")

    print("🖨️ Generando gráficas...")
    plot_metrics(m)
    print(f"✅ Gráficas guardadas en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
