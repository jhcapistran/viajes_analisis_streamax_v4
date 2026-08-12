"""Genera un HTML de revisión visual para los 5 casos de miss_review_manifest_v2.csv.

Para cada caso muestra la secuencia de imágenes (ventana alrededor del timestamp
del cambio detectado), con el identity_id de cada fila y resaltando la fila
donde se detectó el cambio.
"""

from pathlib import Path

import pandas as pd

MANIFEST_CSV = Path("outputs_csv_comparison/miss_review_manifest_v2.csv")
IMAGES_DIR = Path("outputs_csv_comparison/miss_review_images_v2")
OUT_HTML = Path("outputs_csv_comparison/miss_review_v2.html")


def local_filename(gs_path: str) -> str:
    p = Path(gs_path)
    parent_name = p.parent.name
    file_name = p.name
    if parent_name and parent_name not in ("", ".", "/"):
        clean_parent = parent_name.replace("-", "")
        return f"{clean_parent}_{file_name}"
    return file_name


def main() -> None:
    df = pd.read_csv(MANIFEST_CSV, dtype=str)
    df["is_event_row"] = df["is_event_row"].astype(str).str.lower() == "true"

    parts = ["""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Revision GT misses v2</title>
<style>
body { font-family: sans-serif; background:#111; color:#eee; }
.case { margin-bottom: 60px; border-bottom: 3px solid #444; padding-bottom: 20px; }
.row-imgs { display:flex; flex-wrap:wrap; gap:14px; }
.cell { text-align:center; width:340px; }
.cell img { width:340px; border:3px solid #333; cursor: zoom-in; }
.event .cell img { border-color:red; }
.label { font-size:15px; margin-top:4px; }
.label b { color:#ff6; }
h2 { color:#5cf; position: sticky; top:0; background:#111; padding:8px 0; }
</style>
<script>
function zoom(img) {
  img.style.width = (img.style.width === '900px') ? '340px' : '900px';
}
</script>
</head><body>"""]

    only_cases = {"miss_02", "miss_03"}
    for case_id, g in df.groupby("case_id", sort=False):
        if case_id not in only_cases:
            continue
        g = g.sort_values("timestamp")
        trip_id = g["trip_id"].iloc[0]
        parts.append(f'<div class="case"><h2>{case_id} - trip {trip_id}</h2><div class="row-imgs">')
        for _, row in g.iterrows():
            fname = local_filename(row["gs_path"])
            img_path = IMAGES_DIR / fname
            cls = "cell event" if row["is_event_row"] else "cell"
            label = row["identity_id"] if pd.notna(row["identity_id"]) else "(sin cara / N/A)"
            marker = " <b>&larr; EVENTO</b>" if row["is_event_row"] else ""
            parts.append(
                f'<div class="{cls}"><img onclick="zoom(this)" src="{img_path.resolve()}">'
                f'<div class="label">{row["timestamp"]}<br><b>{label}</b>{marker}</div></div>'
            )
        parts.append("</div></div>")

    parts.append("</body></html>")
    OUT_HTML.write_text("\n".join(parts), encoding="utf-8")
    print(f"HTML generado en {OUT_HTML}")


if __name__ == "__main__":
    main()
