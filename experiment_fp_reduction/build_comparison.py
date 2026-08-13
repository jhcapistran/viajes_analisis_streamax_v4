"""
Genera la tabla comparativa final baseline vs Prototype Memory para el
reporte (a partir de los CSVs ya generados por run_baseline.py /
run_memory_algo.py con la configuracion final de algo_memory.py).
"""
import os
import pandas as pd

from common import DATA_DIR
from evaluate import evaluate

rows = []
for name in ["random_04", "reviewed_07"]:
    gt = pd.read_csv(os.path.join(DATA_DIR, f"gt_events_{name}.csv"))
    gt['gt_suspect'] = gt['gt_suspect'].astype(bool)

    base_detail = pd.read_csv(os.path.join(DATA_DIR, f"baseline_detail_{name}.csv"))
    mem_detail = pd.read_csv(os.path.join(DATA_DIR, f"memory_detail_{name}.csv"))

    m_base, _, _ = evaluate(base_detail, gt, label=f"baseline_{name}")
    m_mem, _, _ = evaluate(mem_detail, gt, label=f"memory_{name}")

    rows.append(m_base)
    rows.append(m_mem)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(DATA_DIR, "comparison_summary.csv"), index=False)
print(df.to_string(index=False))
