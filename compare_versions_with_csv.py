import csv
import importlib.util
import io
import time
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd


CSV_INPUT = Path("random_trips_data_2026_04.csv")
OUTPUT_DIR = Path("outputs_csv_comparison")
SUMMARY_CSV = OUTPUT_DIR / "version_summary.csv"
TRIP_METRICS_CSV = OUTPUT_DIR / "trip_metrics.csv"
SUMMARY_XLSX = OUTPUT_DIR / "comparison_metrics.xlsx"

CSV_FIELDS = [
    "trip_id",
    "asset_id",
    "timestamp",
    "gs_path",
    "speed",
    "identity_id",
    "embedding",
]

ALERT_DECISIONS = {"POSIBLE_CAMBIO", "CAMBIO_CONFIRMADO"}
VALID_DRIVER_VALUES = {"N/A", "nan", ""}

VERSIONS = {
    "v1_main": {
        "module_path": Path("main_analisis_completo.py"),
        "embedding_column": "embedding",
    },
    "v2_prod": {
        "module_path": Path("main_analisis_completo_v2.py"),
        "embedding_column": "embedding",
    },
    "v3_experimental": {
        "module_path": Path("main_analisis_completo_v3.py"),
        "embedding_column": "embedding_adaface",
    },
}


def load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"No se pudo cargar {module_path}")
    spec.loader.exec_module(module)
    return module


def is_non_empty_embedding(value: str) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return text != "" and text.lower() != "nan"


def build_image_file(asset_id: str, gs_path: str) -> str:
    basename = Path(str(gs_path)).name
    return f"{asset_id}_{basename}"


def parse_speed(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def normalize_trip_rows(rows: list[dict[str, str]], embedding_column: str) -> pd.DataFrame:
    normalized_rows = []

    for row in rows:
        embedding = row.get("embedding")
        if not is_non_empty_embedding(embedding):
            continue

        asset_id = str(row.get("asset_id", "unknown"))
        gs_path = str(row.get("gs_path", ""))
        basename = Path(gs_path).name

        normalized = {
            "trip_id": str(row.get("trip_id", "")),
            "asset_id": asset_id,
            "driver_id": row.get("identity_id") or "N/A",
            "image_file": build_image_file(asset_id, gs_path),
            "source_image": basename,
            "timestamp": row.get("timestamp", ""),
            "speed": parse_speed(row.get("speed", 0.0)),
            embedding_column: embedding,
        }
        normalized_rows.append(normalized)

    return pd.DataFrame(normalized_rows)


def iter_trip_groups(csv_path: Path):
    total_rows = 0
    total_rows_with_embedding = 0
    total_trips = 0

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        current_trip_id = None
        current_rows: list[dict[str, str]] = []

        for row in reader:
            total_rows += 1
            if is_non_empty_embedding(row.get("embedding")):
                total_rows_with_embedding += 1

            trip_id = str(row["trip_id"])
            if current_trip_id is None:
                current_trip_id = trip_id

            if trip_id != current_trip_id:
                total_trips += 1
                yield current_trip_id, current_rows
                current_trip_id = trip_id
                current_rows = []

            current_rows.append(row)

        if current_trip_id is not None:
            total_trips += 1
            yield current_trip_id, current_rows

    iter_trip_groups.total_rows = total_rows
    iter_trip_groups.total_rows_with_embedding = total_rows_with_embedding
    iter_trip_groups.total_trips = total_trips


iter_trip_groups.total_rows = 0
iter_trip_groups.total_rows_with_embedding = 0
iter_trip_groups.total_trips = 0


def write_records(writer, version_name: str, trip_id: str, source_asset_id: str, records: list[dict]):
    rows_to_write = []
    for record in records:
        row = dict(record)
        row["VERSION"] = version_name
        row["TRIP_ID"] = trip_id
        row["SOURCE_ASSET_ID"] = source_asset_id
        rows_to_write.append(row)
    writer.writerows(rows_to_write)


def compute_trip_metrics(version_name: str, trip_id: str, source_asset_id: str, out_df: pd.DataFrame) -> dict:
    decision_counts = out_df["DECISION_SISTEMA"].value_counts().to_dict()
    persona_frame_counts = out_df.groupby("PERSONA_ID").size()
    mini_ids = int((persona_frame_counts <= 2).sum())

    fragmented_drivers = 0
    if "DRIVER_ID" in out_df.columns:
        valid_driver_mask = ~out_df["DRIVER_ID"].astype(str).isin(VALID_DRIVER_VALUES)
        if valid_driver_mask.any():
            driver_persona_counts = (
                out_df.loc[valid_driver_mask].groupby("DRIVER_ID")["PERSONA_ID"].nunique()
            )
            fragmented_drivers = int((driver_persona_counts > 1).sum())

    return {
        "version": version_name,
        "trip_id": trip_id,
        "source_asset_id": source_asset_id,
        "rows_output": int(len(out_df)),
        "unique_personas": int(out_df["PERSONA_ID"].nunique()),
        "alerts": int(out_df["DECISION_SISTEMA"].isin(ALERT_DECISIONS).sum()),
        "manual_review": int((out_df["REQUIERE_VERIFICACION"] == "SI").sum()),
        "cambio_confirmado": int(decision_counts.get("CAMBIO_CONFIRMADO", 0)),
        "posible_cambio": int(decision_counts.get("POSIBLE_CAMBIO", 0)),
        "mismo_conductor": int(decision_counts.get("MISMO_CONDUCTOR", 0)),
        "indeterminado": int(decision_counts.get("INDETERMINADO", 0)),
        "inicio_viaje": int(decision_counts.get("INICIO_VIAJE", 0)),
        "mini_ids": mini_ids,
        "fragmented_drivers": fragmented_drivers,
    }


def init_version_state(version_name: str, module):
    OUTPUT_DIR.mkdir(exist_ok=True)
    result_path = OUTPUT_DIR / f"{version_name}_results.csv"
    handle = result_path.open("w", newline="")
    writer = None

    if hasattr(module, "GLOBAL_ID_COUNTER"):
        module.GLOBAL_ID_COUNTER = 0

    return {
        "module": module,
        "result_path": result_path,
        "handle": handle,
        "writer": writer,
        "trip_metrics": [],
        "decision_counts": Counter(),
        "rows_output": 0,
        "alerts": 0,
        "manual_review": 0,
        "total_ids": 0,
        "mini_ids": 0,
        "fragmented_drivers": 0,
        "trips_processed": 0,
        "trips_with_results": 0,
        "runtime_seconds": 0.0,
    }


def finalize_version_state(state: dict, total_trips: int, rows_input_with_embedding: int) -> dict:
    state["handle"].close()
    average_output_rows = state["rows_output"] / state["trips_with_results"] if state["trips_with_results"] else 0.0
    average_alerts = state["alerts"] / state["trips_with_results"] if state["trips_with_results"] else 0.0
    average_ids = state["total_ids"] / state["trips_with_results"] if state["trips_with_results"] else 0.0

    summary = {
        "version": state["name"],
        "rows_input_with_embedding": rows_input_with_embedding,
        "trips_total_csv": total_trips,
        "trips_processed": state["trips_processed"],
        "trips_with_results": state["trips_with_results"],
        "trips_without_results": total_trips - state["trips_with_results"],
        "rows_output": state["rows_output"],
        "total_persona_ids": state["total_ids"],
        "alerts": state["alerts"],
        "manual_review": state["manual_review"],
        "mini_ids": state["mini_ids"],
        "fragmented_drivers": state["fragmented_drivers"],
        "cambio_confirmado": state["decision_counts"].get("CAMBIO_CONFIRMADO", 0),
        "posible_cambio": state["decision_counts"].get("POSIBLE_CAMBIO", 0),
        "mismo_conductor": state["decision_counts"].get("MISMO_CONDUCTOR", 0),
        "indeterminado": state["decision_counts"].get("INDETERMINADO", 0),
        "inicio_viaje": state["decision_counts"].get("INICIO_VIAJE", 0),
        "avg_rows_output_per_trip": round(average_output_rows, 3),
        "avg_alerts_per_trip": round(average_alerts, 3),
        "avg_personas_per_trip": round(average_ids, 3),
        "runtime_seconds": round(state["runtime_seconds"], 3),
        "result_csv": str(state["result_path"]),
    }
    return summary


def main():
    if not CSV_INPUT.exists():
        raise FileNotFoundError(f"No existe el CSV de entrada: {CSV_INPUT}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    modules = {}
    for version_name, config in VERSIONS.items():
        modules[version_name] = load_module(version_name, config["module_path"])

    version_states = {}
    for version_name, module in modules.items():
        state = init_version_state(version_name, module)
        state["name"] = version_name
        version_states[version_name] = state

    overall_start = time.perf_counter()
    for trip_index, (trip_id, rows) in enumerate(iter_trip_groups(CSV_INPUT), start=1):
        source_asset_id = str(rows[0].get("asset_id", "unknown")) if rows else "unknown"

        for version_name, config in VERSIONS.items():
            state = version_states[version_name]
            module = state["module"]
            state["trips_processed"] += 1

            df_trip = normalize_trip_rows(rows, config["embedding_column"])
            if df_trip.empty:
                continue

            trip_start = time.perf_counter()
            with redirect_stdout(io.StringIO()):
                records = module.process_asset_group(df_trip, asset_name=str(trip_id))
            state["runtime_seconds"] += time.perf_counter() - trip_start

            if not records:
                continue

            out_df = pd.DataFrame(records)
            trip_metrics = compute_trip_metrics(version_name, str(trip_id), source_asset_id, out_df)
            state["trip_metrics"].append(trip_metrics)
            state["trips_with_results"] += 1
            state["rows_output"] += trip_metrics["rows_output"]
            state["alerts"] += trip_metrics["alerts"]
            state["manual_review"] += trip_metrics["manual_review"]
            state["total_ids"] += trip_metrics["unique_personas"]
            state["mini_ids"] += trip_metrics["mini_ids"]
            state["fragmented_drivers"] += trip_metrics["fragmented_drivers"]
            state["decision_counts"].update(out_df["DECISION_SISTEMA"].tolist())

            if state["writer"] is None:
                fieldnames = ["VERSION", "TRIP_ID", "SOURCE_ASSET_ID", *list(records[0].keys())]
                state["writer"] = csv.DictWriter(state["handle"], fieldnames=fieldnames)
                state["writer"].writeheader()

            write_records(state["writer"], version_name, str(trip_id), source_asset_id, records)

        if trip_index % 100 == 0:
            print(f"Procesados {trip_index} viajes...")

    total_runtime = time.perf_counter() - overall_start
    total_trips = iter_trip_groups.total_trips
    rows_input_with_embedding = iter_trip_groups.total_rows_with_embedding

    summary_rows = []
    trip_metrics_rows = []
    for version_name in VERSIONS:
        state = version_states[version_name]
        summary_rows.append(
            finalize_version_state(
                state,
                total_trips=total_trips,
                rows_input_with_embedding=rows_input_with_embedding,
            )
        )
        trip_metrics_rows.extend(state["trip_metrics"])

    summary_df = pd.DataFrame(summary_rows).sort_values("version").reset_index(drop=True)
    trip_metrics_df = pd.DataFrame(trip_metrics_rows).sort_values(["version", "trip_id"]).reset_index(drop=True)

    summary_df.to_csv(SUMMARY_CSV, index=False)
    trip_metrics_df.to_csv(TRIP_METRICS_CSV, index=False)

    with pd.ExcelWriter(SUMMARY_XLSX) as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        trip_metrics_df.to_excel(writer, sheet_name="trip_metrics", index=False)

    print(f"CSV total filas: {iter_trip_groups.total_rows}")
    print(f"CSV filas con embedding: {rows_input_with_embedding}")
    print(f"CSV viajes: {total_trips}")
    print(f"Tiempo total comparacion: {total_runtime:.2f}s")
    print(f"Resumen: {SUMMARY_CSV}")
    print(f"Metricas por viaje: {TRIP_METRICS_CSV}")
    print(f"Workbook: {SUMMARY_XLSX}")
    for row in summary_rows:
        print(
            f"{row['version']}: outputs={row['rows_output']}, "
            f"ids={row['total_persona_ids']}, alerts={row['alerts']}, "
            f"review={row['manual_review']}, runtime={row['runtime_seconds']}s"
        )


if __name__ == "__main__":
    main()
