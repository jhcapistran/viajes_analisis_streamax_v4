"""
Precomputo de frames (embeddings ya parseados + speed + ts_seconds) para un
conjunto de trip_id, reutilizable por el grid search sin tener que releer
los CSV crudos ni re-parsear embeddings en cada configuracion probada.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prototype_memory"))
from common import CSV_FILES, load_raw_csv, preprocess_trip, load_emb


def precompute_trips(trip_ids_by_dataset):
    """trip_ids_by_dataset: dict dataset_name -> set/list de trip_id.
    Devuelve dict trip_id -> lista de frames precomputados (dicts)."""
    cache = {}
    for name, trip_ids in trip_ids_by_dataset.items():
        if not trip_ids:
            continue
        trip_ids = set(trip_ids)
        path = CSV_FILES[name]
        print(f"  cargando {name} crudo para precomputar {len(trip_ids)} viajes...")
        df = load_raw_csv(path)
        df = df[df['trip_id'].isin(trip_ids)]

        for t in trip_ids:
            df_trip = df[df['trip_id'] == t]
            if df_trip.empty:
                continue
            df_clean = preprocess_trip(df_trip)
            if df_clean is None:
                continue
            frames = []
            for _, row in df_clean.iterrows():
                frames.append({
                    'emb': load_emb(row.get('embedding')),
                    'speed': float(row.get('speed', 0.0)),
                    'ts_seconds': int(row['ts_seconds']),
                    'driver_id': row.get('driver_id', 'N/A'),
                    'identity_id': row.get('identity_id', ''),
                    'empty_cabin': row.get('empty_cabin', ''),
                    'image_file': row.get('image_file', 'N/A'),
                })
            cache[t] = frames
    return cache
