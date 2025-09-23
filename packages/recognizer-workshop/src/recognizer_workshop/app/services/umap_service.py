import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import umap
from tqdm import tqdm

log = logging.getLogger(__name__)

# --- Helper Functions ---

def _get_umap_paths(db_path: Path) -> tuple[Path, Path]:
    """Generates the standard paths for the model and coordinate cache files."""
    model_path = db_path / "umap_model.joblib"
    coords_path = db_path / "umap_cache.parquet"
    return model_path, coords_path

def _fit_and_save_model(db, model_path: Path, subset_size: int):
    """Fits a UMAP reducer on a subset of data and saves the model to disk."""
    log.info(f"Fitting new UMAP model on a subset of {subset_size} points...")
    all_ids = db.get_embedding_ids()
    if len(all_ids) < subset_size:
        log.warning(f"Dataset has fewer points ({len(all_ids)}) than subset_size ({subset_size}). Using all points.")
        subset_size = len(all_ids)
    
    # Get a random subset of embeddings to fit on
    subset_ids = np.random.choice(all_ids, subset_size, replace=False)
    _, subset_embeddings = db.get_embeddings(subset_ids)

    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine', verbose=True)
    reducer.fit(subset_embeddings)
    
    joblib.dump(reducer, model_path)
    log.info(f"UMAP model saved to {model_path}")
    return reducer

def _transform_and_cache_coords(db, reducer, cache_path: Path, batch_size: int):
    """
    Transforms embeddings in batches and saves them to a Parquet cache file.
    This function performs an incremental update, only transforming new embeddings.
    """
    all_ids = db.get_embedding_ids()
    
    # Check if a cache already exists to find which embeddings are new
    if cache_path.exists():
        df_existing = pd.read_parquet(cache_path)
        existing_ids = set(df_existing['embedding_id'])
        ids_to_transform = np.array([i for i in all_ids if i not in existing_ids])
        log.info(f"Found {len(existing_ids)} existing coordinates. Transforming {len(ids_to_transform)} new embeddings.")
    else:
        df_existing = None
        ids_to_transform = all_ids
        log.info(f"No existing cache. Transforming all {len(ids_to_transform)} embeddings.")

    if len(ids_to_transform) == 0:
        log.info("No new embeddings to transform.")
        return

    # --- Transform in batches ---
    new_coords = []
    _, embeddings_to_transform = db.get_embeddings(ids_to_transform)
    
    for i in tqdm(range(0, len(embeddings_to_transform), batch_size), desc="Transforming Batches"):
        batch = embeddings_to_transform[i : i + batch_size]
        transformed_batch = reducer.transform(batch)
        new_coords.append(transformed_batch)
    
    df_new = pd.DataFrame({
        'embedding_id': ids_to_transform,
        'x': np.concatenate(new_coords, axis=0)[:, 0],
        'y': np.concatenate(new_coords, axis=0)[:, 1],
    })

    # Combine new coordinates with existing ones and save
    df_combined = pd.concat([df_existing, df_new]) if df_existing is not None else df_new
    df_combined.to_parquet(cache_path)
    log.info(f"UMAP coordinate cache updated at {cache_path}. Total points: {len(df_combined)}")

# --- Main Service Function (to be called by the UI) ---

def generate_umap_cache(db, db_path: Path, fit_subset_size=100_000, transform_batch_size=100_000):
    """
    Main entry point. Ensures a UMAP model exists, then incrementally transforms all embeddings.
    """
    model_path, coords_path = _get_umap_paths(db_path)

    # Step 1: Fit and save the model if it doesn't exist
    if not model_path.exists():
        reducer = _fit_and_save_model(db, model_path, subset_size=fit_subset_size)
    else:
        log.info(f"Loading existing UMAP model from {model_path}")
        reducer = joblib.load(model_path)

    # Step 2: Transform all embeddings (incrementally) and save the coordinates
    _transform_and_cache_coords(db, reducer, coords_path, batch_size=transform_batch_size)

# --- Data Loading Function (for the UI plot) ---

def load_umap_data(db, db_path: Path):
    """
    Loads the cached UMAP coordinates and merges them with current labels from the DB.
    """
    model_path, cache_path = _get_umap_paths(db_path)
    if not cache_path.exists():
        return None

    log.info(f"Loading UMAP cache from {cache_path}...")
    df_cache = pd.read_parquet(cache_path)

    all_labels = db.get_all_labels()
    # print("all_labels:", all_labels)
    # print(f'num labels: {len(all_labels)}')
    if all_labels:
        df_labels = pd.DataFrame(all_labels, columns=['embedding_id', 'label', 'type'])
        df_labels = df_labels.drop_duplicates(subset='embedding_id', keep='last')
        df_labels['composite_label'] = df_labels.apply(
            lambda row: f"{row['label']} ({'Positive' if row['type'] == 1 else 'Negative'})",
            axis=1
        )
        
        df_plot = pd.merge(df_cache, df_labels, on='embedding_id', how='left')
    else:
        df_plot = df_cache
        df_plot['label'] = None
        df_plot['composite_label'] = None

    df_plot['label'] = df_plot['label'].fillna('unlabeled')
    df_plot['composite_label'] = df_plot['composite_label'].fillna('unlabeled').astype('category')
    log.info("UMAP data loaded and merged with labels.")
    return df_plot