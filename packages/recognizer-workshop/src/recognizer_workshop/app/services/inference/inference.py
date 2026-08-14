from pathlib import Path
import logging

import numpy as np
import pandas as pd

from recognizer_workshop.perch_hoplite_custom_modifications import brutalism
from recognizer_workshop.app.services.inference.helpers import make_site_map
from recognizer_workshop.app.services.verification import sample_for_verification
from baw_helpers import baw_utils

from recognizer_workshop.perch_hoplite_custom_modifications import classifier as classifier_mod

log = logging.getLogger(__name__)

SUBSET_CHUNK_SIZE = 100_000


def _create_score_shaped_inference_subset(
    inference_csv: str | Path,
    subset_csv: str | Path,
    limit: int,
) -> int:
    """Preserve the original score-shaped random sampler for future use."""
    frame = pd.read_csv(inference_csv)
    if frame.empty:
        frame.to_csv(subset_csv, index=False)
        return 0

    selected_frames = []
    for _, label_frame in frame.groupby('label', sort=False):
        sampled_indices, _, _, _ = sample_for_verification.sample_for_validation(
            label_frame['score'].values,
            pos_target_y=0.05,
            neg_target_y=0.05,
            sample_size=limit,
        )
        selected_frames.append(label_frame.iloc[sampled_indices])

    subset_frame = pd.concat(selected_frames).reset_index(drop=True)
    subset_frame.to_csv(subset_csv, index=False)
    return len(subset_frame)


def _select_subset_row_indices(
    inference_csv: str | Path,
    limit: int,
    mode: str,
) -> list[int]:
    """Select row numbers without loading the full inference CSV."""
    columns = pd.read_csv(inference_csv, nrows=0).columns
    required_columns = {'label'} | ({'score'} if mode == 'max' else set())
    missing_columns = required_columns.difference(columns)
    if missing_columns:
        missing = ', '.join(sorted(missing_columns))
        raise ValueError(f"Inference CSV is missing required column(s): {missing}")

    if mode == 'max':
        top_by_label: dict[object, pd.DataFrame] = {}
        row_offset = 0
        for chunk in pd.read_csv(
            inference_csv,
            usecols=['label', 'score'],
            chunksize=SUBSET_CHUNK_SIZE,
        ):
            chunk = chunk.reset_index(drop=True)
            chunk['_row_index'] = range(row_offset, row_offset + len(chunk))
            for label, label_frame in chunk.groupby('label', sort=False):
                candidates = label_frame.nlargest(limit, 'score')
                previous = top_by_label.get(label)
                if previous is not None:
                    candidates = pd.concat([previous, candidates], ignore_index=True)
                    candidates = candidates.nlargest(limit, 'score')
                top_by_label[label] = candidates
            row_offset += len(chunk)
        return [
            int(row_index)
            for label_frame in top_by_label.values()
            for row_index in label_frame['_row_index']
        ]

    # Reservoir sampling gives every row equal probability while retaining at
    # most `limit` row numbers for each label.
    reservoirs: dict[object, list[int]] = {}
    seen_by_label: dict[object, int] = {}
    rng = np.random.default_rng()
    row_offset = 0
    for chunk in pd.read_csv(
        inference_csv,
        usecols=['label'],
        chunksize=SUBSET_CHUNK_SIZE,
    ):
        for local_index, label in enumerate(chunk['label']):
            row_index = row_offset + local_index
            seen = seen_by_label.get(label, 0) + 1
            seen_by_label[label] = seen
            reservoir = reservoirs.setdefault(label, [])
            if len(reservoir) < limit:
                reservoir.append(row_index)
            else:
                replacement_index = int(rng.integers(0, seen))
                if replacement_index < limit:
                    reservoir[replacement_index] = row_index
        row_offset += len(chunk)
    return [row_index for reservoir in reservoirs.values() for row_index in reservoir]


def _select_global_random_row_indices(
    inference_csv: str | Path,
    limit: int,
) -> list[int]:
    """Select up to ``limit`` rows uniformly across the entire CSV."""
    reservoir: list[int] = []
    rng = np.random.default_rng()
    row_offset = 0
    for chunk in pd.read_csv(inference_csv, chunksize=SUBSET_CHUNK_SIZE):
        for local_index in range(len(chunk)):
            row_index = row_offset + local_index
            seen = row_index + 1
            if len(reservoir) < limit:
                reservoir.append(row_index)
            else:
                replacement_index = int(rng.integers(0, seen))
                if replacement_index < limit:
                    reservoir[replacement_index] = row_index
        row_offset += len(chunk)
    return reservoir


def _write_selected_inference_rows(
    inference_csv: str | Path,
    subset_csv: str | Path,
    selected_row_indices: list[int],
) -> int:
    """Write selected full rows in the order chosen by the first pass."""
    header = pd.read_csv(inference_csv, nrows=0)
    if not selected_row_indices:
        header.to_csv(subset_csv, index=False)
        return 0

    selected_set = set(selected_row_indices)
    selection_order = {row_index: order for order, row_index in enumerate(selected_row_indices)}
    selected_frames = []
    row_offset = 0
    for chunk in pd.read_csv(inference_csv, chunksize=SUBSET_CHUNK_SIZE):
        global_indices = range(row_offset, row_offset + len(chunk))
        selected_mask = [row_index in selected_set for row_index in global_indices]
        if any(selected_mask):
            selected = chunk.loc[selected_mask].copy()
            selected['_subset_order'] = [
                selection_order[row_index]
                for row_index, is_selected in zip(global_indices, selected_mask)
                if is_selected
            ]
            selected_frames.append(selected)
        row_offset += len(chunk)

    subset_frame = (
        pd.concat(selected_frames)
        .sort_values('_subset_order')
        .drop(columns='_subset_order')
        .reset_index(drop=True)
    )
    subset_frame.to_csv(subset_csv, index=False)
    return len(subset_frame)


def create_inference_subset(
    inference_csv: str | Path,
    subset_csv: str | Path,
    limit: int = 100,
    mode: str = 'random',
) -> int:
    """Write up to ``limit`` inference rows per label using the selected mode."""
    if limit < 1:
        raise ValueError('limit must be at least 1')
    if mode not in {'random', 'max', 'score_shaped'}:
        raise ValueError("mode must be 'random', 'max', or 'score_shaped'")

    if mode == 'score_shaped':
        return _create_score_shaped_inference_subset(inference_csv, subset_csv, limit)

    selected_row_indices = _select_subset_row_indices(inference_csv, limit, mode)
    subset_count = _write_selected_inference_rows(
        inference_csv,
        subset_csv,
        selected_row_indices,
    )
    log.info(
        'Created %s inference subset of %d rows (%d per label) from %s',
        mode,
        subset_count,
        limit,
        inference_csv,
    )
    return subset_count


def cap_inference_rows(
    inference_csv: str | Path,
    subset_csv: str | Path,
    limit: int,
) -> int:
    """Write a uniform random sample of at most ``limit`` rows from a CSV."""
    if limit < 1:
        raise ValueError('limit must be at least 1')
    selected_row_indices = _select_global_random_row_indices(inference_csv, limit)
    return _write_selected_inference_rows(
        inference_csv,
        subset_csv,
        selected_row_indices,
    )

def run_inference(classifier,
                  db, 
                  output_filepath: str | Path, 
                  baw_api,
                  threshold=0.0, 
                  labels=None, 
                  dataset=None, 
                  subset=1, 
                  exclude_labeled=False,
                  warning_callback=None):
    """
    Calls the classifier.write_inference_csv() function to write the inference results to a csv file
    injecting the site_id and link columns via a row_func that uses baw_utils and 
    """

    if subset is None:
        subset = 1.0
    subset = float(subset)

    if subset <= 0 or subset > 1:
        raise ValueError('Subset must be between 0 and 1')
    
    embedding_ids = brutalism.get_brute_search_ids(db, subset, exclude_labeled = exclude_labeled)

    # if subset < 1:
    #     n_samples = int(subset * len(all_embedding_ids))
    #     indices = np.arange(len(all_embedding_ids))
    #     np.random.shuffle(indices)
    #     selected_indices = sorted(indices[:n_samples])
    #     embedding_ids = all_embedding_ids[selected_indices]
    # else:
    #     embedding_ids = all_embedding_ids

    log.info(f'classifying {len(embedding_ids)} ({round(subset*100,0)}%) embeddings from dataset {dataset} ({round(len(embedding_ids) / 12 / 60, 2)} hours) with threshold {threshold}')

    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)



    site_lookup = None
    try:
        site_lookup = make_site_map(db, baw_api)
    except Exception as exc:
        message = f'Site information was not retrieved: {exc}'
        log.warning(message)
        if warning_callback is not None:
            warning_callback(message)

    def row_func(row=None):
        if row is None:
            headers = ['end_offset', 'audio_recording_id', 'link']
            if site_lookup is not None:
                headers.extend(['site_id', 'site_name'])
            return headers
        baw_domain, arid  = baw_utils.extract_arid_and_domain(row[2])
        link = baw_utils.make_baw_listen_link_from_arid(arid, row[3], 5.0, baw_domain)
        end_offset = row[3] + 5.0  # assuming the window size is 5 seconds
        values = [end_offset, arid, link]
        if site_lookup is not None:
            site_info = site_lookup.get(int(arid), {}) if arid is not None else {}
            values.extend([site_info.get('site_id', ''), site_info.get('site_name', '')])
        return values
        
    classifier_mod.write_inference_csv(classifier,
                                    db,
                                    str(output_filepath),
                                    threshold,
                                    labels,
                                    embedding_ids,
                                    row_func=row_func)
