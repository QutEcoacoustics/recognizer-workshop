from pathlib import Path
import logging

import pandas as pd

from recognizer_workshop.perch_hoplite_custom_modifications import brutalism
from recognizer_workshop.app.services.inference.helpers import make_site_map
from recognizer_workshop.app.services.verification import sample_for_verification
from baw_helpers import baw_utils

from recognizer_workshop.perch_hoplite_custom_modifications import classifier as classifier_mod

log = logging.getLogger(__name__)


def create_inference_subset(
    inference_csv: str | Path,
    subset_csv: str | Path,
    limit: int = 100,
    mode: str = 'random',
) -> int:
    """Write up to ``limit`` inference rows per label using the selected mode."""
    if limit < 1:
        raise ValueError('limit must be at least 1')
    if mode not in {'random', 'max'}:
        raise ValueError("mode must be either 'random' or 'max'")

    frame = pd.read_csv(inference_csv)
    if frame.empty:
        frame.to_csv(subset_csv, index=False)
        return 0
    if 'label' not in frame.columns or 'score' not in frame.columns:
        raise ValueError("Inference CSV must contain 'label' and 'score' columns")

    selected_frames = []
    for label, label_frame in frame.groupby('label', sort=False):
        if mode == 'max':
            selected_frames.append(
                label_frame.sort_values('score', ascending=False).head(limit)
            )
            continue

        sampled_indices, _, _, _ = sample_for_verification.sample_for_validation(
            label_frame['score'].values,
            pos_target_y=0.05,
            neg_target_y=0.05,
            sample_size=limit,
        )
        selected_frames.append(label_frame.iloc[sampled_indices])

    subset_frame = pd.concat(selected_frames).reset_index(drop=True)
    subset_frame.to_csv(subset_csv, index=False)
    log.info(
        'Created %s inference subset of %d rows (%d per label) from %s',
        mode,
        len(subset_frame),
        limit,
        inference_csv,
    )
    return len(subset_frame)

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
