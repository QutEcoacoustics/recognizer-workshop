import copy
from pathlib import Path
import logging
import numpy as np
import dataclasses
from typing import Any, Dict, List, Optional, Callable
import re
from collections import Counter
from recognizer_workshop.app.agile_modeling_helpers import list_audio_files
from perch_hoplite import audio_io
from perch_hoplite.db import interface as hoplite_interface

from baw_helpers.baw_utils import extract_arid_and_domain
from recognizer_workshop.app.services.db_labels.dataclasses import LabelCounts
from baw_helpers.baw_helpers import get_events_for_sites, get_verifications_for_sites
from recognizer_workshop.app.agile_modeling_helpers import get_embeddings_by_source
from recognizer_workshop.app.create_database.create_database import add_parquet_to_db
from recognizer_workshop.app.services.baw_insertions import record_embeddings_added

import random

log = logging.getLogger(__name__)




def add_labeled_audio_examples(
        db,
        source_folder, 
        label, 
        embedding_model,
        dataset_name, 
        sample_rate_hz = 32000,
        annotator_id = ""):
    """
    Adds labeled examples to the database from a folder containing audio files
    """

    window_size_s = 5.0

    labeled_examples_folder = Path(source_folder)
    labeled_examples = list_audio_files(labeled_examples_folder)

    for example_source in labeled_examples:

        log.info(f'Adding {label} example to db:{dataset_name}: {example_source}')

        audio = audio_io.load_audio(example_source, sample_rate_hz)
        # choose the middle 5s of the audio
        start = int(max(len(audio)//2 - sample_rate_hz * window_size_s // 2, 0))
        end = int(start + sample_rate_hz * window_size_s) # python handles index past the end
        audio = audio[start: end]
    
        embedding = embedding_model.embed(audio).embeddings[0, 0]
        add_embedding_to_db(db, dataset_name, embedding, example_source, label, annotator_id=annotator_id)
        log.info('done')


# previously: add_query_to_db
def add_embedding_to_db(db, dataset_name, embedding, source, label, annotator_id = ""):
    
    # TOOD: do we also need to add the config, so that it can construct a full path from config
    # base path and file id?
    source = str(source)
    source = hoplite_interface.EmbeddingSource(dataset_name=dataset_name, source_id=source, offsets=np.array(0.0))
    embedding_id = db.insert_embedding(embedding, source)
    label = hoplite_interface.Label(embedding_id, label, 
                                    type = hoplite_interface.LabelType.POSITIVE, 
                                    provenance=annotator_id)
    db.insert_label(label, skip_duplicates=True)
    db.commit()



def build_parquet_source_list(items):
    """
    Transforms the items dict into the list format needed for parquet download.
    """
    parquet_source_list = []
    for source_url, labels_data in items.items():
        needed_offsets = set()
        for _label_str, type_data in labels_data.items():
            for _label_type_str, offsets in type_data.items():
                needed_offsets.update(offsets)
        
        if needed_offsets:
            parquet_source_list.append({
                'source_url': source_url,
                'offsets': list(needed_offsets)
            })
    return parquet_source_list



def ensure_embeddings_exist(db, config, items, api_client, filter_dict=None, progress_callback: Callable[[dict], None] | None = None):
    """
    Scans the items dict to find all required offsets for each source.
    Calls add_parquet_to_db once to fetch any missing data in parallel.
    """
    if not api_client:
        return

    parquet_source_list = build_parquet_source_list(items)

    if not parquet_source_list:
        return

    log.info(f"Ensuring embeddings exist for {len(parquet_source_list)} sources...")
    
    add_parquet_to_db(
        db=db,
        parquet_source=parquet_source_list,
        api_client=api_client,
        dataset_name='fetched_labels',
        #TODO: figure out where to read this from
        hop_size_s=5.0,
        filter_dict=filter_dict,
        progress_callback=progress_callback,
    )




def add_labels_to_db(db, config, items, api_client=None, filter_dict=None, progress_callback: Callable[[dict], None] | None = None):
    """
    Add labels to the database.
    items is in the format:
    {
        "source_id": {
            "label_str": {
                "pos": [offsets],
                "neg": [offsets]
    }
    Returns:
        A results dictionary in the same format as items, but with the offsets lists replaced by lists of dicts with offset and outcome.
        The outcome can be one of:
        - 'added': The label was successfully inserted.
        - 'already_exists': The exact same label (including type) was already present.
        - 'embedding_not_found': No embedding was found for the source/offset.
        - 'replaced_type': A label with the same name but different type already exists, and the new type replaced it
        - 'error': An unexpected database error occurred.
    """

    # results (of adding) will be in the same format as the input items,
    # but with the offsets lists replaced by lists of dicts with offset and outcome
    results = copy.deepcopy(items)

    total_sources = len(items)
    if progress_callback:
        progress_callback({
            'stage': 'start',
            'total_sources': total_sources,
            'message': f'Preparing to add labels for {total_sources} sources...'
        })

    ensure_embeddings_exist(
        db,
        config,
        items,
        api_client=api_client,
        filter_dict=filter_dict,
        progress_callback=progress_callback,
    )

    # Persist the filter used for this add as a baseline (count 0) under the
    # 'fetched_labels' dataset, so the labeled-examples editor can re-seed from it
    # next time even when no new embeddings were inserted. If embeddings were added,
    # ensure_embeddings_exist has already recorded the real count for this same
    # filter, and adding 0 here is a no-op bump.
    if filter_dict is not None:
        record_embeddings_added(db, 'fetched_labels', filter_dict, 0)

    for source_index, (source_id, labels_data) in enumerate(items.items(), start=1):

        if progress_callback:
            progress_callback({
                'stage': 'source_start',
                'source_id': source_id,
                'source_index': source_index,
                'total_sources': total_sources,
                'message': f'Processing source {source_index}/{total_sources}...'
            })

        try:
            # Build offset->embedding_id from both the main search set and the
            # fetched_labels dataset (embeddings may have been downloaded into
            # either, depending on whether they existed before this run).
            offset_to_embedding_id = {}
            for dataset_name in (config.search_dataset_name, 'fetched_labels'):
                try:
                    source_embeddings = db.get_embeddings_by_source(
                        dataset_name=dataset_name,
                        source_id=source_id
                    )
                    for eid in source_embeddings:
                        offset = db.get_embedding_source(eid).offsets[0]
                        # search_dataset_name takes precedence if present in both
                        offset_to_embedding_id.setdefault(offset, eid)
                except Exception:
                    pass
        except Exception as e:
            log.error(f"Could not retrieve embeddings for source {source_id}: {e}")
            offset_to_embedding_id = {} # Treat as if no embeddings were found

        # --- Loop through the labels and offsets for this source ---
        for label_str, type_data in labels_data.items():
            for label_type_str, offsets in type_data.items():
                
                outcome_list = []
                label_type = hoplite_interface.LabelType.POSITIVE if label_type_str == 'pos' else hoplite_interface.LabelType.NEGATIVE

                for offset in offsets:
                    outcome = "error" # Default outcome

                    embedding_id = offset_to_embedding_id.get(offset)
                    if embedding_id is None:
                        outcome = "embedding_not_found"
                    else:
                        try:
                            # check for duplicates. If the exact same label already exists, we skip it
                            new_label = hoplite_interface.Label(embedding_id, label_str, label_type, config.annotator_id)
                            existing_labels = db.get_labels(embedding_id)

                            if new_label in existing_labels:
                                outcome = "already_exists"
                            else:
                                # check if the label exists with a different type. 
                                # if so, we replace it
                                conflicting_label = next((l for l in existing_labels if l.label == new_label.label and l.type != new_label.type), None)
                                if conflicting_label:
                                    # --- REPLACE THE EXISTING LABEL ---
                                    db.update_label_type(embedding_id, label_str, new_label.type)
                                    outcome = "replaced_type"
                                else:
                                    # --- ADD THE NEW LABEL ---
                                    db.insert_label(new_label)
                                    outcome = "added"

                        except Exception as e:
                            log.error(f"Failed to insert label for embedding {embedding_id}: {e}")
                            outcome = "error"

                    outcome_list.append({'offset': offset, 'outcome': outcome})
                

                results[source_id][label_str][label_type_str] = outcome_list

        if progress_callback:
            progress_callback({
                'stage': 'source_complete',
                'source_id': source_id,
                'source_index': source_index,
                'total_sources': total_sources,
                'message': f'Processed source {source_index}/{total_sources}.'
            })
    

    db.commit()

    if progress_callback:
        progress_callback({
            'stage': 'complete',
            'total_sources': total_sources,
            'message': f'Finished adding labels for {total_sources} sources.'
        })

    return results



def cap_fetched_label_examples(result_sets: list[dict] | None, max_examples: int) -> list[dict] | None:
    """Randomly cap fetched examples per label/type across all result sets."""

    if result_sets is None:
        return None

    if max_examples < 1:
        raise ValueError("max_examples must be at least 1.")

    candidates: dict[tuple[str, str], set[tuple[str, object]]] = {}
    for result_set in result_sets:
        for source_id, labels_data in result_set.get("labels", {}).items():
            for label_name, type_data in labels_data.items():
                for label_type in ("pos", "neg"):
                    key = (str(label_name), label_type)
                    examples = candidates.setdefault(key, set())
                    examples.update((str(source_id), offset) for offset in type_data.get(label_type, []))

    selected: dict[tuple[str, str], set[tuple[str, object]]] = {}
    for key, examples in candidates.items():
        examples_list = list(examples)
        selected[key] = set(
            random.sample(examples_list, min(max_examples, len(examples_list)))
        )

    capped_result_sets = copy.deepcopy(result_sets)
    for result_set in capped_result_sets:
        for source_id, labels_data in result_set.get("labels", {}).items():
            for label_name, type_data in labels_data.items():
                for label_type in ("pos", "neg"):
                    key = (str(label_name), label_type)
                    allowed = selected.get(key, set())
                    type_data[label_type] = [
                        offset for offset in type_data.get(label_type, [])
                        if (str(source_id), offset) in allowed
                    ]

    return capped_result_sets



# def add_labels_to_db_old(db, config, items):
#     """
#     Add labels to the database.
#     items is in the format:
#     {
#         "source_id": {
#             "label_str": {
#                 "pos": [offsets],
#                 "neg": [offsets]
#     }
#     """

#     # from recognizer_workshop.debug_missing_2 import debug_missing

#     embedding_count = 0
#     source_count = 0

#     inserted = {"pos": 0, "neg": 0}
#     duplicates = {"pos": 0, "neg": 0}
#     touched_sources = set()


#     for source_id, labels_data in items.items():

#         # get all offsets mentioned for this source. We need to add any missing embeddings for this source to the db
#         required_offsets = set()
#         for label_str, type_data in labels_data.items():
#                     for label_type_str, offsets in type_data.items():
#                         required_offsets.update(offsets)


#         for label_str in items[source]:
#             for label_type_str in ['pos', 'neg']:
#                 # the source has this label at these offsets
#                 offsets = np.array(items[source][label_str][label_type_str])

#                 # gets the embedding indexes for the source and offsets (we are doing 2 queries that each get all offsets for the source then filter. We could maybe 
#                 # refactor to 1  query then do pos and neg at the same time, but this is easier to read and understand)
#                 embeddings = get_embeddings_by_source(db, dataset_name = config.search_dataset_name,  
#                                                             source_id = source,  
#                                                             offsets = offsets)
#                 if len(embeddings) != len(offsets):
#                     # if there are zero embeeddings returned, it means that the source is not in the database or the offsets are not in the database
#                     # which might happen due to floating point errors (this shoudld have been fixed). 
#                     log.warning(f"Warning: {len(embeddings)} embeddings found for {len(offsets)} offsets for source {source} and label {label_str} ({label_type_str}).")
          
#                 label_type = hoplite_interface.LabelType.POSITIVE if label_type_str == 'pos' else hoplite_interface.LabelType.NEGATIVE

#                 for embedding_id in embeddings:

#                     label = hoplite_interface.Label(embedding_id, label_str, type = label_type, provenance=config.annotator_id)
#                     db.insert_label(label, skip_duplicates=True)
#                     db.commit()
                    
#                 if len(embeddings) > 0:
#                     source_count += 1
#                     embedding_count += len(embeddings)
                    
#     # sources on ecosounds might not have been added to the database.      
#     log.info(f"Added {embedding_count} embeddings from {source_count} of {len(items.keys())} sources to the database.")

