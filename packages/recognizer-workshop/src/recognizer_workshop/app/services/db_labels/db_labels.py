from pathlib import Path
import numpy as np
import dataclasses
from typing import Any, Dict, List, Optional
import re
from collections import Counter
from recognizer_workshop.app.agile_modeling_helpers import list_audio_files
from perch_hoplite import audio_io
from perch_hoplite.db import interface as hoplite_interface

from baw_helpers.baw_utils import extract_arid_and_domain
from recognizer_workshop.app.services.db_labels.dataclasses import LabelCounts, DatabaseSummary
from baw_helpers.baw_helpers import get_events_for_sites, get_verifications_for_sites
from recognizer_workshop.app.agile_modeling_helpers import get_embeddings_by_source


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

        print(f'Adding {label} example to db:{dataset_name}: {example_source}')

        audio = audio_io.load_audio(example_source, sample_rate_hz)
        # choose the middle 5s of the audio
        start = int(max(len(audio)//2 - sample_rate_hz * window_size_s // 2, 0))
        end = int(start + sample_rate_hz * window_size_s) # python handles index past the end
        audio = audio[start: end]
    
        embedding = embedding_model.embed(audio).embeddings[0, 0]
        add_embedding_to_db(db, dataset_name, embedding, example_source, label, annotator_id=annotator_id)
        print('done')


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


def get_label_counts(db) -> Dict[str, LabelCounts]:
    """
    Retrieves the number of positive and negative labels from the database.    
    Returns:
        A dictionary mapping label names to LabelCounts objects.
    """
    positive_counts = db.get_class_counts(hoplite_interface.LabelType.POSITIVE)
    negative_counts = db.get_class_counts(hoplite_interface.LabelType.NEGATIVE)
    
    all_counts: Dict[str, LabelCounts] = {}
    
    for lbl, count in positive_counts.items():
        all_counts[lbl] = LabelCounts(
            label_name=lbl,
            positive=count,
            negative=negative_counts.get(lbl, 0) # Use .get for safety
        )
    return all_counts


def get_db_summary(db) -> DatabaseSummary:
    """
    Returns:
        A DatabaseSummary instance containing the state of the DB.
    """
    total_embeddings = db.count_embeddings()
    dataset_names = db.get_dataset_names()
    label_counts_data = get_label_counts(db)
    
    return DatabaseSummary(
        total_embeddings=total_embeddings,
        datasets=dataset_names,
        label_counts=label_counts_data
    )


def get_db_arids(db) -> set:
    #TODO: this might be general enough to move to agile modeling helpers

    all_sources = db.get_all_sources()
    arids = set([extract_arid_and_domain(source) for source in all_sources])
    return arids




def add_labels_to_db_old(db, config, items):
    """
    Add labels to the database.
    items is in the format:
    {
        "source_id": {
            "label_str": {
                "pos": [offsets],
                "neg": [offsets]
    }
    """

    # from recognizer_workshop.debug_missing_2 import debug_missing

    embedding_count = 0
    source_count = 0

    inserted = {"pos": 0, "neg": 0}
    duplicates = {"pos": 0, "neg": 0}
    touched_sources = set()


    for source in items:
        for label_str in items[source]:
            for label_type_str in ['pos', 'neg']:
                # the source has this label at these offsets
                offsets = np.array(items[source][label_str][label_type_str])

                # gets the embedding indexes for the source and offsets (we are doing 2 queries that each get all offsets for the source then filter. We could maybe 
                # refactor to 1  query then do pos and neg at the same time, but this is easier to read and understand)
                embeddings = get_embeddings_by_source(db, dataset_name = config.search_dataset_name,  
                                                            source_id = source,  
                                                            offsets = offsets)
                if len(embeddings) != len(offsets):
                    # if there are zero embeeddings returned, it means that the source is not in the database or the offsets are not in the database
                    # which might happen due to floating point errors (this shoudld have been fixed). 
                    print(f"Warning: {len(embeddings)} embeddings found for {len(offsets)} offsets for source {source} and label {label_str} ({label_type_str}).")
          
                label_type = hoplite_interface.LabelType.POSITIVE if label_type_str == 'pos' else hoplite_interface.LabelType.NEGATIVE

                for embedding_id in embeddings:

                    label = hoplite_interface.Label(embedding_id, label_str, type = label_type, provenance=config.annotator_id)
                    db.insert_label(label, skip_duplicates=True)
                    db.commit()
                    
                if len(embeddings) > 0:
                    source_count += 1
                    embedding_count += len(embeddings)
                    
    # sources on ecosounds might not have been added to the database.      
    print(f"Added {embedding_count} embeddings from {source_count} of {len(items.keys())} sources to the database.")


import copy
import logging
log = logging.getLogger(__name__)

def add_labels_to_db(db, config, items):
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
    

    for source_id, labels_data in items.items():

        # Query the database ONCE per source to get all its embeddings and offsets for this source
        try:
            source_embeddings = db.get_embeddings_by_source(
                dataset_name=config.search_dataset_name,
                source_id=source_id
            )
            # Create a fast lookup map of {offset: embedding_id}
            offset_to_embedding_id = {
                db.get_embedding_source(eid).offsets[0]: eid for eid in source_embeddings
            }
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
    

    db.commit()
    return results






def labels_from_event_taggings(api, filter_params, target_labels):
    """
    For the given audio recording filter params
    - gets all events for the target label 
    - determines if verification is requires for each event, based on provenance
    - for any that do require verifications, get verifications
    - determine the 5-second segments that each event covers
    - insert them into the hoplite database
    - gets events that are tagged with the negative_tags_for_target tags, inserts them
    into the hoplite database as negative label for the target label.
    """

    # convert our target label objects to baw tags:
    tags = [l.get_baw_tags()[0] for l in target_labels]
    
    # all events, not just those with the target tags - we need these to determine negative labels
    events = get_events_with_verifications_and_tags(api, filter_params, tags)

    # for each event, determine if it is trusted and its type (positive or negative)

    for event in events:

        for tagging in event.get('taggings', []):
            if len(tagging['verifications']) == 0 and event['provenance_id'] is None:
                # if there is no provenance, it should be likely that it was manually created and therefore treat as verified
                # but if there are verifications, we should check them to know for sure
                tagging['type'] = 'positive'
            else:
                tagging['type'] = get_consensus(tagging['verifications'])
   

    # for each event, determine the 5-second segments that each event covers
    for event in events:
        event['segments'] = get_event_segments(event)

    # reshape into a format so we can determine negative labels in a multi-class setting

    items = {}
    counts = {}
    for event in events:
        arid = event['audio_recording_id']
        source = f"{api.base_url}/audio_recordings/{arid}/original"
        if source not in items:
            items[source] = {}
        
        for segment in event['segments']:
            if segment['start'] not in items[source]:
                items[source][segment['start']] = {}

            for label in target_labels:
                label_str = str(label) # the string representation of the label, e.g. "magpie (song)"
                tag_str = label.get_baw_tags()[0] # the baw tag, e.g. "Magpie song"
                if label_str not in items[source][segment['start']]:
                    items[source][segment['start']][label_str] = None
                if label_str not in counts:
                    counts[label_str] = {'positive': 0, 'negative': 0}

                for tagging in event.get('taggings', []):
                    if items[source][segment['start']][label_str] == 'pos':
                        # this label for this source:offset is already positive from a previous label or tagging, 
                        # so we don't need to do anything
                        continue

                    cur_label_matches_tag = (tagging['tag']['text'] == tag_str)

                    if cur_label_matches_tag and tagging['type'] == 'positive':
                        items[source][segment['start']][label_str] = 'pos'
                        counts[label_str]['positive'] += 1
                    elif cur_label_matches_tag and tagging['type'] == 'negative':
                        items[source][segment['start']][label_str] = 'neg'
                        counts[label_str]['negative'] += 1
                    elif not cur_label_matches_tag and tagging['type'] == 'positive':
                        items[source][segment['start']][label_str] = 'neg'
                        counts[label_str]['negative'] += 1
                    
                    # Note: unverified taggings which require verification will not have a tagging['type'], so are ignored
                    # Note: we don't add a positive label for the true class where a tagging has been verified as FP (negative)
                    #       because we don't have that information (and we probably don't care)
    
    print(f'counts: {counts}')

    # reshape into the format that is best for inserting into the hoplite database
    items_for_insert = {}
    for source in items:
        for offset in items[source]:
            for label_str in items[source][offset]:
                if items[source][offset][label_str] is not None:
                    if source not in items_for_insert:
                        items_for_insert[source] = {}
                    if label_str not in items_for_insert[source]:
                        items_for_insert[source][label_str] = {'pos': [], 'neg': []}
                    
                    if items[source][offset][label_str] == 'pos':
                        items_for_insert[source][label_str]['pos'].append(offset)
                    elif items[source][offset][label_str] == 'neg':
                        items_for_insert[source][label_str]['neg'].append(offset)


    return items_for_insert, counts



def get_events_with_verifications_and_tags(api, filter_params, target_tags=None):
    """
    Get all events for the given filter params (site, project, region etc). 
    We don't filter by tags here, because we want to get non-target events too, to use as negative examples.
    Then get all verifications for those events, and attach them to the taggings. 
    """


    if target_tags is None:
        target_tags = []
    elif isinstance(target_tags, str):
        target_tags = [target_tags]

    # these are non-taxonomic tags that have been used to describe something
    # which should be included as a negative example for the targets
    special_tags = ["bird", "frog", "insect", "mammal", "fish", "reptile", "amphibian"]


    events = get_events_for_sites(api, filter_params)
    print(f"Found {len(events)} events for: {filter_params}")
    # remove events that have no taggings
    # this is only applicable if target_label is None
    events_without_taggings = [event for event in events if 'taggings' not in event or len(event['taggings']) == 0]
    events = [event for event in events if 'taggings' in event and len(event['taggings']) > 0]
    event_ids = [event['id'] for event in events]
    verifications = get_verifications_for_sites(api, event_ids)
    
    tag_ids = list(set([tagging['tag_id'] for event in events for tagging in event.get('taggings', [])]))

    # all tags that are in the events. We need this later to know which are taxonomic tags and which are not.
    tags = api.post("/tags/filter?disable_paging=true", data={
        "filter": {
            "id": {
                "in": tag_ids
            }
        }
    }).get('data', [])

    if len(tags) != len(tag_ids):
        print(f"Warning: {len(tags)} tags found for {len(tag_ids)} tag ids. Some tags may not exist in the database.")
        tags_with_invalid_ids = [tag for tag in tags if tag['id'] not in tag_ids]
        print(f"Invalid tags: {tags_with_invalid_ids}")

    # convert tags to a dictionary for easy lookup
    tags_dict = {tag['id']: tag for tag in tags}

    # for each event, match the verification

    for event in events:
        event_id = event['id']
        verifications_for_event = [v for v in verifications if v['audio_event_id'] == event_id]

        # for each tagging on the event, check if it has a verification, and apply the verification to the tagging. 
        for tagging in event.get('taggings', []):
            tagging['tag'] = tags_dict.get(tagging['tag_id'], None)
            tag_id = tagging['tag_id']
            verifications_for_tag_and_event = [v for v in verifications_for_event if v['tag_id'] == tag_id]

            tagging['verifications'] = verifications_for_tag_and_event

        # remove non-taxonomic taggings except for the target labels
        event['taggings'] = [tagging 
                            for tagging in event.get('taggings', []) 
                            if tagging['tag']['text'] in target_tags + special_tags or tagging['tag'].get('is_taxonomic', False)]
        
        print(f"Event {event_id} has {len(event.get('taggings', []))} taggings after filtering.")

    # remove events that have no taggings
    events_without_taxonomic_taggings = [event for event in events if len(event['taggings']) == 0]
    events = [event for event in events if len(event['taggings']) > 0]

    print(f"Found {len(events)} events with taxonomic taggings after filtering.")
    print(f"Found {len(events_without_taxonomic_taggings)} events with only non-taxonomic taggings.")
    print(f"Found {len(events_without_taggings)} events without taggings.")
        
    # we now have 
    # - 1 or more taxonomig taggings for each event
    # - 0 or more verifications for each tagging on each event, 
    # - 1 tag record for each tagging on each event.
    return events





def get_event_segments(event: dict, segment_duration: int = 5) -> list[dict]:
    """
    Analyzes a single event and breaks it down into non-overlapping segments.

    For each 5-second segment the event touches, it calculates the proportion
    of the segment covered by the event and the proportion of the event
    that lies within the segment.

    Args:
        event (dict): A dictionary with "start_time_seconds" and "end_time_seconds".
        segment_duration (int): The duration of each segment in seconds.

    Returns:
        list[dict]: A list of dictionaries, each representing a covered segment.
    """
    event_start = event["start_time_seconds"]
    event_end = event["end_time_seconds"]
    total_event_duration = event_end - event_start

    # Handle zero-duration events to avoid division by zero.
    if total_event_duration <= 0:
        return []

    results = []

    # Calculate the start of the first 5-second segment this event touches.
    # e.g., if event_start is 14, (14 // 5) * 5 = 10.
    current_segment_start = (event_start // segment_duration) * segment_duration

    # Loop through 5-second segments as long as they overlap with the event.
    while current_segment_start < event_end:
        current_segment_end = current_segment_start + segment_duration

        # Calculate the actual intersection between the event and the current segment.
        overlap_start = max(event_start, current_segment_start)
        overlap_end = min(event_end, current_segment_end)
        overlap_duration = overlap_end - overlap_start

        # Calculate the required proportions
        segment_cover = overlap_duration / segment_duration
        event_cover = overlap_duration / total_event_duration

        # Store the results for this segment
        results.append({
            "start": current_segment_start,
            "end": current_segment_end,
            "segment_cover": round(segment_cover, 2), # Round for cleaner output
            "event_cover": round(event_cover, 2),
            "cover_score": round(segment_cover + event_cover, 2),

        })

        # Move to the next segment
        current_segment_start += segment_duration
    
    max_cover_score = max(seg['cover_score'] for seg in results)

    # filter out segments that don't have enough coverage
    # the average of amount of the segment covered and the amount of the event inside the segment must be greater than the threshold
    # or if none of segments have that much then keep just the one with the most coverage'
    results = [seg for seg in results if seg['cover_score'] > 0.2 or seg['cover_score'] == max_cover_score]

    return results


def get_consensus(verifications):
    """
    Get the consensus of the verifications.
    If there are no verifications, return None.
    If ther are an equal number of positive and negative verifications, return None.
    Otherwise, return whichever of positive or negative is more common. 
    """
    if not verifications:
        return None

    counts = Counter([v['confirmed'] for v in verifications])
    if counts['correct'] > counts['incorrect']:
        return 'positive'
    elif counts['incorrect'] > counts['correct']:
        return 'negative'
    else:
        return None

