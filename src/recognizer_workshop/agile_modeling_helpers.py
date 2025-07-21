import dataclasses
import numpy as np
import pandas as pd
from pathlib import Path

from perch_hoplite.db import interface as hoplite_interface

from . import baw_api as baw_api
from . import baw_helpers
from . import sample_for_verification
from collections import Counter


# name of the baw provenance to attach to detection events 
PROVENANCE_NAME = "perch-detection"


@dataclasses.dataclass
class SearchResultEvent:
  arid: int
  offset: float
  label: str
  score: float
  duration: float = 5



# @dataclasses.dataclass
# class SearchResultSet:
#   events: list[SearchResultEvent]
#   provenance



def list_audio_files(path, recursive=True, quiet=False, extensions=(".wav", ".flac", ".mp3")):
    """
    Prints a numbered list of audio files in a folder (and optionally subfolders)
    And returns a the list
    """
    path = Path(path)
    files = path.rglob("*") if recursive else path.glob("*")
    audio_files = [
        f for f in files 
        if f.is_file()
        and f.suffix.lower() in extensions
        and not f.name.startswith(".")
    ]

    if not quiet:
        print(f"Found {len(audio_files)} audio files in {str(path)}")
        print("\n".join([f"{i}: {f}" for i, f in enumerate(audio_files)]))
    return audio_files



def upload_events_for_verification(events):
  """
  Use the event upload endpoint to upload a json file as a multipart mime type.
  Add provenance information
  """
  pass



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

    events = get_events_with_verifications_and_tags(api, filter_params, target_labels)



    # for each event, determine if it is trusted and its type (positive or negative)

    for event in events:
        
        if event['provenance_id'] is None:
            # no provenance: probably not machine generated (TODO: revisit this assumption)
            for tagging in event.get('taggings', []):
                tagging['type'] = 'positive'

        else:
            # if there is a provenance, probably it is auto generated and needs verification
            for tagging in event.get('taggings', []):
                tagging['type'] = get_consensus(tagging['verifications'])

    # for each event, determine the 5-second segments that each event covers
    for event in events:
        event['segments'] = get_event_segments(event)

    # reshape into a format so we can determine negative labels in a multi-class setting

    items = {}
    for event in events:
        arid = event['audio_recording_id']
        source = f"{api.base_url}/audio_recordings/{arid}/original"
        if source not in items:
            items[source] = {}
        
        for segment in event['segments']:
            if segment['start'] not in items[source]:
                items[source][segment['start']] = {}

            for label_str in target_labels:
                if label_str not in items[source][segment['start']]:
                    items[source][segment['start']][label_str] = None

                for tagging in event.get('taggings', []):
                    if items[source][segment['start']][label_str] == 'pos':
                        # this label for this source:offset is already positive, so we don't need to do anything
                        continue

                    if tagging['tag']['text'] == label_str and tagging['type'] == 'positive':
                        items[source][segment['start']][label_str] = 'pos'
                    elif tagging['tag']['text'] == label_str and tagging['type'] == 'negative':
                        items[source][segment['start']][label_str] = 'neg'
                    elif tagging['tag']['text'] != label_str and tagging['type'] == 'positive':
                        items[source][segment['start']][label_str] = 'neg'
                    
                    # Note: unverified taggings which require verification will not have a tagging['type'], so are ignored
                    # Note: we don't add a positive label for the true class where a tagging has been verified as FP (negative)
                    #       because we don't have that information. 
    
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


    return items_for_insert


def add_labels_to_db(agile, config, items):
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


    # debug_missing(
    #     db_path='/Users/eichinsp/Documents/github/peichins/recognizer-workshop/local_scripts/test_folder/db/116',
    #     db=agile.db,
    #     region_id=116,
    #     source_ids=list(items.keys()))

    for source in items:
        for label_str in items[source]:
            for label_type_str in ['pos', 'neg']:
                # the source has this label at these offsets
                offsets = np.array(items[source][label_str][label_type_str])

                # gets the embedding indexes for the source and offsets (we are doing 2 queries that each get all offsets for the source then filter. We could maybe 
                # refactor to 1  query then do pos and neg at the same time, but this is easier to read and understand)
                embeddings = get_embeddings_by_source(agile, dataset_name = config.search_dataset_name,  
                                                            source_id = source,  
                                                            offsets = offsets)

          
                label_type = hoplite_interface.LabelType.POSITIVE if label_type_str == 'pos' else hoplite_interface.LabelType.NEGATIVE

                for embedding_id in embeddings:

                    label = hoplite_interface.Label(embedding_id, label_str, type = label_type, provenance=config.annotator_id)
                    agile.db.insert_label(label, skip_duplicates=True)
                    agile.db.commit()
                    
                if len(embeddings) > 0:
                    source_count += 1
                    embedding_count += len(embeddings)
                    
    # sources on ecosounds might not have been added to the database.      
    print(f"Added {embedding_count} embeddings from {source_count} of {len(items.keys())} sources to the database.")


round_to_nearest_5 = lambda x: np.round(x / 5) * 5

# taken from the hoplite interface with a few modifications
def get_embeddings_by_source(
      agile,
      dataset_name: str,
      source_id: str | None,
      offsets: np.ndarray | None = None,
  ) -> np.ndarray:

    cursor = agile.db.db.cursor()

    query = (
        'SELECT hoplite_embeddings.id, offsets FROM hoplite_embeddings '
        'JOIN hoplite_sources ON hoplite_embeddings.source_idx = hoplite_sources.id '
        'WHERE hoplite_sources.source = ?;'
    )
    
    # debug query, just get anything
    # query2 = (
    #     'SELECT hoplite_embeddings.id, offsets FROM hoplite_embeddings '
    #     'JOIN hoplite_sources ON hoplite_embeddings.source_idx = hoplite_sources.id '
    #     'LIMIT 20;'
    # )

    cursor.execute(query, (source_id,))
    #cursor.execute(query2)

    result_pairs = cursor.fetchall()

    metadata = agile.db.get_metadata('legacy_config')
    #embedding_dtype = metadata['embed_fn_config']['tensor_dtype']
    offset_dtype = np.float32

    # HACK! there was a bug where offsets greater than 2049 came with floating point precision errors.
    # we need to convert them to multiples of 5, since we know this to be the case for the FSP data.
    # deserialize and round the offsets in the result pairs
    result_pairs_rounded = [(embedding_id, round_to_nearest_5(deserialize_embedding(offset_bytes, offset_dtype)[0])) for embedding_id, offset_bytes in result_pairs]


    matching_embedding_ids = [embedding_id for embedding_id, offset_rounded 
                              in result_pairs_rounded 
                              if offset_rounded in offsets]

    return np.array(matching_embedding_ids)


# taken from the hoplie interface
# why can't we just import this?
def deserialize_embedding(
    serialized_embedding: bytes, embedding_dtype
) -> np.ndarray:
  return np.frombuffer(
      serialized_embedding,
      dtype=np.dtype(embedding_dtype).newbyteorder('<'),
  )



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






def get_events_with_verifications_and_tags(api, filter_params, target_labels=None):
    """
    For the target label, get all events that have that tag.
    Then get all verifications for those events, and attach them to the taggings. 
    """

    if target_labels is None:
        target_labels = []
    elif isinstance(target_labels, str):
        target_labels = [target_labels]

    # these are non-taxonomic tags that have been used to describe something
    # which should be included as a negative example for the targets
    special_tags = ["bird", "frog", "insect", "mammal", "fish", "reptile", "amphibian"]


    events = baw_helpers.get_events_for_sites(api, filter_params)
    print(f"Found {len(events)} events for: {filter_params}")
    # remove events that have no taggings
    # this is only applicable if target_label is None
    events_without_taggings = [event for event in events if 'taggings' not in event or len(event['taggings']) == 0]
    events = [event for event in events if 'taggings' in event and len(event['taggings']) > 0]
    event_ids = [event['id'] for event in events]
    verifications = baw_helpers.get_verifications_for_sites(api, event_ids)
    
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
                            if tagging['tag']['text'] in target_labels + special_tags or tagging['tag'].get('is_taxonomic', False)]
        
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



def fix_source_format(db):
    """
    Fix the source format in the hoplite_sources table.
    This is a workaround for the issue where sources end with a slash (e.g. 'https://api.ecosounds.org/audio_recordings/123/original/').
    This will remove the trailing slash from the source.
    """

    sql = """
            UPDATE hoplite_sources
            SET source = RTRIM(source, '/')
            WHERE source LIKE '%/';
            """
    cursor = db.db.cursor()
    cursor.execute(sql)
    db.db.commit()



def get_or_create_workshop_provenance(api):
    """
    Create a provenance object for the workshop.
    This will be used to track the source of the events.
    """



    # TODO: some way to set this for different workshop. Maybe env variable?
    # or probably some sort of workshop-config file

    # example:
    # {
    #   "id": 1,
    #   "name": "Perch",
    #   "version": "4",
    #   "url": "https://github.com/QutEcoacoustics/perch-runner",
    #   "description": "Extract feature embeddings",
    #   "score_minimum": null,
    #   "score_maximum": null,
    #   "creator_id": 1,
    #   "created_at": "2025-05-30T04:50:09.726Z",
    #   "updater_id": null,
    #   "updated_at": "2025-05-30T04:50:09.726Z",
    #   "deleter_id": null,
    #   "deleted_at": null,
    #   "description_html_tagline": "Extract feature embeddings",
    #   "description_html": "<p>Extract feature embeddings</p>\n"
    # }

    #all_provenaces = api.get("/provenances")

    # first check if the provenance already exists
    response = api.baw_filter("/provenances", {"name": {"eq": PROVENANCE_NAME}})
    existing_provenances = response.get("data", [])
    baw_api.baw_api.print_response(existing_provenances)

    if len(existing_provenances) > 0:
        print(f"Provenance {PROVENANCE_NAME} already exists. Using existing provenance.")
        return existing_provenances[0]
    else:
        print(f"Creating new provenance {PROVENANCE_NAME}.")

        provenance_data = {
            "name": PROVENANCE_NAME,
            "description": "Generated by classification of a perch-based recognizer",
            "url": "https://notebook-01.openecoacoustics.org",
            "version": 0.1
        }

        # create the provenance
        response = api.post("/provenances", data=provenance_data)
        baw_api.baw_api.print_response(response)

    return response.get("data", {})

    
def make_inference_subset(inference_csv, subset_csv, limit=100):
    """
    Creates a random subset of a CSV file, preserving the original relative order of the selected rows.

    Args:
        inference_csv (str): The path to the input CSV file.
        subset_csv (str): The path where the output subset CSV file will be saved.
        limit (int): The number of random rows to include in the subset.
    """

    df = pd.read_csv(inference_csv)

    # for each unique label in the 'label' column:
    all_indices = []
    labels = df['label'].unique()
    for label in labels:

        # subset of the dataframe with that label
        label_df = df[df['label'] == label]
        sampled_indices, _, _, _ = sample_for_verification.sample_for_validation(
            label_df['logits'].values,
            pos_target_y=0.05,
            neg_target_y=0.05,
            sample_size=limit
        )

        # add the sampled indices to the all_indices list
        all_indices.extend(label_df.index[sampled_indices].tolist())
        

    subset_df = df.loc[all_indices].reset_index(drop=True)
    subset_df.to_csv(subset_csv, index=False)

    print(f"Successfully created a random subset of {len(subset_df)} rows ({limit} for each of {len(labels)} labels) from '{inference_csv}'.")
    print(f"Output saved to '{subset_csv}'")

