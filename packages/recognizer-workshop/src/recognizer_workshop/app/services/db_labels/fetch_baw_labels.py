import copy
from pathlib import Path
import logging
import numpy as np
import dataclasses
from typing import Any, Dict, List, Optional
import re
from collections import Counter
from recognizer_workshop.app.agile_modeling_helpers import list_audio_files
from perch_hoplite import audio_io
from perch_hoplite.db import interface as hoplite_interface
from collections import UserDict
from recognizer_workshop.app.services.db_labels.dataclasses import LabelName, LabelProposition, PropositionType, EventSegment

from baw_helpers.baw_utils import extract_arid_and_domain

from recognizer_workshop.app.services.db_labels.dataclasses import LabelCounts, DatabaseSummary
from baw_helpers.baw_helpers import get_events_for_sites, get_verifications_for_sites
from recognizer_workshop.app.agile_modeling_helpers import get_embeddings_by_source
from recognizer_workshop.app.create_database.create_database import add_parquet_to_db

log = logging.getLogger(__name__)



def labels_from_event_taggings(api, filter_params, target_labels: List[LabelName], target_negatives: UserDict):
    """
    For the given audio recording filter params
    - gets all events for the target labels
    - TODO: gets events for specified negatives of target labels. (e.g. fairy wren gets confused for grasswren, so use all fairwren as negative examples for grasswren) 
    - includes verification summaries for those events
    - determine the 5-second segments that each event covers
    - consolodate them into sets of labels for each unique segment (overlapping events might have duplicate segments with different or conflicting labels)
    @param api: BAW API client
    @param filter_params: dict of filter params for which sites/projects/regions to get events for
    @param target_labels: list of LabelName objects representing the target labels to get events for
    @param target_negatives: dict of target label string -> list of negative label strings, to get events for those negative labels 
    @return: dict of source -> label -> {pos: [offsets], neg: [offsets]}, and counts of positive and negative labels
    """

    # convert our target label objects to baw tags:
    tags = [l.get_baw_tags()[0] for l in target_labels]

    # flatten negative tags for each target into a single list
    extra_tags = [tag.get_baw_tags()[0] for neg_list in target_negatives.values() if isinstance(neg_list, list) for tag in neg_list]
  
    all_tags = list(set(tags + extra_tags))
    
    # get target events
    events = get_events_with_verifications_and_tags(api, filter_params, all_tags)

    # for each event, determine the positive and negative labels based on
    # verifications, taggings and provenance.
    for event in events:
        
        # for each event, we attach a dict that has the label text and label type, based on verifications, taggings and provenance
        # including for non target labels, so we can use those as negative examples if needed. 
        event['label_info'] = {}

        requires_verification = event['provenance_id'] is not None

        if requires_verification:
            for verification_summary_item in event['verification_summary']:
                tag_id = int(verification_summary_item['tag_id'])
                tag = api.tags.get_by_id(tag_id)
                if tag is None:
                    log.warning(f"Tag id {tag_id} not found in tag cache.")
                    continue
                tag_text = tag['text']

                if verification_summary_item['correct'] > verification_summary_item['incorrect']:
                    consensus = 'positive'
                elif verification_summary_item['incorrect'] > verification_summary_item['correct']:
                    consensus = 'negative'
                else:
                    consensus = None

                if consensus is not None:
                    event['label_info'][tag_text] = consensus
        else:

            for tagging in event.get('taggings', []):
                # if there is no provenance, it should be likely that it was manually created (before provenance was added to the schema, events were almost always drawn on spectrograms)
                # and therefore treat as verified
                # but if there are verifications, we should check them to know for sure
                tag_text = tagging['tag']['text']
                event['label_info'][tag_text] = 'positive'

    # mutates events
    for event in events:
        compile_label_proposition_types(event, target_labels, target_negatives)
                   
    # for each event, determine the 5-second segments that each event covers
    for event in events:
        event['segments'] = get_event_segments(event)

    # reshape into a format so we can determine negative labels 
    items = {}

    for event in events:
        arid = event['audio_recording_id']
        source = f"{api.base_url}/audio_recordings/{arid}/original"
        if source not in items:
            items[source] = {}
        
        for event_segment in event['segments']:
            seg_start = event_segment.segment_start
            if seg_start not in items[source]:
                items[source][seg_start] = {}

            for label in target_labels:
                label_str = str(label) # the hoplite string representation of the label, e.g. "magpie (song)"
                
                # proposition for this target label on this event-segment e.g. None, TYPE_VERIFIED_POS etc ..
                prop_type = event['label_proposition_types'].get(label)

                if label_str not in items[source][seg_start]:
                    items[source][seg_start][label_str] = []

                # if the current target label has a proposition type for this event
                # add a LabelProposition for all the eventSegments of this event, with that label and proposition type
                if prop_type:
                    entry = LabelProposition(
                        label=label,
                        proposition_type=prop_type,
                        event_segment=event_segment
                    )
                
                    # so we now have a label type (e.g. TYPE_VERIFIED_POS) for this target label on this event/segment
                    # with its local start and end within the segment
                    items[source][seg_start][label_str].append(entry)

    return resolve_conflicts_and_format(items)
                    
 


def compile_label_proposition_types(event, target_labels: list[LabelName], target_negatives: UserDict[LabelName, list[LabelName]]):
    """
    For each target label assigns a label proposition type (e.g. VERIFIED_POS, VERIFIED_NEG, IMPLIED_NEG) for this event,
    """

    # there may be overlapping events, such as a "positive for grasswren" event overlapping with a "negative for grasswren" event
    # or, conflicts between target-negatives and actual targets. E.g. "positive for target species grasswren" but "positive for non-target species fairywren, which is a negative example for grasswren"
    # leading to contradictory labels for the same segment of the source. 
    # let's compile a list of label propositions for each target label on this event, then later we can resolve conflicts when we consolodate segments across events.
    # IMPLIED_NEG: negative label based on non-target positive tagging. 
    # VERIFIED_POS: positive label based on target positive tagging or verification
    # VERIFIED_NEG: negative label based on target negative verification

    event['label_proposition_types'] = {}
    for target_label in target_labels:

        target_tag = target_label.get_baw_tags()[0]
        
        if event['label_info'].get(target_tag, None) == 'negative':
            event['label_proposition_types'][target_label] = PropositionType.VERIFIED_NEG
   
        elif event['label_info'].get(target_tag, None) == 'positive':
            event['label_proposition_types'][target_label] = PropositionType.VERIFIED_POS
    
        elif target_label in target_negatives:
            # for each positive tag associated with this event, check if it's in the list of negatives for this target label
            for label_info_text, label_info_type in event['label_info'].items():
                if label_info_type == 'positive' and target_negatives.target_has_item(target_label, label_info_text):
                    event['label_proposition_types'][target_label] = PropositionType.IMPLIED_NEG
                    break
                    
 

def resolve_conflicts_and_format(items):

    # we now have, for each 5sec seg of audio that overlapped with any of the events, for each target label a list of propositions (including empty lists) 

    # reshape into the format that is best for inserting into the hoplite database.
    # also resolve conflicts (multiple prpopositions for the same label at the same offset of the same source) by priority
    # i.e. instead of nesting labels under offsets, we nest offsets under labels.
    items_for_insert = {}
    counts = {}
    for source in items:
        for offset in items[source]:
            for label_str in items[source][offset]:
                
                # a list of pos or neg propositions for this label at this source and offset, and their respective priorities
                cur_item_propositions = items[source][offset][label_str]
                if cur_item_propositions:

                    winning_label_type = resolve_conflicts(cur_item_propositions)

                    # initialize the structure and counts
                    if source not in items_for_insert:
                        items_for_insert[source] = {}
                    if label_str not in items_for_insert[source]:
                        items_for_insert[source][label_str] = {'pos': [], 'neg': []}
                    if label_str not in counts:
                        counts[label_str] = {'positive': 0, 'negative': 0}
                    
                    if winning_label_type == 'positive':
                        items_for_insert[source][label_str]['pos'].append(offset)
                        counts[label_str]['positive'] += 1
                    elif winning_label_type == 'negative':
                        items_for_insert[source][label_str]['neg'].append(offset)
                        counts[label_str]['negative'] += 1

    log.info(f'counts: {counts}')

    return items_for_insert, counts


def should_use_negative_tag_for_target(target_label, target_negatives_tag_text, spec):
    """
    Given a specification of which tags to use as negatives for a given positive tags,
    determine if the target is in the list.  

    """
    if target_label not in spec:
        return False
    return target_negatives_tag_text in spec[target_label]



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

    
    events = get_events_for_sites(api, filter_params, tags=target_tags)

    log.info(f"Found {len(events)} events for: {filter_params}")

    # we have a verification summary on the event, which is a list of verification counts for each tag id.
 
    return events



def get_event_segments(event: dict, segment_duration: int = 5) -> list[dict]:
    """
    Analyzes a single event and breaks it down into non-overlapping fixed-lengthsegments.

    For each 5-second segment the event touches, it calculates the proportion
    of the segment covered by the event and the proportion of the event
    that lies within the segment (these can be used later to decide whether to apply the labels to the segment)

    Args:
        event (dict): A dictionary with "start_time_seconds" and "end_time_seconds".
        segment_duration (int): The duration of each segment in seconds.

    Returns:
        list[dict]: A list of dictionaries, each representing a covered segment.
    """
    event_start = event["start_time_seconds"]
    event_end = event["end_time_seconds"]

    if event_end - event_start <= 0:
        # just in case the event is zero length for whatever reason
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

        cur_event_segment = EventSegment(
            segment_start=current_segment_start,
            segment_end=current_segment_end,
            start=overlap_start,
            end=overlap_end,
            event_start=event_start,
            event_end=event_end
        )

        # Store the results for this segment
        results.append(cur_event_segment)

        # Move to the next segment
        current_segment_start += segment_duration
    
    return results




def calculate_proposition_weights(p: LabelProposition, num_slots: int, slot_duration: float) -> tuple[int, int, np.ndarray]:
    """
    Calculates the weight array for a single proposition, applying gradient logic
    to penalize edges of events based on 'event_cover'.
    
    Returns:
        (start_index, end_index, weights_array)
        - start_index: The index in the global slot array where this prop starts.
        - end_index: The index where it ends (exclusive).
        - weights_array: An array of floats (0.0 to 1.0) for these specific slots.
    """
    MAX_RAMP_SECONDS = 2.0
    RAMP_PERCENTAGE = 0.10

    event_len_sec = p.event_segment.event_duration

    if event_len_sec <= 0:
        # probably unnecessary safety check
        event_len_sec = 5.0

    total_event_slots = int(event_len_sec / slot_duration)

    # 2. Ramp Calculation
    num_ramp_slots = min(MAX_RAMP_SECONDS / slot_duration, int(total_event_slots * RAMP_PERCENTAGE))
    
    # 3. Base Edge Weight
    edge_weight = p.event_segment.event_cover # Assumes upstream value is valid (0.0-1.0)

    # 4. Determine Bounds in Segment
    start_slot = max(0, int(p.event_segment.start_offset_in_segment / slot_duration))
    end_slot = min(num_slots, int(p.event_segment.end_offset_in_segment / slot_duration))
    
    # Pre-calculate offset relative to event start
    prop_start_in_event_slots = int(p.event_segment.start_offset_in_event / slot_duration)
    
    # Create the weights for just this slice
    length = end_slot - start_slot
    if length <= 0:
        return 0, 0, np.array([])
        
    weights = np.zeros(length, dtype=float)
    divisor = num_ramp_slots if num_ramp_slots > 0 else 1.0

    for i in range(length):
        # 'i' is the step inside this specific slice
        current_slot_in_event = prop_start_in_event_slots + i
        
        dist_from_start = current_slot_in_event
        dist_from_end = total_event_slots - current_slot_in_event - 1
        dist_from_edge = min(dist_from_start, dist_from_end)
        
        if dist_from_edge >= num_ramp_slots:
            weights[i] = 1.0
        else:
            progress = dist_from_edge / divisor
            weights[i] = edge_weight + (1.0 - edge_weight) * progress
            
    return start_slot, end_slot, weights

def resolve_conflicts(propositions: List[LabelProposition]):
    """
    Resolves labeling conflicts (Positive vs Negative) by assigning scores for pos or neg to 
    each slot of time.
    """
    
    SEG_LENGTH_SECONDS = 5.0
    NUM_SLOTS = 50
    MIN_NET_POS_SCORE = 3.0  
    MIN_NET_NEG_SCORE = 30.0 
    MAX_CONFLICT_SCORE = 10.0 
    MAX_POS_CONFLICT_RATIO = 0.5 
    MAX_TOLERATED_POS_FOR_NEG = 1.0

 
    pos_weights = np.zeros(NUM_SLOTS, dtype=float)
    neg_weights = np.zeros(NUM_SLOTS, dtype=float)
    implied_weights = np.zeros(NUM_SLOTS, dtype=float)
    slot_duration = SEG_LENGTH_SECONDS / NUM_SLOTS
    

    for p in propositions:

        # an array of slot weights for this proposition. 
        # different parts of the event that overlap with this segment are trusted more or less
        # depending on the duration of the event and how far from the edge of the event the slot is
        start_idx, end_idx, prop_weights = calculate_proposition_weights(p, NUM_SLOTS, slot_duration)
        
        if len(prop_weights) == 0:
            continue

        if p.proposition_type == PropositionType.VERIFIED_POS:
            pos_weights[start_idx:end_idx] += prop_weights
        elif p.proposition_type == PropositionType.VERIFIED_NEG:
            neg_weights[start_idx:end_idx] += prop_weights
        elif p.proposition_type == PropositionType.IMPLIED_NEG: 
            implied_weights[start_idx:end_idx] += prop_weights

    # Clamp Accumulations to 1.0
    # multiple pos or neg events shouldn't increase weight above 1.0
    pos_weights = np.minimum(1.0, pos_weights)
    neg_weights = np.minimum(1.0, neg_weights)
    implied_weights = np.minimum(1.0, implied_weights)

    # Check for overlaps between pos and neg
    # and see whether they exceed the conflict threshold
    conflict_mask = (pos_weights > 0) & (neg_weights > 0)
    conflict_score = np.sum(pos_weights[conflict_mask] + neg_weights[conflict_mask])
    
    if conflict_score > MAX_CONFLICT_SCORE:
        return None

    # Net Score Calculation: how much pos or neg
    # after cancelling out the overlaps
    effective_pos_signal = np.maximum(0, pos_weights - neg_weights)
    effective_neg_signal = np.maximum(0, neg_weights - pos_weights)
    
    total_net_pos_score = np.sum(effective_pos_signal)
    total_net_neg_score = np.sum(effective_neg_signal)
    total_raw_pos_score = np.sum(pos_weights)
    
    # Final Decision Logic
    # Positive Check
    if total_net_pos_score >= MIN_NET_POS_SCORE:
        loss = total_raw_pos_score - total_net_pos_score
        conflict_ratio = loss / total_raw_pos_score if total_raw_pos_score > 0 else 0.0
        if conflict_ratio <= MAX_POS_CONFLICT_RATIO:
            return 'positive'
                    
    # Negative Check
    if total_net_neg_score >= MIN_NET_NEG_SCORE:
        if total_net_pos_score <= MAX_TOLERATED_POS_FOR_NEG:
            # we must have at least enough negative coverage to consider it negative
            # but also we can't risk an incorrect positive label being present, even if the positive
            # score was low enough not to pass the positive threshold 
            return 'negative'

        
    # Implied Negative Fallback
    if total_raw_pos_score < MIN_NET_POS_SCORE and np.sum(implied_weights) > 0:
        return 'negative'
        
    return None
