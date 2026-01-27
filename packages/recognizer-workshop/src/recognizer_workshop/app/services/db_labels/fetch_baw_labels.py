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
from recognizer_workshop.app.services.db_labels.dataclasses import LabelName, LabelProposition, PropositionType

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
        compile_label_propositions(event, target_labels, target_negatives)
                   
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
        
        for segment in event['segments']:
            seg_start = segment['start']
            if seg_start not in items[source]:
                items[source][seg_start] = {}

            for label in target_labels:
                label_str = str(label) # the hoplite string representation of the label, e.g. "magpie (song)"
                
                # proposition for this target label on this event-segment e.g. None, TYPE_VERIFIED_POS etc ..
                prop_type = event['label_propositions'].get(label)

                if label_str not in items[source][segment['start']]:
                    items[source][segment['start']][label_str] = []

                if prop_type:
                    entry = LabelProposition(
                        label=label,
                        local_start=segment['local_start'],
                        local_end=segment['local_end'],
                        proposition_type=prop_type,
                        event_cover=segment['event_cover']
                    )
                
                    # so we now have a label type (e.g. TYPE_VERIFIED_POS) for this target label on this event/segment
                    # with its local start and end within the segment
                    items[source][seg_start][label_str].append(entry)

    return resolve_conflicts_and_format(items)
                    
 


def compile_label_propositions(event, target_labels: list[LabelName], target_negatives: UserDict[LabelName, list[LabelName]]):
    """
    For each target label, creates a list of label propositions for this event. 
    """

    # there may be overlapping events, such as a "positive for grasswren" event overlapping with a "negative for grasswren" event
    # or, conflicts between target-negatives and actual targets. E.g. "positive for target species grasswren" but "positive for non-target species fairywren, which is a negative example for grasswren"
    # leading to contradictory labels for the same segment of the source. 
    # let's compile a list of label propositions for each target label on this event, then later we can resolve conflicts when we consolodate segments across events.
    # IMPLIED_NEG: negative label based on non-target positive tagging. 
    # VERIFIED_POS: positive label based on target positive tagging or verification
    # VERIFIED_NEG: negative label based on target negative verification

    event['label_propositions'] = {}
    for target_label in target_labels:

        target_tag = target_label.get_baw_tags()[0]

        
        if event['label_info'].get(target_tag, None) == 'negative':
            event['label_propositions'][target_label] = PropositionType.VERIFIED_NEG
   
        elif event['label_info'].get(target_tag, None) == 'positive':
            event['label_propositions'][target_label] = PropositionType.VERIFIED_POS
    
        elif target_label in target_negatives:
            # for each positive tag associated with this event, check if it's in the list of negatives for this target label
            for label_info_text, label_info_type in event['label_info'].items():
                if label_info_type == 'positive' and target_negatives.target_has_item(target_label, label_info_text):
                    event['label_propositions'][target_label] = PropositionType.IMPLIED_NEG
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


def resolve_conflicts(propositions):
    """
    Resolves labeling conflicts (Positive vs Negative) by assigning scores for pos or neg to 
    each slot of time. Score is lower if the amount of the original event on this segment is small
    (because it is more likely to to be inaccurate bounding box causing the conflict). 
     
    Then we use a 3-Stage Process:
    
    1. CONFLICT CHECK:
       Wherever there is an overlap of pos and negative, we generate a conflict score. 
       If a Positive and Negative overlap for too long, we discard the segment as "conflicted".
       "Too long" depends on the scores that have gone into the overlap. 
       
    2. THE VOTING (Net Score Check):
       If the conflict is quiet (e.g., just boundary errors or weak "crumbs"), we let the 
       signals vote. We subtract Negative weights from Positive weights to find the 
       "Net Surviving Signal."
       
    3. THE SAFETY NET (Ratio Check):
       Even if a Positive signal survives the vote, we discard it if >50% of its original 
       strength was lost in the battle. This catches cases where a short event is technically 
       "won" but was fatally compromised.
    """
    
    NUM_SLOTS = 50
    MIN_NET_POS_SCORE = 2.0  # ~0.2s of clear positive
    MIN_NET_NEG_SCORE = 30.0 # ~3.0s of clear negative
    
    # THRESHOLD: How much disagreement do we tolerate?
    # A single slot (0.1s) with full Pos(1.0) vs Neg(1.0) has a conflict sum of 2.0.
    # A 0.5s overlap (5 slots) would have a score of 10.0.
    # Let's set the limit at 5.0. Any overlap larger than ~0.25s is a "Hard Conflict".
    MAX_CONFLICT_SCORE = 10.0 

    MAX_POS_CONFLICT_RATIO = 0.5  # If more than this much of the positive signal is disputed, discard it.

    # 1. Initialize & Populate Weights
    pos_weights = np.zeros(NUM_SLOTS, dtype=float)
    neg_weights = np.zeros(NUM_SLOTS, dtype=float)
    implied_weights = np.zeros(NUM_SLOTS, dtype=float)
    
    for p in propositions:
        idx_start = int(max(0, min(NUM_SLOTS, p.local_start * 10)))
        idx_end = int(max(0, min(NUM_SLOTS, p.local_end * 10)))
        weight = min(1.0, p.event_cover)
        
        if p.proposition_type == PropositionType.VERIFIED_POS:
            pos_weights[idx_start:idx_end] += weight
        elif p.proposition_type == PropositionType.VERIFIED_NEG:
            neg_weights[idx_start:idx_end] += weight
        elif p.proposition_type == PropositionType.IMPLIED_NEG: 
            implied_weights[idx_start:idx_end] += weight

    # Clamp to 1.0
    # This means that we reduce the weight below 1 if e.g. there are overlapping of the end of
    # a very long event, but adding multple events does not increase the weight beyond 1.0. 
    pos_weights = np.minimum(1.0, pos_weights)
    neg_weights = np.minimum(1.0, neg_weights)
    implied_weights = np.minimum(1.0, implied_weights)


    # Identify slots where we have active disagreement
    conflict_mask = (pos_weights > 0) & (neg_weights > 0)
    
    # Sum the energy from BOTH sides in the conflict zone
    conflict_score = np.sum(pos_weights[conflict_mask] + neg_weights[conflict_mask])
    
    if conflict_score > MAX_CONFLICT_SCORE:
        # Too much disagreement. The data is contradictory.
        return None

    # --- proceed to check for scores of pos and neg, now that we know the conflict is not too large ---

    # Calculate Net Scores for each slot. 
    # multiple positives on the same slot will add up, same for negatives.
    # where there are conflicts, assuming they were not big enough to discard, 
    # we subtract the weights against each other, i.e. disputed positive regions lose weight
    # based on the weight of the negative that is disputing it, and vice versa.
    effective_pos_signal = np.maximum(0, pos_weights - neg_weights)
    effective_neg_signal = np.maximum(0, neg_weights - pos_weights)
    
    total_net_pos_score = np.sum(effective_pos_signal)
    total_net_neg_score = np.sum(effective_neg_signal)
    total_raw_pos_score = np.sum(pos_weights)
    
    # Positive Check
    # We need to satisfy both the net score and the ratio check to be valid positive.
    # 1. total positive score > MIN_NET_POS_SCORE
    # 2. conflict ratio <= MAX_POS_CONFLICT_RATIO (i.e. not too much of the original positive signal was disputed)
    if total_net_pos_score >= MIN_NET_POS_SCORE:
        # raw positive score is positive score before considering negatives
        # net positive score is after subtracting negatives
        loss = total_raw_pos_score - total_net_pos_score
        conflict_ratio = loss / total_raw_pos_score if total_raw_pos_score > 0 else 0.0
        if conflict_ratio <= MAX_POS_CONFLICT_RATIO:
            return 'positive'
                    
    # Negative Check
    if total_net_neg_score >= MIN_NET_NEG_SCORE:
        return 'negative'
        
    # Implied Negative is only considered if there is negligible positive score
    # (i.e. we accept Implied Negs even if there are tiny positive "crumbs" that 
    # failed the threshold, effectively treating them as noise) 
    if total_raw_pos_score < MIN_NET_POS_SCORE and np.sum(implied_weights) > 0:
        return 'negative'
        
    return None


def resolve_conflicts_01(propositions):
    """
    We have segments with different proposals from different events. They don't necessarily take up the same part of the segment. 
    Depending on their priority and how much they overlap, we decide the final label for the segment.
    For example, a verified positive in the start of the segment and a verified negative at the end of the segment results in a positive. 
    But a verified negative completely covering a verified positive results in a contradiction and the segment is discarded for this label. 
    To simplify resolving overlaps, we divide the segment into a fixed number of slots then perform the binary masking on those slots.
    """
    
    NUM_SLOTS = 50
    MIN_POSITIVE_SLOTS_FOR_POSITIVE_LABEL = 2
    MIN_UNCONTESTED_POS_RATIO = 0.5 # If more than this much of the positive signal is disputed, discard it.

    # Consider negative only if at least this much of the segment is covered by negative propositions.
    # i.e. we can't really rule out that there are no un-tagged positives in the other part, but 
    #      we can be reasonably sure if the negative covers most of the segment
    MIN_UNCONTESTED_NEG_COVER = 0.6 



    # initialize masks to all false
    # where do we have positive propositions
    pos_mask = np.zeros(NUM_SLOTS, dtype=bool)
    # where do we have negative propositions
    veto_mask = np.zeros(NUM_SLOTS, dtype=bool)
    
    has_implied_negatives = False # Priority 1 tracker

    for p in propositions:
        # Convert time (0.0 - 5.0) to indices (0 - 50)
        idx_start = int(max(0, min(NUM_SLOTS, p.local_start * 10)))
        idx_end = int(max(0, min(NUM_SLOTS, p.local_end * 10)))
        
        if p.proposition_type == PropositionType.VERIFIED_POS:
            pos_mask[idx_start:idx_end] = True
        elif p.proposition_type == PropositionType.VERIFIED_NEG:
            veto_mask[idx_start:idx_end] = True
        elif p.proposition_type == PropositionType.IMPLIED_NEG: 
            has_implied_negatives = True


    # slots covered by positive but not negative
    uncontested_pos_mask = pos_mask & (~veto_mask)

    # slots covered by both positive and negative (inconsistent data)
    conflict_mask = pos_mask & veto_mask
    
    # number of uncontested positive slots
    num_uncontested_pos = np.sum(uncontested_pos_mask)

    total_pos_slots = np.sum(pos_mask)

    uncontested_ratio = num_uncontested_pos / total_pos_slots if total_pos_slots > 0 else 0.0

    if total_pos_slots > MIN_POSITIVE_SLOTS_FOR_POSITIVE_LABEL and uncontested_ratio >= MIN_UNCONTESTED_POS_RATIO: 
        # if there were enough positive parts not vetoed -> positive
        final_label = 'positive'

    elif np.any(conflict_mask):
        # all the positive parts completely covered by negative parts
        # that is probably bad data, so we discard
        final_label = None


    elif np.any(veto_mask):
        # if there were any explicit negative parts -> negative
        final_label = 'negative'
    elif has_implied_negatives:
        # if there were any implied negatives (presence of a different label we want to use as a negative) -> negative
        final_label = 'negative'
    else:
        final_label = None 
    return final_label



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

        # Where does this event start/end inside this specific 0-5s window?
        local_start = overlap_start - current_segment_start
        local_end = overlap_end - current_segment_start

        # Calculate the required proportions
        segment_cover = overlap_duration / segment_duration
        event_cover = overlap_duration / total_event_duration

        # Store the results for this segment
        results.append({
            "start": current_segment_start,
            "end": current_segment_end,
            "local_start": round(local_start, 3),
            "local_end": round(local_end, 3),
            "segment_cover": round(segment_cover, 2), # Round for cleaner output
            "event_cover": round(event_cover, 2),
            "cover_score": round(segment_cover + event_cover, 2),

        })

        # Move to the next segment
        current_segment_start += segment_duration
    
    max_cover_score = max(seg['cover_score'] for seg in results)

    # filter out segments that don't have enough "coverage"
    # the sum of the amount of the segment covered and the amount of the event inside the segment must be greater than the threshold. 
    # i.e. if only a small fraction of the segment is covered by the event, and only a small fraction of the event is inside the segment, then discard that segment.
    # or if none of segments have that much then keep just the one segment for this event with the most coverage' - probably this shouldn't be possible
    # for low enough thresholds, but just in case.     
    coverage_threshold = 0.2
    results = [seg for seg in results if seg['cover_score'] > coverage_threshold or seg['cover_score'] == max_cover_score]

    return results


