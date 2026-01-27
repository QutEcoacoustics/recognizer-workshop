import pytest
from unittest.mock import MagicMock, patch

# Import the module under test
from recognizer_workshop.app.services.db_labels import fetch_baw_labels as target_module

# Import the REAL dataclass
from recognizer_workshop.app.services.db_labels.dataclasses import LabelName, TargetNegatives, PropositionType, LabelProposition

# --- Fixtures & Mocks ---

@pytest.fixture(scope="module")
def mock_api():
    """Mocks the BAW API client."""
    api = MagicMock()
    api.base_url = "https://api.ecosounds.org"
    
    # Mock the Tag Cache lookup
    tag_lookup = {
        555: {'id': 555, 'text': 'Magpie'},
        666: {'id': 666, 'text': 'Crow'},
        777: {'id': 777, 'text': 'Fairy Wren'},
        888: {'id': 888, 'text': 'Magpie Song'}, 
    }
    
    api.tags.get_by_id.side_effect = lambda tag_id: tag_lookup.get(tag_id)
    return api


@pytest.fixture
def sample_events():
    """Sample events for the Integration Test."""

    events = [
        # Event 1: Verified Positive 'Magpie Song' (Priority 2)
        {
            'start_time_seconds': 11.0, 'end_time_seconds': 17.0,
            'taggings': [{'tag_id': 888}],
            # Correct > Incorrect = Positive
            'verification_summary': [{'tag_id': 888, 'correct': 5, 'incorrect': 0}], 
        },
        # Event 2: Verified Negative 'Magpie Song' (Priority 3)
        {
            'start_time_seconds': 18.0, 'end_time_seconds': 25.0,
            'taggings': [{'tag_id': 888}],
            # Incorrect > Correct = Negative
            'verification_summary': [{'tag_id': 888, 'correct': 0, 'incorrect': 5}],
        },
        # Event 3: Implied Negative (Priority 1)
        # Positive for 'Fairy Wren', which is a known negative for Magpie
        {
            'start_time_seconds': 30.0, 'end_time_seconds': 35.0,
            'taggings': [{'tag_id': 777}],
            'verification_summary': [{'tag_id': 777, 'correct': 5, 'incorrect': 0}],
        }
    ]

    for i, event in enumerate(events):
        event['id'] = i + 1
        event['audio_recording_id'] = 999
        event['provenance_id'] = 10

# --- Unit Tests ---

def test_compile_label_propositions():
    """
    Test Priority Logic using the REAL LabelName class.
    """

    # Note: LabelName auto-capitalizes, so "song" becomes "Song"
    target = LabelName("Magpie", "song") 
    target_labels = [target]
    
    # We are saying that our target of Maqpie song might be confused with a Fairy Wren, so we want to 
    # add Fairy Wren examples as negatives for Magpie Song.
    target_negatives = TargetNegatives({ LabelName("Magpie", "song"): [LabelName("Fairy Wren")]})
    
    # 2. Setup Events
    # The keys here must match target.get_baw_tags()[0] -> "Magpie Song"
    events = [
        {'label_info': {'Magpie Song': 'positive'}, 'label_propositions': {}},     # Pri 2
        {'label_info': {'Magpie Song': 'negative'}, 'label_propositions': {}},     # Pri 3
        {'label_info': {'Fairy Wren': 'positive'}, 'label_propositions': {}},      # Pri 1
    ]
    
    # modifies events in place
    target_module.compile_label_propositions(events[0], target_labels, target_negatives)
    target_module.compile_label_propositions(events[1], target_labels, target_negatives)
    target_module.compile_label_propositions(events[2], target_labels, target_negatives)
    
    
    assert events[0]['label_propositions'][target] == PropositionType.VERIFIED_POS
    assert events[1]['label_propositions'][target] == PropositionType.VERIFIED_NEG
    assert events[2]['label_propositions'][target] == PropositionType.IMPLIED_NEG


def test_resolve_conflicts_and_format():
    """Test the resolution logic and formatting of label propositions."""
    target_label = LabelName("Magpie", "song")
    target_label_str = str(target_label)



    items = {
        'source_1': {
            10.0: {
                target_label_str: [LabelProposition(target_label, PropositionType.VERIFIED_NEG, 0.1, 0.9), 
                                   LabelProposition(target_label, PropositionType.VERIFIED_POS, 2.1, 2.9)] # pos wins
            },
            20.0: {
                target_label_str: [LabelProposition(target_label, PropositionType.VERIFIED_NEG, 0.1, 3.9), 
                                   LabelProposition(target_label, PropositionType.VERIFIED_POS, 2.1, 2.9)] # neg fully covers small pos -> Neg
            },
            25.0: {
                target_label_str: [LabelProposition(target_label, PropositionType.VERIFIED_NEG, 0.1, 3.9), 
                                   LabelProposition(target_label, PropositionType.VERIFIED_POS, 2.1, 2.9)] # invalid neg fully covers large pos -> discarded
            },
            30.0: {
                target_label_str: [LabelProposition(target_label, PropositionType.VERIFIED_POS, 0.1, 3.9), 
                                   LabelProposition(target_label, PropositionType.VERIFIED_NEG, 2.1, 2.9)] # pos fully covers neg -> pos wins
            },
            40.0: {
                target_label_str: [LabelProposition(target_label, PropositionType.IMPLIED_NEG, 0.1, 3.9), 
                                   LabelProposition(target_label, PropositionType.VERIFIED_POS, 2.1, 2.9)] # implied neg fully covers pos -> pos wins
            },
            50.0: {
                target_label_str: [LabelProposition(target_label, PropositionType.VERIFIED_NEG, 0.1, 3.9), 
                                   LabelProposition(target_label, PropositionType.IMPLIED_NEG, 2.1, 2.9)] # verified neg + implied neg -> neg wins
            }
        }
    }
    
    inserted, counts = target_module.resolve_conflicts_and_format(items)
    
    assert inserted['source_1'][target_label_str]['pos'] == [10.0, 30.0, 40.0]
    assert inserted['source_1'][target_label_str]['neg'] == [50.0]
    
    assert counts[target_label_str]['positive'] == 3
    assert counts[target_label_str]['negative'] == 1




# Convenience aliases
POS = PropositionType.VERIFIED_POS
NEG = PropositionType.VERIFIED_NEG
IMP = PropositionType.IMPLIED_NEG
TARGET = LabelName("Magpie", "song")

def p(prop_type, start, end, event_cover=1.0):
    """
    Helper to create propositions. 
    Defaults to event_cover=1.0 per your update.
    """
    return LabelProposition(
        label=TARGET,
        proposition_type=prop_type,
        local_start=start,
        local_end=end,
        event_cover=event_cover
    )

@pytest.mark.parametrize("scenario, propositions, expected", [
    (
        "Case 1: Verified Pos disjoint from Verified Neg -> Pos wins",
        [
            p(POS, 0.0, 1.0),
            p(NEG, 2.0, 3.0)
        ],
        "positive"
    ),
    (
        "Case 2: Verified Pos fully covered by Verified Neg -> Discarded",
        [
            p(POS, 1.0, 2.0),
            p(NEG, 0.0, 3.0)
        ],
        None # Expect Discard because Pos is completely neutralized
    ),
    (
        "Case 3: Verified Pos fully covers Verified Neg and exceeds max conflict ->  Discarded ",
        [
            p(POS, 0.0, 3.0),
            p(NEG, 1.0, 1.6)
        ],
        None # Pos has 2.0s clear signal vs 1.0s conflict. This is too much conflict and is discarded
    ),
    (
        "Case 3.2: Verified Pos fully covers Verified Neg and doesn't max conflict ->  Pos wins ",
        [
            p(POS, 0.0, 3.0),
            p(NEG, 1.0, 1.5)
        ],
        "positive" # Pos has 2.5s clear signal vs 0.5s conflict. This is acceptable and Pos wins
    ),
    (
        "Case 4: Verified Pos overlapping Implied Neg -> Pos wins",
        [
            p(POS, 0.0, 2.0),
            p(IMP, 1.0, 3.0)
        ],
        "positive" # Implied Negs don't subtract from Pos signal
    ),
    (
        "Case 5: Verified Pos overlapping Implied Neg (Duplicate logic of Case 4) -> Pos wins",
        [
            p(POS, 0.0, 2.0),
            p(IMP, 1.0, 3.0)
        ],
        "positive"
    ),
    (
        "Case 6: Verified Neg overlapping Implied Neg -> Neg wins",
        [
            p(NEG, 0.0, 2.0),
            p(IMP, 1.0, 3.0)
        ],
        "negative"
    ),
    (
        "Case 7: Verified Neg fully covers Implied Neg -> Neg wins",
        [
            p(NEG, 0.0, 3.0),
            p(IMP, 1.0, 2.0)
        ],
        "negative"
    ),
    (
        "Case 8: Verified Neg overlapping Implied Neg overlapping Verified Pos (Disjoint Pos/Neg) -> Pos wins",
        [
            p(NEG, 0.0, 2.0),
            p(IMP, 1.0, 3.0),
            p(POS, 2.0, 4.0) # Pos starts where Neg ends. 
        ],
        "positive"
    )
])
def test_resolve_conflicts_small_events(scenario, propositions, expected):
    """Test the conflict resolution logic for standard 'small' events (cover=1.0)."""
    result = target_module.resolve_conflicts(propositions)
    assert result == expected, f"Failed on scenario: {scenario}"



@pytest.mark.parametrize("scenario, propositions, expected", [
    (
        "Case 1: Pos Crumb (0.1 cover) vs Disjoint Neg",
        [
            p(POS, 0.0, 1.0, event_cover=0.1), # Score: 1.0s * 10 slots * 0.1 = 1.0
            p(NEG, 2.0, 3.0)
        ],
        None # EXPECTED: Discard. Score (1.0) < Threshold (2.0).
    ),
    (
        "Case 2: Tiny Pos Crumb (0.05 cover)",
        [
            p(POS, 0.0, 1.0, event_cover=0.05), # Score: 0.5
            p(NEG, 2.0, 3.0)
        ],
        None # EXPECTED: Discard. Score (0.5) < Threshold (2.0).
    ),
    (
        "Case 3: Pos Crumb fully covered by Large Neg",
        [
            p(POS, 0.0, 0.5, event_cover=0.2), # Weight 0.2
            p(NEG, 0.0, 5.0)                   # Weight 1.0 (Dominates)
        ],
        "negative" 
        # EXPECTED: Negative.
        # Net Neg Calculation:
        # - Overlap (0-0.5s): 5 slots * (1.0 - 0.2) = 4.0.
        # - Clear (0.5-5.0s): 45 slots * 1.0 = 45.0.
        # - Total Net Neg: 49.0. (49.0 > 30.0) -> Negative.
    ),
    (
        "Case 3.2: Pos Crumb fully covered by Large Neg",
        [
            p(POS, 0.0, 1.0, event_cover=0.2), # Weight 0.2
            p(NEG, 0.0, 5.0)                   # Weight 1.0
        ],
        None
        # EXPECTED: Discard (None).
        # 10 slots * (0.2 Pos + 1.0 Neg) = 12.0 Score.
        # Check: 12.0 > 10.0 (Max Conflict). -> FAIL.
    ),
    (
        "Case 4: Pos Crumb covered by Medium Neg",
        [
            p(POS, 0.0, 0.5, event_cover=0.2), 
            p(NEG, 0.0, 3.5) # 3.5s duration
        ],
        "negative" 
        # EXPECTED: Negative.
        # Net Neg Calculation:
        # - Overlap (0-0.5s): 5 slots * (1.0 - 0.2) = 4.0.
        # - Clear (0.5-3.5s): 30 slots * 1.0 = 30.0.
        # - Total Net Neg: 34.0. (34.0 > 30.0) -> Negative.
    ),
    (
        "Case 5: Solid Pos overlapping Solid Neg (Small Overlap)",
        [
            p(POS, 0.0, 3.0), # 3.0s * 1.0
            p(NEG, 2.5, 3.5)  # 1.0s * 1.0
        ],
        "positive"
        # EXPECTED: Positive.
        # Overlap (1-2s): Cancelled out (0 score).
        # Clear (0-1, 2-3): 2.0s * 1.0 = 20.0 score.
        # Raw Pos Score = 30.0. Net = 20.0.
        # Conflict Ratio = (30-20)/30 = 0.33.
        # 0.33 < 0.5 Max Ratio -> Positive.
    ),
    (
        "Case 6: Solid Pos overlapping Implied Neg",
        [
            p(POS, 0.0, 2.0),
            p(IMP, 1.0, 3.0) 
        ],
        "positive"
        # EXPECTED: Positive.
        # Implied negatives do not subtract from positive signal.
    ),
    (
        "Case 7: Solid Neg overlapping Implied Neg",
        [
            p(NEG, 0.0, 2.0),
            p(IMP, 1.0, 3.0)
        ],
        "negative" 
        # EXPECTED: Negative. No positive signal.
    ),
    (
        "Case 8: Neg + Implied + Pos (Disjoint)",
        [
            p(NEG, 0.0, 2.0),
            p(IMP, 1.0, 3.0),
            p(POS, 2.0, 4.0) # Starts after Neg ends. No conflict.
        ],
        "positive" 
        # EXPECTED: Positive. Net Pos Score is 20.0 (High enough).
    )
])
def test_resolve_conflicts_scenarios(scenario, propositions, expected):
    """
    Parametrized test for conflict resolution.
    """
    result = target_module.resolve_conflicts(propositions)
    assert result == expected, f"Failed on scenario: {scenario}"


# integration test for the full flow

@pytest.fixture(scope="module")
def full_flow_results(mock_api):
    """
    Sets up the Mock API and Events, runs the logic ONCE, 
    and returns the processed items and counts for the specific source/label.
    """
    magpieSong = LabelName("Magpie", "song")
    fairyWren = LabelName("Fairy Wren")

    # Define overlapping events
    events = [
        # Event 1: Verified Positive 'Magpie Song' (11-17s)
        {
            'start_time_seconds': 11.0, 'end_time_seconds': 17.0,
            'taggings': [{'tag_id': 888}],
            'verification_summary': [{'tag_id': 888, 'correct': 5, 'incorrect': 0}], 
        },
        # Event 2: Verified Negative 'Magpie Song' (18-38s)
        {
            'start_time_seconds': 18.0, 'end_time_seconds': 38.0,
            'taggings': [{'tag_id': 888}],
            'verification_summary': [{'tag_id': 888, 'correct': 0, 'incorrect': 5}],
        },
        # Event 3: Verified Positive 'Magpie Song' (26-29s)
        {
            'start_time_seconds': 26.0, 'end_time_seconds': 29.0,
            'taggings': [{'tag_id': 888}],
            'verification_summary': [{'tag_id': 888, 'correct': 5, 'incorrect': 0}], 
        },
        # Event 4: Verified Positive 'Magpie Song' (37-43s)
        {
            'start_time_seconds': 37.0, 'end_time_seconds': 43.0,
            'taggings': [{'tag_id': 888}],
            'verification_summary': [{'tag_id': 888, 'correct': 5, 'incorrect': 0}], 
        },
        # Event 5: Implied Negative 'Fairy Wren' (39-48s)
        {
            'start_time_seconds': 39.0, 'end_time_seconds': 48.0,
            'taggings': [{'tag_id': 777}],
            'verification_summary': [{'tag_id': 777, 'correct': 5, 'incorrect': 0}],
        }
    ]

    # Add dummy IDs
    for i, event in enumerate(events):
        event['id'] = i + 1
        event['audio_recording_id'] = 999
        event['provenance_id'] = 10

    target_labels = [magpieSong]
    target_negatives = TargetNegatives({magpieSong: [fairyWren]})
    
    # Run the Logic
    with patch.object(target_module, 'get_events_for_sites', return_value=events):
        items, counts = target_module.labels_from_event_taggings(
            mock_api,
            filter_params={},
            target_labels=target_labels,
            target_negatives=target_negatives
        )
    
    source_url = "https://api.ecosounds.org/audio_recordings/999/original"
    label_key = str(magpieSong)
    
    # Return specific results for the Magpie label on this source
    return items[source_url][label_key], counts[label_key]


# --- 2. The Parametrized Tests ---

@pytest.mark.parametrize("offset, expected_type, reason", [
    (10.0, 'pos', "From Event 1 only"),
    
    (15.0, 'pos', "Event 1 plus overlap with Event 2 (still pos)"),
    
    (20.0, 'neg', "Event 2 only"),
    
    (25.0, None,  "DISCARD: Event 3 (Pos) completely covered by Event 2 (Neg) -> Bad Data"),
    
    (30.0, 'neg', "From Event 2 only"),
    
    (35.0, 'pos', "From Event 4 (overlap with neg Event 2 and implied neg resolves to pos)"),
    
    (40.0, 'pos', "From Event 4: Implied Neg (Ev 5) doesn't block Verified Pos"),
    
    (45.0, 'neg', "From Event 5 implied neg only"),
])
def test_integration_scenarios(full_flow_results, offset, expected_type, reason):
    """
    Validates that each 5-second segment resolves to the correct label.
    """
    segment_data, _ = full_flow_results
    
    pos_list = segment_data['pos']
    neg_list = segment_data['neg']

    if expected_type == 'pos':
        assert offset in pos_list, f"Offset {offset} should be POSITIVE. Reason: {reason}"
        assert offset not in neg_list, f"Offset {offset} should NOT be negative."
    
    elif expected_type == 'neg':
        assert offset in neg_list, f"Offset {offset} should be NEGATIVE. Reason: {reason}"
        assert offset not in pos_list, f"Offset {offset} should NOT be positive."
        
    elif expected_type is None:
        assert offset not in pos_list, f"Offset {offset} should be DISCARDED (found in pos). Reason: {reason}"
        assert offset not in neg_list, f"Offset {offset} should be DISCARDED (found in neg). Reason: {reason}"


def test_integration_counts(full_flow_results):
    """
    Validates the global counts for the integration run.
    """
    _, counts = full_flow_results
    
    # Expected Positives: 10.0, 15.0, 35.0, 40.0
    assert counts['positive'] == 4, f"Expected 4 positives, got {counts['positive']}"
    
    # Expected Negatives: 20.0, 30.0, 45.0
    assert counts['negative'] == 3, f"Expected 3 negatives, got {counts['negative']}"

# # --- Integration Test ---

# def test_labels_from_event_taggings_full_flow(mock_api, sample_events):
#     """
#     Tests the full flow with Real Labels + Mock API + Mock Event Fetcher.
#     """

#     magpieSong = LabelName("Magpie", "song")
#     fairyWren = LabelName("Fairy Wren")

#     # sample events
#     # all events overlap with each other to test priority resolution
#     # segments where e2 overlaps with e1 and e3 should resolve to negative for 888
#     # segments where e3 overlaps with e4 should resolve to positive. 
#     events = [
#         # Event 1: Verified Positive 'Magpie Song'
#         {
#             'start_time_seconds': 11.0, 'end_time_seconds': 17.0,
#             'taggings': [{'tag_id': 888}],
#             # Correct > Incorrect = Positive
#             'verification_summary': [{'tag_id': 888, 'correct': 5, 'incorrect': 0}], 
#         },
#         # Event 2: Verified Negative 'Magpie Song' 
#         {
#             'start_time_seconds': 18.0, 'end_time_seconds': 38.0,
#             'taggings': [{'tag_id': 888}],
#             # Incorrect > Correct = Negative
#             'verification_summary': [{'tag_id': 888, 'correct': 0, 'incorrect': 5}],
#         },
#         # Event 3: Verified Positive 'Magpie Song'
#         {
#             'start_time_seconds': 26.0, 'end_time_seconds': 29.0,
#             'taggings': [{'tag_id': 888}],
#             # Correct > Incorrect = Positive
#             'verification_summary': [{'tag_id': 888, 'correct': 5, 'incorrect': 0}], 
#         },
#         # Event 4: Verified Positive 'Magpie Song'
#         {
#             'start_time_seconds': 37.0, 'end_time_seconds': 43.0,
#             'taggings': [{'tag_id': 888}],
#             # Correct > Incorrect = Positive
#             'verification_summary': [{'tag_id': 888, 'correct': 5, 'incorrect': 0}], 
#         },
#         # Event 5: Implied Negative (Priority 1)
#         # Positive for 'Fairy Wren', which is a known negative for Magpie
#         {
#             'start_time_seconds': 39.0, 'end_time_seconds': 48.0,
#             'taggings': [{'tag_id': 777}],
#             'verification_summary': [{'tag_id': 777, 'correct': 5, 'incorrect': 0}],
#         }
#     ]

#     expected_segments_labels = [
#         (10.0, str(magpieSong), 'pos'),  # From Event 1 only
#         (15.0, str(magpieSong), 'pos'),  # Event 1 plus overlap with Event 2 (still pos)
#         (20.0, str(magpieSong), 'neg'),  # Event 2 only
#         # no segment label at 25. a pos(event 3) completely covered by a neg (event 2)is discarded as bad data
#         (30.0, str(magpieSong), 'neg'),  # From Event 2 only 
#         (35.0, str(magpieSong), 'pos'),  # From Event 4 (overlap with neg Event 2 and implied neg event 5 resolves to pos)
#         (40.0, str(magpieSong), 'pos'),  # From Event 4, even though it's completely covered by the implied neg Event 5, this doesn't block the pos from the verified pos
#         (45.0, str(magpieSong), 'neg'),  # From Event 5 implied neg only
#     ]

#     for i, event in enumerate(events):
#         event['id'] = i + 1
#         event['audio_recording_id'] = 999
#         event['provenance_id'] = 10

#     # Setup Real Data Objects
#     target_labels = [magpieSong]
    
#     # Key must match str(LabelName) -> "Magpie - Song"
#     target_negatives = TargetNegatives({magpieSong: [fairyWren]})
    
#     # Patch the network call to return our sample events
#     with patch.object(target_module, 'get_events_for_sites', return_value=events) as mock_get:
        
#         items, counts = target_module.labels_from_event_taggings(
#             mock_api,
#             filter_params={},
#             target_labels=target_labels,
#             target_negatives=target_negatives
#         )
        
#     # 4. Assertions
#     source_url = "https://api.ecosounds.org/audio_recordings/999/original"
    
#     # NOTE: The dict keys use the string representation of the Label object
#     # For LabelName("Magpie", "song"), str() returns "Magpie - Song"
#     label_key = str(target_labels[0]) 

#     # Verify we got data for this source
#     assert source_url in items, "Source URL missing from results"
#     source_results = items[source_url]

#     # Helper function to check individual segments
#     def check_segment(offset, expected_type):
#         """
#         offset: float time
#         expected_type: 'pos', 'neg', or None (for discard)
#         """
#         # Ensure the offset exists in the map (even if lists are empty)
#         # Note: If logic completely skips a segment, this might raise KeyError, 
#         # which is good for testing.
#         #segment_data = source_results[label_key]
        
#         pos_list = source_results[label_key]['pos']
#         neg_list = source_results[label_key]['neg']

#         if expected_type == 'pos':
#             assert offset in pos_list, f"Offset {offset} should be POSITIVE"
#             assert offset not in neg_list, f"Offset {offset} should NOT be negative"
        
#         elif expected_type == 'neg':
#             assert offset in neg_list, f"Offset {offset} should be NEGATIVE"
#             assert offset not in pos_list, f"Offset {offset} should NOT be positive"
            
#         elif expected_type is None:
#             # Discard Case: Should not appear in either list for this label
#             assert offset not in pos_list, f"Offset {offset} should be DISCARDED (found in pos)"
#             assert offset not in neg_list, f"Offset {offset} should be DISCARDED (found in neg)"

#     # A. Validate the Expected Segments (Pos/Neg)
#     for offset, _label_str, outcome in expected_segments_labels:
#         check_segment(offset, outcome)

#     # B. Validate the Explicit Discard (The "Bad Data" case at 25.0)
#     # Event 3 (Pos) was fully covered by Event 2 (Neg) -> Contradiction
#     check_segment(25.0, None)

#     # C. Validate Counts
#     # Positives: 10.0, 15.0, 35.0, 40.0 = 4
#     # Negatives: 20.0, 30.0, 45.0 = 3
#     # Discarded: 25.0 = 0
#     actual_pos_count = counts[label_key]['positive']
#     actual_neg_count = counts[label_key]['negative']
    
#     assert actual_pos_count == 4, f"Expected 4 positives, got {actual_pos_count}"
#     assert actual_neg_count == 3, f"Expected 3 negatives, got {actual_neg_count}"
