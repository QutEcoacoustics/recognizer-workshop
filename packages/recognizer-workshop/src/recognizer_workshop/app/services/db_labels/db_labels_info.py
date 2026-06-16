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

from baw_helpers.baw_utils import extract_arid_and_domain
from recognizer_workshop.app.services.db_labels.dataclasses import LabelCounts
from baw_helpers.baw_helpers import get_events_for_sites, get_verifications_for_sites
from recognizer_workshop.app.agile_modeling_helpers import get_embeddings_by_source
from recognizer_workshop.app.create_database.create_database import add_parquet_to_db

log = logging.getLogger(__name__)



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


def get_db_arids(db) -> set:
    #TODO: this might be general enough to move to agile modeling helpers

    all_sources = db.get_all_sources()
    arids = set([extract_arid_and_domain(source) for source in all_sources])
    return arids


