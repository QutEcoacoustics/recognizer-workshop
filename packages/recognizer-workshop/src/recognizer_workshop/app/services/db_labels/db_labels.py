from pathlib import Path
import numpy as np

from recognizer_workshop.app.agile_modeling_helpers import list_audio_files
from perch_hoplite import audio_io
from perch_hoplite.db import interface as hoplite_interface



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


import dataclasses
from typing import Dict, List, Optional


@dataclasses.dataclass
class LabelCounts:
    """A simple data structure to hold positive and negative counts for a label."""
    label_name: str
    positive: int = 0
    negative: int = 0

    def __str__(self) -> str:
        return f"{self.label_name}: {self.positive} positive, {self.negative} negative"


@dataclasses.dataclass
class DatabaseSummary:
    total_embeddings: int
    datasets: List[str]
    label_counts: Dict[str, LabelCounts]

    def __str__(self) -> str:
        lines = [
            "DB Summary:",
            f"  Number of embeddings: {self.total_embeddings}",
            f"  {len(self.datasets)} Dataset(s): {', '.join(self.datasets)}",
            ""
        ]
        
        if self.label_counts:
            lines.append("Label counts:")
            for label_name in sorted(self.label_counts.keys()):
                lines.append(f"  {str(self.label_counts[label_name])}")
        else:
            lines.append("No labeled embeddings in db.")
            
        return "\n".join(lines)


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






# def print_label_counts(db, label: str | None = None):
#     """Prints the number of positive and negative of each label in the db"""
    
#     positive = db.get_class_counts(hoplite_interface.LabelType.POSITIVE)
#     negative = db.get_class_counts(hoplite_interface.LabelType.NEGATIVE)
#     if (len(positive.items()) > 0):
#         print('Label counts:')
#         for lbl, count in positive.items():
#             if label is None or lbl == label:
#                 print(f'{lbl}: {count} positive, {negative[lbl]} negative')
#     else:
#         print('No labeled embeddings in db.')


# def db_summary(db):
#     """Debugging/sanity checking to see what's in the database"""

#     print('DB Summary:')
#     print(f'Number of embeddings: {db.count_embeddings()}')
#     datasets = db.get_dataset_names()
#     print(f'{len(datasets)} Dataset(s):')
#     for dataset in datasets:
#         # This is slow, so commented out for now
#         #print(f'{dataset}: {len(self.db.get_embeddings_by_source(dataset_name=dataset, source_id=None))} embeddings')
#         print(f'{dataset}')
#     print_label_counts(db)


# def print_db_summary(db_summary):
#     """Prints the database summary in a user-friendly format."""
#     print("Database Summary:")
#     for key, value in db_summary.items():
#         print(f"  {key}: {value}")