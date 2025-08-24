from pathlib import Path
import numpy as np

from recognizer_workshop.agile_modeling_helpers import list_audio_files
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


def print_label_counts(db, label: str | None = None):
    """Prints the number of positive and negative of each label in the db"""
    
    positive = db.get_class_counts(hoplite_interface.LabelType.POSITIVE)
    negative = db.get_class_counts(hoplite_interface.LabelType.NEGATIVE)
    if (len(positive.items()) > 0):
        print('Label counts:')
        for lbl, count in positive.items():
            if label is None or lbl == label:
                print(f'{lbl}: {count} positive, {negative[lbl]} negative')
    else:
        print('No labeled embeddings in db.')


def db_summary(db):
    """Debugging/sanity checking to see what's in the database"""

    print('DB Summary:')
    print(f'Number of embeddings: {db.count_embeddings()}')
    datasets = db.get_dataset_names()
    print(f'{len(datasets)} Dataset(s):')
    for dataset in datasets:
        # This is slow, so commented out for now
        #print(f'{dataset}: {len(self.db.get_embeddings_by_source(dataset_name=dataset, source_id=None))} embeddings')
        print(f'{dataset}')
    print_label_counts(db)
