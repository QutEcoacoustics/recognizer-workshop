import dataclasses
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from perch_hoplite.db import interface as hoplite_interface

from baw_helpers import baw_api as baw_api
from baw_helpers import baw_helpers

from recognizer_workshop.app.services.verification import sample_for_verification


log = logging.getLogger(__name__)

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
        log.info(f"Found {len(audio_files)} audio files in {str(path)}")
        log.debug("\n".join([f"{i}: {f}" for i, f in enumerate(audio_files)]))
    return audio_files



def upload_events_for_verification(events):
  """
  Use the event upload endpoint to upload a json file as a multipart mime type.
  Add provenance information
  """
  pass








round_to_nearest_5 = lambda x: np.round(x / 5) * 5

# taken from the hoplite interface with a few modifications
def get_embeddings_by_source(
      db,
      dataset_name: str,
      source_id: str | None,
      offsets: np.ndarray | None = None,
  ) -> np.ndarray:

    cursor = db.db.cursor()

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

    metadata = db.get_metadata('legacy_config')
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
        log.info(f"Provenance {PROVENANCE_NAME} already exists. Using existing provenance.")
        return existing_provenances[0]
    else:
        log.info(f"Creating new provenance {PROVENANCE_NAME}.")

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

    log.info(f"Successfully created a random subset of {len(subset_df)} rows ({limit} for each of {len(labels)} labels) from '{inference_csv}'.")
    log.info(f"Output saved to '{subset_csv}'")

