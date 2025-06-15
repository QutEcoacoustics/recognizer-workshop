import numpy as np
from perch_hoplite.db import interface

# augmented from perch_hoplite.db.sqlite_usearch_impl
# to allow for dataset filtering
def get_embedding_ids(db, dataset=None) -> np.ndarray:
    cursor = db._get_cursor()
    if dataset is not None:
        cursor.execute(
          """
          SELECT e.id 
          FROM hoplite_embeddings e
          JOIN hoplite_sources s ON s.id = e.source_idx
          WHERE s.dataset = ?
          """,
            (dataset,),
        )
    else:
        cursor.execute("""SELECT id FROM hoplite_embeddings;""")
    
    return np.array(tuple(int(c[0]) for c in cursor.fetchall()))


def get_all_sources(db) -> list[interface.EmbeddingSource]:
    """
    Efficiently retrieves all unique sources from the database.
    
    This method queries the hoplite_sources table directly, which is far
    more efficient than iterating through every embedding. Note that this
    does not return offset information, as offsets are per-embedding.
    """
    cursor = db._get_cursor()
    # We only need to select from the sources table.
    cursor.execute(
        """
        SELECT DISTINCT dataset, source
        FROM hoplite_sources;
        """
    )
    all_sources = []
    for row in cursor.fetchall():
        dataset, source_id = row
        # Create an EmbeddingSource object. Since offsets are per-embedding,
        # we can represent a "generic" source with empty offsets.
        source_obj = interface.EmbeddingSource(
            dataset_name=dataset, source_id=source_id, offsets=np.array([])
        )
        all_sources.append(source_obj)
    
    return all_sources