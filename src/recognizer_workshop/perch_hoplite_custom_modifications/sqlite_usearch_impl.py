import numpy as np

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