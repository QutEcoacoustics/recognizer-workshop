import numpy as np
from perch_hoplite.db import interface


import sqlite3
import perch_hoplite.db.interface as hoplite_interface
from perch_hoplite.db.sqlite_usearch_impl import SQLiteUsearchDB, USEARCH_DTYPES, HOPLITE_FILENAME, USEARCH_CONFIG_KEY, UINDEX_FILENAME, _setup_sqlite_tables, _get_metadata, _insert_metadata

from etils import epath
from ml_collections import config_dict

from usearch import index as uindex


# this is a modified version of the SQLiteUsearchDB from perch_hoplite
# to allow for cross-thread usage of the DB connection
# an alternative would be to create a standalone factory function (no class or OO inheritance)
# which just does the create() setup and then calls the constructor of SQLiteUsearchDB. 
# the difference is that this returns a different type (ThreadedSQLiteUsearchDB)
class ThreadedSQLiteUsearchDB(SQLiteUsearchDB):
    """
    An inherited version of SQLiteUsearchDB that allows connections
    to be shared across threads by setting check_same_thread=False.
    - we are accessing private methods/attributes of the parent class,
      but this is necessary to avoid re-implementing everything or modifying the parent class.
    """
    @classmethod
    def create(
        cls,
        db_path: str,
        usearch_cfg: config_dict.ConfigDict | None = None,
    ):
        """
        Duplicate the parent class method but modify the sqlite3.connect with the check_same_thread=False option.
        """

        db_path = epath.Path(db_path)
        db_path.mkdir(parents=True, exist_ok=True)
        hoplite_path = db_path / HOPLITE_FILENAME 

        # Only this line is modified from the parent class, to allow cross-thread usage
        db = sqlite3.connect(hoplite_path.as_posix(), check_same_thread=False)

        # The rest of the method is exactly the same as the original
        cursor = db.cursor()
        cursor.execute('PRAGMA journal_mode=WAL;')
        _setup_sqlite_tables(cursor) 
        db.commit()

        metadata = _get_metadata(cursor, None) # Assuming accessible
        if (
            USEARCH_CONFIG_KEY in metadata
            and usearch_cfg is not None
            and metadata[USEARCH_CONFIG_KEY] != usearch_cfg
        ):
            raise ValueError(
                'A usearch_cfg was provided, but one already exists in the DB.'
            )
        elif USEARCH_CONFIG_KEY in metadata:
            usearch_cfg = metadata[USEARCH_CONFIG_KEY]
        elif usearch_cfg is None:
            raise ValueError('No usearch config found in DB and none provided.')
        else:
            _insert_metadata(cursor, USEARCH_CONFIG_KEY, usearch_cfg)
            db.commit()

        usearch_dtype = USEARCH_DTYPES[usearch_cfg.dtype]
        ui_mem = uindex.Index(
            ndim=usearch_cfg.embedding_dim,
            metric=getattr(uindex.MetricKind, usearch_cfg.metric_name),
            expansion_add=usearch_cfg.expansion_add,
            expansion_search=usearch_cfg.expansion_search,
            dtype=usearch_dtype,
        )
        index_path = db_path / UINDEX_FILENAME
        ui_disk_view = uindex.Index(ndim=usearch_cfg.embedding_dim)
        if index_path.exists():
            ui_disk_view.view(index_path.as_posix())
            ui = ui_disk_view
        else:
            ui = ui_mem
        
        return cls(
            db_path=db_path.as_posix(),
            db=db,
            ui=ui,
            _ui_mem=ui_mem, 
            _ui_disk_view=ui_disk_view,
            embedding_dim=usearch_cfg.embedding_dim,
            embedding_dtype=usearch_cfg.dtype,
        )
    

    def update_label_type(self, embedding_id: int, label: str, new_type: hoplite_interface.LabelType):
        """
        Finds a label by its embedding_id and string, and updates its type.
        """
        cursor = self._get_cursor()
        cursor.execute(
            """
            UPDATE hoplite_labels
            SET type = ?
            WHERE embedding_id = ? AND label = ?
            """,
            (new_type.value, embedding_id, label),
        )











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