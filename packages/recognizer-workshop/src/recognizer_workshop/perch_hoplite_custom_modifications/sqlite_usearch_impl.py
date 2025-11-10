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
        db_path = epath.Path(db_path)
        db_path.mkdir(parents=True, exist_ok=True)
        hoplite_path = db_path / HOPLITE_FILENAME


        # Only this line is modified from the parent class, to allow cross-thread usage
        #db = sqlite3.connect(hoplite_path.as_posix())
        db = sqlite3.connect(hoplite_path.as_posix(), check_same_thread=False)

        cursor = db.cursor()
        cursor.execute('PRAGMA journal_mode=WAL;')  # Enable WAL mode
        _setup_sqlite_tables(cursor)
        db.commit()

        # TODO(tomdenton): Check that config is consistent with the DB.
        metadata = _get_metadata(cursor, None)
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
        index_path = db_path / UINDEX_FILENAME
        if index_path.exists():
            ui = uindex.Index(
                ndim=usearch_cfg.embedding_dim,
                path=index_path,
                view=True,
            )
            ui_in_memory = False
        else:
            ui = uindex.Index(
                ndim=usearch_cfg.embedding_dim,
                metric=getattr(uindex.MetricKind, usearch_cfg.metric_name),
                expansion_add=usearch_cfg.expansion_add,
                expansion_search=usearch_cfg.expansion_search,
                dtype=usearch_dtype,
                path=index_path,
                view=False,
            )
            ui_in_memory = True

        return cls(
            db_path=db_path.as_posix(),
            db=db,
            ui=ui,
            embedding_dim=usearch_cfg.embedding_dim,
            embedding_dtype=usearch_cfg.dtype,
            _ui_loaded=ui_in_memory,
            _ui_updated=ui_in_memory,
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

    def get_all_labels(self) -> list[tuple]:
        """
        Efficiently fetches all labels (embedding_id, label, type) 
        from the database in a single query.
        """
        cursor = self._get_cursor()
        cursor.execute("SELECT embedding_id, label, type FROM hoplite_labels")
        return cursor.fetchall()











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