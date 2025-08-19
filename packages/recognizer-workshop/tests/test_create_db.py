
import io
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from perch_hoplite.db import sqlite_usearch_impl
from recognizer_workshop.perch_hoplite_custom_modifications import sqlite_usearch_impl as sqlite_usearch_impl_custom

from recognizer_workshop.create_database.create_database import (
    iter_parquet_dataframes,
    add_parquet_to_db,
    convert_source_to_original_url,
    convert_parquet,
    create_empty_db
)



def _write_parquet(df: pd.DataFrame, path: Path):
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)

def _parquet_bytes(df: pd.DataFrame) -> io.BytesIO:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df), buf)
    buf.seek(0)
    return buf

def _make_df_for_arid(arid: int, n: int = 3, emb_dim: int = 2) -> pd.DataFrame:
    """
    Build a minimal dataframe for a single ARID with meta columns first:
    ['source', 'channel', 'offset', 'f0', 'f1', ...]
    """
    # Source is already in final URL form so convert_source_to_original_url() is a no-op
    base = f"https://api.ecosounds.org/audio_recordings/{arid}/original"
    data = {
        "source": [base] * n,
        "channel": [0] * n,
        "offset": [float(i) for i in range(n)],
    }
    # Add emb columns e0..e{emb_dim-1}; values vary so duplicates can be detected by count, not content
    for j in range(emb_dim):
        data[f"f{j}"] = [float(arid + i + j / 10.0) for i in range(n)]
    return pd.DataFrame(data)


def _write_parquets_with_arids(dirpath: Path, arids: Iterable[int], rows_per_file: int = 3, emb_dim: int = 2) -> list[Path]:
    """
    Create one parquet per ARID. The filename contains the ARID (e.g., a_111.parquet);
    this is important because the local-fragment path is how ARID skipping works.
    """
    paths = []
    for idx, arid in enumerate(arids):
        df = _make_df_for_arid(arid, n=rows_per_file, emb_dim=emb_dim)
        # include the arid in the filename so iter_parquet_dataframes() can extract it
        p = dirpath / f"file_{idx}_{arid}.parquet"
        _write_parquet(df, p)
        paths.append(p)
    return paths



def test_iter_parquet_dataframes_local_skip(tmp_path):
    # Make two files with ARIDs in filenames: a_111.parquet, b_222.parquet
    df = pd.DataFrame({"source": ["s"], "channel": [0], "offset": [0.0], "e0": [1.0]})

    p1 = tmp_path / "a_111.parquet"
    p2 = tmp_path / "b_222.parquet"
    _write_parquet(df, p1)
    _write_parquet(df, p2)

    # Skip 222
    dfs = list(iter_parquet_dataframes(tmp_path, skip_arids_in=[222]))
    assert len(dfs) == 1
    # Make sure we got the content of the kept file
    assert list(dfs[0].columns[:3]) == ["source", "channel", "offset"]


def make_fake_api(
    *,
    n_rows: int = 3,
    n_features: int = 3,
    dtype: str | np.dtype = "float16",
    hop_size: float = 5.0,
    source_base: str = "https://api.ecosounds.org",
):
    """
    Factory returning a FakeAPI with configurable shape/dtype.

    - n_rows: number of frames per ARID
    - n_features: number of feature columns (f0..f{n_features-1})
    - dtype: feature dtype ('float16'|'float32'|'float64' or numpy dtype)
    - hop_size: spacing between offsets (multiples of hop_size)
    - source_base: base URL for source
    """
    dtype = np.dtype(dtype)

    class FakeAPI:
        def __init__(self):
            self.n_rows = n_rows
            self.n_features = n_features
            self.dtype = dtype
            self.hop_size = float(hop_size)
            self.source_base = source_base.rstrip("/")

        def _make_df(self, arid: int) -> pd.DataFrame:
            # realistic meta columns
            src = f"{self.source_base}/audio_recordings/{arid}/original"
            data = {
                "source": [src] * self.n_rows,
                "channel": [0] * self.n_rows,
                "offset": [i * self.hop_size for i in range(self.n_rows)],  # multiples of hop_size
            }
            # feature columns f0..f{n_features-1}, deterministic per arid/row/feature
            # value will be the (arid + the row + the feature column) / 100
            for j in range(self.n_features):
                col = f"f{j}"
                vals = np.array(
                    [(float(arid) + i + j) / 100.0 for i in range(self.n_rows)],
                    dtype=self.dtype,
                )
                data[col] = vals
            return pd.DataFrame(data)

        def analysis_result_file(
            self,
            *,
            arid,
            analysis_job_id,
            analysis_name,
            filename=None,
            to_memory=True,
            timeout=None,
            **_,
        ):
            df = self._make_df(int(arid))
            if to_memory:
                return _parquet_bytes(df)
            if filename is None:
                raise ValueError("filename must be provided when to_memory=False")
            table = pa.Table.from_pandas(df)
            pq.write_table(table, filename)
            return str(filename)

    return FakeAPI()


def test_iter_parquet_dataframes_remote_skip():
    api = make_fake_api()
    # We’ll request [1,2,3], but skip 2
    dfs = list(iter_parquet_dataframes([1,2,3], api_client=api, prefetch=1, skip_arids_in=[2]))
    assert len(dfs) == 2
    sources = sorted(df["source"].iloc[0] for df in dfs)
    expected = sorted([
        "https://api.ecosounds.org/audio_recordings/1/original",
        "https://api.ecosounds.org/audio_recordings/3/original",
    ])
    assert sources == expected


def test_add_to_new_db_then_readd_same_files_is_idempotent(tmp_path):
    """
    Create a brand-new DB (vector dim=2), ingest two ARIDs from local parquet,
    then ingest the same directory again and ensure no duplicates are added.
    """
    # Prepare tiny parquet dataset with ARIDs 111 and 222
    data_dir = tmp_path / "parquet"
    data_dir.mkdir()
    _ = _write_parquets_with_arids(data_dir, arids=[111, 222], rows_per_file=3, emb_dim=2)

    # Create a new empty usearch DB with embedding dim=2
    db_dir = tmp_path / "db"

    db =convert_parquet(
        db_path=db_dir,
        parquet_source=str(data_dir)
    )

    num_embeddings_1 = db.count_embeddings()

    db2 =convert_parquet(
        db_path=db_dir,
        parquet_source=str(data_dir)
    )

    num_embeddings_2 = db2.count_embeddings()

    assert num_embeddings_2 == num_embeddings_1, "Re-adding identical files should not insert duplicates"


def test_add_only_new_arid_when_mixing_old_and_new(tmp_path):
    """
    Start with ARID {111, 222} then re-run with files {111, 222, 333, 444}.
    Only ARID 333 should be ingested on the second run.
    """
 

    data_dir = tmp_path / "parquet"
    data_dir.mkdir()
    _ = _write_parquets_with_arids(data_dir, arids=[111, 222], rows_per_file=3, emb_dim=2)

    db_dir = tmp_path / "db"

    db =convert_parquet(
        db_path=db_dir,
        parquet_source=str(data_dir)
    )

    num_embeddings_1 = db.count_embeddings()

    _ = _write_parquets_with_arids(data_dir, arids=[333, 444], rows_per_file=3, emb_dim=2)

    db2 =convert_parquet(
        db_path=db_dir,
        parquet_source=str(data_dir)
    )

    num_embeddings_2 = db2.count_embeddings()

    assert num_embeddings_2 == num_embeddings_1 * 2


def test_add_parquet_respects_max_count(tmp_path):
    """
    Ensure max_count prevents inserting more than the allowed number of embeddings.
    """
    data_dir = tmp_path / "parquet"
    data_dir.mkdir()
    # Two files, 3 rows each = 6 total available
    _write_parquets_with_arids(data_dir, arids=[111, 222], rows_per_file=3, emb_dim=2)

    db_dir = tmp_path / "db"
    db, model_config = create_empty_db(
        db_path=db_dir,
        parquet_source=str(data_dir),
        config=None
    )

    add_parquet_to_db(
        db=db,
        parquet_source=str(data_dir),
        dataset_name="search_set",
        max_count=4,  # stop early
        max_size=None,
        source_map_fn=lambda s: s,
        hop_size_s=5.0,
    )
    assert db.count_embeddings() == 4, "DB should not exceed max_count insertions"


def test_custom_source_map_fn_is_used(tmp_path):
    """
    Verify that source_map_fn is applied to the stored source IDs.
    """
    data_dir = tmp_path / "parquet"
    data_dir.mkdir()
    _write_parquets_with_arids(data_dir, arids=[777], rows_per_file=1, emb_dim=2)

    def mapper(s: str) -> str:
        # prove mapping happened by tagging the URL
        return f"{s}?tag=unittest"


    db_dir = tmp_path / "db"
    db =convert_parquet(
        db_path=db_dir,
        parquet_source=str(data_dir),
        source_map_fn=mapper,
    )




    # Pull all sources and check the one we inserted contains the mapped suffix
    loaded_db = sqlite_usearch_impl.SQLiteUsearchDB.create(db_dir)  # reopen to simulate fresh session
    all_sources = sqlite_usearch_impl_custom.get_all_sources(db)
    assert any(s.source_id and s.source_id.endswith("?tag=unittest") for s in all_sources), \
        "source_map_fn should be applied to stored source IDs"
    





