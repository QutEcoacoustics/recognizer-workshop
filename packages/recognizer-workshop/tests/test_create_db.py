
import io
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from recognizer_workshop.create_database import iter_parquet_dataframes

def _write_parquet(df: pd.DataFrame, path: Path):
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)

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


def _parquet_bytes(df: pd.DataFrame) -> io.BytesIO:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df), buf)
    buf.seek(0)
    return buf

class FakeAPI:
    def analysis_result_file(self, *, arid, analysis_job_id, analysis_name, filename=None, to_memory=True, timeout=None, **_):
        df = pd.DataFrame({"source": [f"s-{arid}"], "channel": [0], "offset": [0.0], "e0": [arid]})
        return _parquet_bytes(df)

def test_iter_parquet_dataframes_remote_skip():
    api = FakeAPI()
    # We’ll request [1,2,3], but skip 2
    dfs = list(iter_parquet_dataframes([1,2,3], api_client=api, prefetch=1, skip_arids_in=[2]))
    assert len(dfs) == 2
    values = sorted(df["e0"].iloc[0] for df in dfs)
    assert values == [1, 3]
