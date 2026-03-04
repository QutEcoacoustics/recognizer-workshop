import logging
import sys
from pathlib import Path
from etils import epath
from ml_collections import config_dict
import numpy as np
from perch_hoplite.agile import embed
from perch_hoplite.agile import source_info
from perch_hoplite.db import db_loader
from perch_hoplite.db import in_mem_impl
from perch_hoplite.db import interface
from perch_hoplite.db import sqlite_usearch_impl
import tensorflow as tf
import tqdm
from collections import deque
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow.compute as pc
from concurrent.futures import ThreadPoolExecutor

from perch_hoplite.agile.convert_legacy import load_embedding_config

from recognizer_workshop.perch_hoplite_custom_modifications import sqlite_usearch_impl as sqlite_usearch_impl_custom
from baw_helpers import baw_api
from baw_helpers.baw_api import AuthenticationError
from baw_helpers.baw_helpers import simple_filter, get_random_audio_recordings
from baw_helpers.baw_utils import extract_arid_and_domain, original_baw_audio_url_from_arid, get_original_url, extract_arid_from_filename

from recognizer_workshop.app.create_database.embedding_config_loader import load_baw_embedding_config

    
log = logging.getLogger(__name__)

PARQUET_METADATA_COLUMNS = ('source', 'channel', 'offset',)

BAW_ANALYSIS_JOB_ID = 3
DEFAULT_BAW_ANALYSIS_NAME = "perch-embeddings_4"


def convert_source_to_original_url(source: str, domain='https://api.ecosounds.org') -> str:
    """
    perch runner currently puts the canonical filename in the source column. We really want the url to the original audio file
    TODO: I don't think this is true anymore
    TODO: move this to baw utils with the rest of those functions
    """
    return get_original_url(source, default_domain=domain)


def create_database_from_baw(
        db_path: str | Path,
        filelist_path: str | Path,
        *,
        config: str | Path | None = None,
        client: baw_api.baw_api | None = None, 
        project_ids: int | list | None = None,
        region_ids: int | list | None = None,
        site_ids: int | list | None = None,
        filter: dict | None = None,
        max_embeddings: int = 1_000_000,
        max_files: int = -1
):
    
    filter_params = simple_filter(
        project_ids=project_ids,
        region_ids=region_ids,
        site_ids=site_ids,
        filter_params=filter
    )

    # get a big list of arids and save json to the given path
    if client is None:
        client = baw_api.baw_api()

    audio_recordings, _ = get_random_audio_recordings(
        api=client,
        filter_params=filter_params,
        save_dir=filelist_path,
        limit=max_files
    )

    arids = [int(ar['id']) for ar in audio_recordings]

    db = convert_parquet(
        db_path=db_path,
        parquet_source=arids,
        api_client=client,    
        max_count=max_embeddings,
        config=config
    )

    # create a baw filters metadata entry. It's a list of filter params dicts, so we can define multiple filters if needed
    # it must be wrapped in a ConfigDict to be storable in the db
    if filter_params is not None:
        db.insert_metadata('baw_filters', config_dict.ConfigDict({"filters": [filter_params['filter']]}))
        db.commit()


def convert_parquet(
    db_path: str | Path,
    *,
    parquet_source: str | list | Path,
    config: str | Path | None = None,
    db_type: str = 'sqlite_usearch',
    dataset_name: str = 'search_set',
    max_count: int = -1,
    max_size: int | None = None,
    source_map_fn = convert_source_to_original_url,
    api_client = None
    ):
 
    # put all kwargs in a dict
    kwargs = {
        "parquet_source": parquet_source,
        "api_client": api_client,
        "db_type": db_type,
        "dataset_name": dataset_name,
        "max_count": max_count,
        "max_size": max_size,
        "source_map_fn": source_map_fn
    }
    db_path = Path(db_path).expanduser()
    has_sqlite = db_path.exists() and any(db_path.glob("*.sqlite"))
    if not has_sqlite:
        db, model_config = create_empty_db(
            db_path=db_path,
            parquet_source=parquet_source,
            config=config,     
            db_type=db_type,
            max_size=max_size,
            api_client=api_client,      
        )
        

    db = add_parquet_to_existing_db(db_path, **kwargs)
    return db


def remove_existing_offsets_from_item(db, item: dict) -> dict:
    """Given an item dict with 'id' and 'offsets', remove offsets that already exist in the db."""
    offsets_in_db = db.get_offsets_for_source_id(item['source_url'])
    if 'offsets' in item:
        item['offsets'] = [o for o in item['offsets'] if o not in offsets_in_db]
    return item


def add_all_offsets_for_duration(item: dict, hop_size: float = 5.0) -> dict:
    total_offsets = int(item['duration_seconds'] // hop_size)
    item['offsets'] = [o for o in range(total_offsets)]
    return item


def remove_existing_offsets_from_df(db, df) -> 'pd.DataFrame':
    """Given a dataframe with 'source' and 'offset' columns, remove rows that already exist in the db."""
    if df.empty:
        return df
    source_id = df['source'].iloc[0] # all rows should have the same source
    offsets_in_db = db.get_offsets_for_source_id(source_id)
    if offsets_in_db:
        mask = ~df['offset'].isin(offsets_in_db)
        df = df[mask]
    return df


def complete_remote_source_dict(item: dict, default_domain = "https://api.ecosounds.org") -> dict:

    input_id = item.get('id')
    input_domain = item.get('domain')
    input_url = item.get('source_url')
    extracted_domain = None
    extracted_id = None
    
    if input_url:
        extracted_domain, extracted_id = extract_arid_and_domain(input_url)
        # Ensure extracted_id is an integer for consistent comparison
        if extracted_id is not None:
            extracted_id = int(extracted_id)

    found_ids = {x for x in [input_id, extracted_id] if x is not None}

    if len(found_ids) > 1:
        raise ValueError(f"Conflicting ID information in {item}")
    if not found_ids:
        raise ValueError(f"Cannot resolve ID in item: {item}")
    
    final_id = found_ids.pop()

    found_domains = {x for x in [input_domain, extracted_domain] if x is not None}

    if len(found_domains) > 1:
        raise ValueError(f"Conflicting Domain information in {item}")
    
    final_domain = found_domains.pop() if found_domains else default_domain

    generated_url = original_baw_audio_url_from_arid(final_id, final_domain)

    if input_url and input_url != generated_url:
        raise ValueError(f"Conflicting source_url in {item}")

    result = item.copy()
    result.update({
        'id': final_id,
        'source_url': generated_url,
        'domain': final_domain
    })
    return result


def normalize_source(source: int | list[int] | list[dict]):
    """
    Normalizes input to a list of dicts: [{'id': 123}, ...]
    
    For local parquet files, this can be:
      - parent directory (Path or str)
      - list of parquet file paths (Path or str)
    Or, for remote baw parquet files, this can be
      - single int ARID
      - list of int ARIDs
      - list of dicts with at least 'id' or 'source_url' key
      - single str URL
      - list of str URLs

    Returns:
        tuple: (normalized_source, source_type: "local" | "remote")
        for source_type "local", normalized_source is a Path or list of Paths
        for source_type "remote", normalized_source is a list of dicts with 'id' and 'source_url' keys
    """

    def is_url(s):
        return str(s).startswith("http://") or str(s).startswith("https://")
    
    def url_to_item_dict(url: str) -> dict:
        domain, arid = extract_arid_and_domain(url)
        if arid is None:
             raise ValueError(f"Could not extract ARID from URL: {url}")
        return {'id': int(arid),
                'source_url': url,
                'domain': domain
                }

    if isinstance(source, (int, np.integer)):
        source = [int(source)]
    elif isinstance(source, str):
        # detect if it's a remote source string (e.g. https://api.ecosounds.org/audio_recordings/123/original)
        if is_url(source):
            source = [url_to_item_dict(source)]            
        else:
            source = Path(source)

    if isinstance(source, Path):
        if not source.exists():
            raise FileNotFoundError(f"Local parquet source path does not exist: {source}")
        else:
            return source, "local"

    # by here it will be a list or tuple
    if all(isinstance(x, (int, float, np.integer)) for x in source):
        source = [{'id': (int(x))} for x in source]
    elif all(is_url(x) for x in source):
        source = [url_to_item_dict(x) for x in source]
    elif all(isinstance(x, (str, Path)) for x in source):
        return [Path(x) for x in source], "local"

    source = [complete_remote_source_dict(x) for x in source]
    
    return source, "remote"


def iter_parquet_dataframes(
    parquet_source: str | Path | list,
    *,
    api_client=None,
    prefetch: int = 1,
    columns: list[str] | None = None,
    analysis_job_id: int = BAW_ANALYSIS_JOB_ID, 
    analysis_name: str = DEFAULT_BAW_ANALYSIS_NAME,
    max_auth_errors: int = 3,
    db = None,
    hop_size_s: float = 5.0,
):
    """
    Master iterator that normalizes input and dispatches to local or remote iterators.
    """
    if prefetch < 0:
        raise ValueError("prefetch must be non-negative")

    parquet_source, source_type = normalize_source(parquet_source)

    if source_type == "remote":
        yield from _iter_remote_parquet_dataframes(
            parquet_source,
            api_client=api_client,
            db=db,
            hop_size_s=hop_size_s,
            prefetch=prefetch,
            columns=columns,
            analysis_job_id=analysis_job_id,
            analysis_name=analysis_name,
            max_auth_errors=max_auth_errors
        )
    else:
        existing_arids = get_existing_arids(db) if db is not None else []

        yield from _iter_local_parquet_dataframes(
            parquet_source,
            columns=columns,
            skip_arids_in=existing_arids,
        )


def _iter_remote_parquet_dataframes(
    items,
    *,
    api_client,
    prefetch: int = 1,
    columns: list[str] | None = None,
    analysis_job_id: int = BAW_ANALYSIS_JOB_ID,
    analysis_name: str = DEFAULT_BAW_ANALYSIS_NAME,
    max_auth_errors: int = 3,
    db = None,
    hop_size_s: float = 5.0
):
    """
    Yield pandas DataFrames from remote parquet files specified by items list

    @param: items
      - a list of dict in the form { id: 123, offsets: [offset1, offset2] } fetched via api_client
    @param: api_client
      - an instance of your module client, used for fetching remote parquet files
    @param: prefetch
      - how many remote reads to queue ahead. For efficiency, it will download the next few files while processing the current one.
    @param: columns
      - optional column projection
    @param: max_auth_errors
      - maximum number of consecutive 401 authentication errors before aborting
    @hop_size_s:
        - hop size in seconds, used when generating offsets for an item with duration_seconds but no offsets
    """

    if not items:
        return
    
    if api_client is None:
        raise ValueError("api_client is required when parquet_source is a list of ARIDs")
    
    log.info("Starting download of %d parquet files from API", len(items))
    
    def fetch_one(item: dict):
        log.debug("Fetching parquet for ARID %d", item['id'])

        buf = api_client.analysis_result_file(
            arid=item['id'],
            analysis_job_id=analysis_job_id,
            analysis_name=analysis_name,
            to_memory=True # get an io buffer rather than save to file
        )
        if buf is None:
            raise RuntimeError(f"no parquet returned for ARID {item['id']}")
        log.debug("Successfully fetched parquet for ARID %d size: %d bytes", item['id'], len(buf.getvalue()))
        table = pq.read_table(buf, columns=columns)
        df = table.to_pandas()

        return df

    # prefetch queue to download some while the current is working
    with ThreadPoolExecutor(max_workers=max(1, prefetch + 1)) as executor:
        q: deque = deque()
        it = iter(items)

        def submit_next():
            """
            keep taking the next item, checking if it has offsets that need to be fetched
            and if so, submit it to the executor and return, else keep trying
            """

            try:
                while True:
                    cur_item = next(it)

                    if 'offsets' in cur_item and db:
                        cur_item = remove_existing_offsets_from_item(db, cur_item) # if offsets are defined in an item, remove existing ones
                    elif 'duration_seconds' in cur_item and db:
                        cur_item = add_all_offsets_for_duration(cur_item, hop_size=hop_size_s) # if no offsets are defined, but duration is, generate all possible offsets and remove existing ones
                        cur_item = remove_existing_offsets_from_item(db, cur_item)

                    # if there are no offsets left after removal, skip fetching and return None
                    if 'offsets' in cur_item and not cur_item['offsets']:
                        log.info("No offsets left to fetch for ARID %d after removing existing offsets, skipping", cur_item['id'])
                        continue
                    
                    q.append((cur_item, executor.submit(fetch_one, cur_item)))
                    return True

            except StopIteration:
                return False

        # fill initial queue
        for _ in range(min(prefetch + 1, len(items))):
            if not submit_next():
                break

        pbar = tqdm.tqdm(total=len(items), desc="remote")
        files_processed = 0
        consecutive_auth_errors = 0
        while q:
            item, fut = q.popleft()
            submit_next()
            try:
                result_df = fut.result()

                offsets = item.get('offsets')
                if offsets:
                    # filter dataframe to only requested offsets
                    # the item offsets were modified earlier to remove existing ones
                    result_df = result_df[result_df['offset'].isin(offsets)]
                elif db is not None:
                    # there might be the case that no offsets were specified (meaning add the whole file), 
                    # and no duration was available (meaning we couldn't calculate non-existing offsets), 
                    # so we need to filter here after the parquet is downloaded
                    result_df = remove_existing_offsets_from_df(db, result_df)

                consecutive_auth_errors = 0
                files_processed += 1
                log.info("Processed parquet %d/%d (ARID: %d, rows: %d)", files_processed, len(items), item['id'], len(result_df))
                yield result_df
            except AuthenticationError as e:
                consecutive_auth_errors += 1
                log.error("Authentication error (%d/%d) for ARID %d: %s", consecutive_auth_errors, max_auth_errors, item['id'], e)
                
                if consecutive_auth_errors >= max_auth_errors:
                    log.error("Aborting: %d consecutive authentication errors. Check your API token.", max_auth_errors)
                    pbar.close()
                    raise RuntimeError(f"Aborted after {max_auth_errors} consecutive authentication errors")
            except Exception as e:
                consecutive_auth_errors = 0  # Reset counter for non-auth errors
                log.error("Error processing item %s: %s", item, e)
                
            pbar.update(1)
        pbar.close()
        
    log.info("Completed downloading %d parquet files", files_processed)
    return


def _iter_local_parquet_dataframes(parquet_source, *, columns: list[str] | None = None, skip_arids_in: set| list | None = None):

    # using local files
    # we don't support dict of offsets here because it's more complicated and we dont' need it for local files right now

    dataset = ds.dataset(parquet_source, format="parquet")

    skip_arids_in = set(skip_arids_in) if skip_arids_in is not None else set()

    for frag in tqdm.tqdm(dataset.get_fragments(), desc="local-fragments"):
        path = getattr(frag, "path", None)
        try:
            arid_from_name = extract_arid_from_filename(path)
        except ValueError as e:
            log.warning("Can't determine arid of %s: %s", path, e)
            arid_from_name = None

        if arid_from_name is not None and arid_from_name in skip_arids_in:
            log.info("Skipping existing ARID: %s (%s)", arid_from_name, path)
            continue

        try:
            table = frag.to_table(columns=columns)
            yield table.to_pandas()
        except Exception as e:
            logging.warning(path or "<fragment>", e)


def _load_embedding_config(config: str | Path | dict | None):
    """
    Accepts:
      - None -> use DEFAULT_BAW_ANALYSIS_NAME via load_baw_embedding_config
      - dict/ConfigDict -> normalize & return ConfigDict
      - Path -> must exist; load via load_embedding_config
      - str -> if path exists load it, else treat as named config via load_baw_embedding_config
    Raises:
      - TypeError for unsupported types
      - FileNotFoundError for non-existent Path
    """
    if config is None:
        config = DEFAULT_BAW_ANALYSIS_NAME

    if isinstance(config, (dict, config_dict.ConfigDict)):
        return config_dict.ConfigDict(config)

    if isinstance(config, Path):
        if config.exists():
            return config_dict.ConfigDict(load_embedding_config(config))
        raise FileNotFoundError(f"Config path does not exist: {config}")

    if isinstance(config, str):
        p = Path(config)
        if p.exists():
            return config_dict.ConfigDict(load_embedding_config(p))
        return config_dict.ConfigDict(load_baw_embedding_config(config))

    raise TypeError(f"Unsupported config type: {type(config).__name__}")


def create_empty_db(    
    db_path: str,
    parquet_source: str | Path | list,
    config: str | Path | None = None,
    db_type: str = 'sqlite_usearch',
    max_size: int | None = 1_000_000,
    api_client=None
    ):
    """Creates a hoplite database from a folder of parquet files"""

    # Peek at one parquet file to get the embedding dimension.
    first_item = next(iter_parquet_dataframes(parquet_source, api_client=api_client, prefetch=0), None)
    if first_item is None:
        raise ValueError("No parquet dataframes available to infer embedding dimension.")
    meta_cols = [c for c in first_item.columns if c in PARQUET_METADATA_COLUMNS]
    emb_dim = first_item.shape[1] - len(meta_cols)
    if emb_dim <= 0:
        raise ValueError(f"Could not infer embedding dim (got {emb_dim}).")
    
    legacy_config = _load_embedding_config(config)

    model_config = embed.ModelConfig(
        model_key=legacy_config.embed_fn_config.model_key,
        embedding_dim=emb_dim,
        model_config=legacy_config.embed_fn_config.model_config,
    )

    if db_type == 'sqlite_usearch':
        if (Path(db_path) / sqlite_usearch_impl_custom.HOPLITE_FILENAME).exists():
            raise ValueError(f'DB path {db_path} already exists.')
        # don't use db_loader because it doesn't use the custom sqlite_usearch_impl
        #db = db_loader.create_new_usearch_db(db_path, emb_dim)
        usearch_cfg = sqlite_usearch_impl.get_default_usearch_config(emb_dim)
        db = sqlite_usearch_impl_custom.ThreadedSQLiteUsearchDB.create(
                    db_path, 
                    usearch_cfg=usearch_cfg
                )
    elif db_type == 'in_mem':
      db = in_mem_impl.InMemoryGraphSearchDB.create(
          embedding_dim=emb_dim,
          max_size=max_size,
      )
    else:
      raise ValueError(f'Unknown db type: {db_type}')

    # I think this is legacy stuff that is only applicable to local files, not baw stuff
    # Not sure if it's a good idea to get rid of it, so make some empty audio sources
    audio_sources = source_info.AudioSources(audio_globs=tuple())
    db.insert_metadata('legacy_config', legacy_config)
    db.insert_metadata('audio_sources', audio_sources.to_config_dict())
    db.insert_metadata('model_config', model_config.to_config_dict())
    db.commit()
    
    return db, model_config

 
def add_parquet_to_existing_db(
    db_path: str,
    parquet_source: str | Path | list,
    *,
    api_client: baw_api.baw_api | None = None,
    db_type: str = 'sqlite_usearch',
    dataset_name: str = 'search_set',
    max_count: int = -1,
    max_size: int | None = None,
    source_map_fn = convert_source_to_original_url
):
    
    if db_type == 'sqlite_usearch':
        db = sqlite_usearch_impl_custom.ThreadedSQLiteUsearchDB.create(db_path)
    else:
        raise ValueError(f'Unsupported db type: {db_type}')

    db_model_config = db.get_metadata('model_config')
    hop_size_s = db_model_config.model_config.hop_size_s
    db = add_parquet_to_db(
        db=db,
        parquet_source=parquet_source,
        api_client=api_client,
        dataset_name=dataset_name,
        max_count=max_count,
        max_size=max_size,
        source_map_fn=source_map_fn,
        hop_size_s=hop_size_s
    )

    return db


def get_existing_arids(db) -> set[int]:
    """Get a list of all ARIDs in the db by extracting from source IDs."""
    existing_sources = sqlite_usearch_impl_custom.get_all_sources(db)
    def extract(source_id: str) -> int | None:
        try:
            _, arid = extract_arid_and_domain(source_id)
            return int(arid) if arid else None
        except Exception:
            return None
    return {a for a in (extract(s.source_id) for s in existing_sources) if a is not None}


def add_parquet_to_db(
    db: interface.HopliteDBInterface,
    parquet_source: str | Path | list,
    api_client: baw_api.baw_api | None = None,
    dataset_name: str = 'search_set',
    max_count: int = -1,
    max_size: int = None,
    source_map_fn = convert_source_to_original_url, 
    hop_size_s: float = 5.0
):
    """Adds parquet files to an existing hoplite database."""
    
    # if the source column values in the parquet files need to be converted, e.g. from friendly filename to original recording url
    if source_map_fn is None:
        source_map_fn = lambda x: x

    total_inserted = 0
    files_processed = 0

    iterator = iter_parquet_dataframes(
        parquet_source,
        api_client=api_client,
        db=db, 
        hop_size_s=hop_size_s
    )

    log.info("Starting to ingest parquet files into database")
    for df in tqdm.tqdm(iterator, desc="ingest"): 
        max_for_df = max_count - total_inserted if max_count > 0 else -1
        num_inserted = add_df_to_db(df, db, source_map_fn=source_map_fn, max_for_df=max_for_df, dataset_name=dataset_name)
        total_inserted += num_inserted
        files_processed += 1
        
        log.info(f"Ingested file {files_processed}: added {num_inserted} embeddings (total: {total_inserted})")
        
        if max_count > 0 and db.count_embeddings() >= max_count:
            log.info(f"Reached max_count limit of {max_count}, stopping ingestion")
            break
        
    db.commit()
    num_embeddings = db.count_embeddings()
    log.info(f'Total embeddings: {num_embeddings}')
    hours_equiv = num_embeddings / 60 / 60 * hop_size_s
    log.info(f'Hours of audio equivalent: {hours_equiv:.2f}')

    return db


def add_df_to_db(df, db, source_map_fn=convert_source_to_original_url, max_for_df=-1, dataset_name='search_set'):
    if source_map_fn is None:
        source_map_fn = lambda x: x
    num_inserted = 0
    for i, row in df.iterrows():
        row_offset = np.array(row['offset'])
        row_source = source_map_fn(row['source'])
        # todo: check if it's fine to hardcode to float16 or should we be getting this from the config?
        row_embedding = np.array(row[3:], dtype=np.float16) # Skip 'source', 'channel', and 'offset'
        source = interface.EmbeddingSource(dataset_name, row_source, row_offset)
        db.insert_embedding(row_embedding, source)
        num_inserted += 1
        if max_for_df > 0 and num_inserted >= max_for_df:
            break
    return num_inserted
