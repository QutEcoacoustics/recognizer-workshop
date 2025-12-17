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
from baw_helpers.baw_helpers import get_filter_result, simple_filter, get_random_audio_recordings
from baw_helpers.baw_utils import extract_arid_and_domain

from recognizer_workshop.app.create_database.embedding_config_loader import load_baw_embedding_config

    
log = logging.getLogger(__name__)

PARQUET_METADATA_COLUMNS = ('source', 'channel', 'offset',)

#TODO: don't hardcode these
BAW_ANALYSIS_JOB_ID = 3
DEFAULT_BAW_ANALYSIS_NAME = "perch-embeddings_4"

def convert_source_to_original_url(source: str, domain='https://api.ecosounds.org') -> str:
    """
    perch runner currently puts the canonical filename in the source column. We really want the url to the audio file
    TODO: move this to baw utils with the rest of those functions
    """
    extracted_domain, arid = extract_arid_and_domain(source)
    if extracted_domain is None:
        return f"{domain}/audio_recordings/{arid}/original"
    else:
        # if we extracted the domain, it's probably already how we want it
        return source


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



def get_parquet_offset_rows(df, offsets) #-> pd.DataFrame
    """Get rows from a parquet dataframe that match the given offsets (in seconds)"""
    if offsets:
        return df[df['offset'].isin(offsets)]
    return df

def iter_parquet_dataframes(
    parquet_source: str | Path | list,
    *,
    api_client=None,
    prefetch: int = 1,
    columns: list[str] | None = None,
    skip_arids_in: list[int] | tuple[int] | set[int] | None = None,
    analysis_job_id: int = BAW_ANALYSIS_JOB_ID, # TODO: some sort of per-host defaults for different embeddings sources, or eventaully look it up on the host to see what's available
    analysis_name: str = DEFAULT_BAW_ANALYSIS_NAME,
    max_auth_errors: int = 3
):
    """
    Yield pandas DataFrames from local or downloaded files

    @param: parquet_source
      - (local) a local directory path OR single file path OR list of file paths; OR
      - (remote) a list of ARIDs or list of dict in the form { id: 123, offsets: [offset1, offset2] } fetched via api_client
      - must be either remote or local, not a mix
    @param: api_client
      - an instance of your module client, used for fetching remote parquet files
    @param: prefetch
      - how many remote reads to queue ahead (ARID mode only). For efficiency, it will download the next few files while processing the current one.
    @param: columns
      - optional column projection
    @param: max_auth_errors
      - maximum number of consecutive 401 authentication errors before aborting
    """

    skip_arids_in = set() if skip_arids_in is None else {int(x) for x in skip_arids_in}
    consecutive_auth_errors = 0  # so we can abort after too many auth errors

    def warn(source, exc):
        log.warning("skipping %s: %s", source, exc)

    if prefetch < 0:
        raise ValueError("prefetch must be non-negative")

    if isinstance(parquet_source, int):
        parquet_source = [parquet_source]
    
    # convert list format to dict format to normalize
    is_seq = isinstance(parquet_source, (list, tuple))

    # convert list of ints to list of dicts for normalised format
    if is_seq and all(isinstance(x, int) for x in parquet_source):
        parquet_source = [{'id': x} for x in parquet_source]

    is_remote = is_seq and all(isinstance(x, dict) for x in parquet_source)

    # just check that they have not tried passing a dict of local file paths and offsets
    # we don't support that because of the way we read local files via pyarrow datasets
    if is_remote:
        for item in parquet_source:
            if 'id' not in item:
                raise ValueError("When passing a dict list for parquet source, each item must have an 'id' of a remote audio recording. Local file paths with offsets are not supported.")

    # downloading parquet based on arids
    if is_remote:
        
        items = [item for item in parquet_source if int(item['id']) not in skip_arids_in]
        if not items:
            return

        if api_client is None:
            raise ValueError("api_client is required when parquet_source is a list of ARIDs")
        
        log.info("Starting download of %d parquet files from API", len(items))
        
        def fetch_one(item: dict):
            # Get BytesIO from API (auth handled by client)
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

            # 3. Filter inside the thread (Efficient)
            offsets = item.get('offsets')
            if offsets:
                # Standard Pandas filtering - works on all versions
                df = df[df['offset'].isin(offsets)]
            
            return df


        # prefetch queue to download some while the current is working
        with ThreadPoolExecutor(max_workers=max(1, prefetch + 1)) as executor:
            q: deque = deque()
            it = iter(items)

            def submit_next():
                try:
                    cur_item = next(it)
                except StopIteration:
                    return False
                q.append((cur_item, executor.submit(fetch_one, cur_item)))
                return True


            # fill initial queue
            for _ in range(min(prefetch + 1, len(items))):
                if not submit_next():
                    break

            pbar = tqdm.tqdm(total=len(items), desc="remote")
            files_processed = 0
            while q:
                item, fut = q.popleft()
                submit_next()
                try:
                    result = fut.result()
                    consecutive_auth_errors = 0
                    files_processed += 1
                    log.info("Processed parquet %d/%d (ARID: %d, rows: %d)", files_processed, len(items), item['id'], len(result))
                    yield result
                except AuthenticationError as e:
                    # Direct authentication error from the API client
                    consecutive_auth_errors += 1
                    log.error("Authentication error (%d/%d) for ARID %d: %s", consecutive_auth_errors, max_auth_errors, item['id'], e)
                    
                    if consecutive_auth_errors >= max_auth_errors:
                        log.error("Aborting: %d consecutive authentication errors. Check your API token.", max_auth_errors)
                        pbar.close()
                        raise RuntimeError(f"Aborted after {max_auth_errors} consecutive authentication errors")
                except Exception as e:
                    # Check if it's an HTTP 401 auth error by inspecting the exception message
                    error_msg = str(e).lower()
                    is_http_auth_error = "401" in error_msg or "unauthorized" in error_msg
                    
                    if is_http_auth_error:
                        consecutive_auth_errors += 1
                        log.error("Authentication error (%d/%d) for ARID %d: %s", consecutive_auth_errors, max_auth_errors, item['id'], e)
                        
                        if consecutive_auth_errors >= max_auth_errors:
                            log.error("Aborting: %d consecutive authentication errors. Check your API token.", max_auth_errors)
                            pbar.close()
                            raise RuntimeError(f"Aborted after {max_auth_errors} consecutive authentication errors")
                    else:
                        consecutive_auth_errors = 0  # Reset counter for non-auth errors
                        warn(item, e)
                    
                pbar.update(1)
            pbar.close()
            
        log.info("Completed downloading %d parquet files", files_processed)
        return
    
    else:
        # using local files
        # we don't support dict of offsets here, because the parquet_source can be a directory of files

        dataset = ds.dataset(parquet_source, format="parquet")

        # Iterate FRAGMENTS to know which file we’re reading
        for frag in tqdm.tqdm(dataset.get_fragments(), desc="local-fragments"):
            # Many fragment types have a .path; if not, you can pull from metadata or skip
            path = getattr(frag, "path", None)
            try:
                _, arid_from_name = extract_arid_and_domain(path)
            except ValueError as e:
                log.warning("Can't determine arid of %s: %s", path, e)
                arid_from_name = None

            if arid_from_name is not None and arid_from_name in skip_arids_in:
                log.info("Skipping existing ARID: %s (%s)", arid_from_name, path)
                continue

            # Now actually read this fragment
            try:
                table = frag.to_table(columns=columns)
                yield table.to_pandas()
            except Exception as e:
                warn(path or "<fragment>", e)


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

    # Already a mapping: normalize and return
    if isinstance(config, (dict, config_dict.ConfigDict)):
        return config_dict.ConfigDict(config)

    # Path-like
    if isinstance(config, Path):
        if config.exists():
            return config_dict.ConfigDict(load_embedding_config(config))
        raise FileNotFoundError(f"Config path does not exist: {config}")

    # String: maybe a path, else named default
    if isinstance(config, str):
        p = Path(config)
        if p.exists():
            return config_dict.ConfigDict(load_embedding_config(p))
        # Treat as named config (will raise inside if unknown)
        return config_dict.ConfigDict(load_baw_embedding_config(config))

    # Anything else: bad type
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
        if (Path(db_path) / sqlite_usearch_impl.HOPLITE_FILENAME).exists():
            raise ValueError(f'DB path {db_path} already exists.')
        db = db_loader.create_new_usearch_db(db_path, emb_dim)
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
    
    #raise NotImplementedError("Adding to existing db is not implemented yet. To create a new database, delete the old one")
    # load the db 
    if db_type == 'sqlite_usearch':
        db = sqlite_usearch_impl.SQLiteUsearchDB.create(db_path)
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

    existing_arids = get_existing_arids(db)
    log.info("Database currently has %d existing ARIDs, will skip these", len(existing_arids))

    total_inserted = 0
    files_processed = 0

    log.info("Starting to ingest parquet files into database")
    for df in tqdm.tqdm(iter_parquet_dataframes(parquet_source, api_client=api_client, skip_arids_in=existing_arids), desc="ingest"):
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
        # todo: check if it's find to hardcode to float16 or should we be getting this from the config?
        row_embedding = np.array(row[3:], dtype=np.float16) # Skip 'source', 'channel', and 'offset'
        source = interface.EmbeddingSource(dataset_name, row_source, row_offset)
        db.insert_embedding(row_embedding, source)
        num_inserted += 1
        if max_for_df > 0 and num_inserted >= max_for_df:
            break
    return num_inserted

