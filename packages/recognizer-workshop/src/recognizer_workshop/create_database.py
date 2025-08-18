

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
from concurrent.futures import ThreadPoolExecutor

from perch_hoplite.agile.convert_legacy import load_embedding_config

from recognizer_workshop.perch_hoplite_custom_modifications import sqlite_usearch_impl as sqlite_usearch_impl_custom
from baw_helpers import baw_api
from baw_helpers.baw_helpers import get_filter_result, simple_filter, get_random_audio_recordings
from baw_helpers.baw_utils import extract_arid_and_domain

from recognizer_workshop.embedding_config_loader import load_baw_embedding_config

PARQUET_METADATA_COLUMNS = ('source', 'channel', 'offset',)

BAW_ANALYSIS_JOB_ID = 3  
BAW_ANALYSIS_NAME = "perch-embeddings_4"

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


def create_database(
        db_path: str | Path,
        filelist_path: str | Path,
        *,
        config_folder: str | Path | None = None,
        client: baw_api.baw_api | None = None, # needed if pulling remote files
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

    convert_parquet(
        db_path=db_path,
        parquet_source=arids,
        api_client=client,    
        max_count=max_embeddings,
        config_folder=config_folder
    )



def convert_parquet(
    db_path: str,
    *,
    parquet_source: str | list | Path,
    config_folder: str | Path | None = None,
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
   
    if not (Path(db_path).exists() and any(Path(db_path).iterdir())):

        db, model_config = create_empty_db(
            db_path=db_path,
            parquet_source=parquet_source,
            config_folder=config_folder,     
            db_type=db_type,
            max_size=max_size,
            api_client=api_client,      
        )
        

    db = add_parquet_to_existing_db(db_path, **kwargs)
    return db



def iter_parquet_dataframes(
    parquet_source: str | Path | list,
    *,
    api_client=None,          # your module client; provides read_parquet(arid) etc.
    prefetch: int = 1,    # how many remote reads to queue ahead (ARID mode only)
    columns: list[str] | None = None,  # optional column projection
    skip_arids_in: list[int] | tuple[int] | set[int] | None = None,
    analysis_job_id: int = BAW_ANALYSIS_JOB_ID, # TODO: some sort of per-host defaults for different embeddings sources, or eventaully look it up on the host to see what's available
    analysis_name: str = BAW_ANALYSIS_NAME,
):
    """
    Yield pandas DataFrames from local or downloaded files

    @param: parquet_source
      - a local directory path OR single file path OR list of file paths; OR
        a list of ARIDs (remote), fetched via api_client.
    @param: api_client
      - an instance of your module client, used for fetching remote parquet files
    @param: prefetch
      - how many remote reads to queue ahead (ARID mode only). For efficiency, it will download the next few files while processing the current one.
    @param: columns
      - optional column projection
    """

  
    skip_arids_in = set() if skip_arids_in is None else {int(x) for x in skip_arids_in}

    def warn(source, exc):
        print(f"[WARN] skipping {source}: {exc}", file=sys.stderr)

    if prefetch < 0:
        raise ValueError("prefetch must be non-negative")

    if isinstance(parquet_source, int):
        parquet_source = [parquet_source]

    is_seq = isinstance(parquet_source, (list, tuple))

    # downloading parquet based on arids
    if is_seq and all(isinstance(x, int) for x in parquet_source):
        
        arids = [a for a in parquet_source if int(a) not in skip_arids_in]
        if not arids:
            return

        if api_client is None:
            raise ValueError("api_client is required when parquet_source is a list of ARIDs")
        
        def fetch_one(arid: int):
            # Get BytesIO from API (auth handled by client)
            buf = api_client.analysis_result_file(
                arid=arid,
                analysis_job_id=analysis_job_id,
                analysis_name=analysis_name,
                to_memory=True # get an io buffer rather than save to file
            )
            if buf is None:
                raise RuntimeError(f"no parquet returned for ARID {arid}")
            return pq.read_table(buf, columns=columns).to_pandas()


        # prefetch queue to download some while the current is working
        with ThreadPoolExecutor(max_workers=max(1, prefetch + 1)) as executor:
            q: deque = deque()
            it = iter(arids)

            def submit_next():
                try:
                    cur_arid = next(it)
                except StopIteration:
                    return False
                q.append((cur_arid, executor.submit(fetch_one, cur_arid)))
                return True


            # fill initial queue
            for _ in range(min(prefetch + 1, len(arids))):
                if not submit_next():
                    break

            pbar = tqdm.tqdm(total=len(arids), desc="remote")
            while q:
                arid, fut = q.popleft()
                submit_next()
                try:
                    yield fut.result()
                except Exception as e:
                    warn(arid, e)
                pbar.update(1)
            pbar.close()

        return
    
    else:
        # using local files

        dataset = ds.dataset(parquet_source, format="parquet")

        # Iterate FRAGMENTS to know which file we’re reading
        for frag in tqdm.tqdm(dataset.get_fragments(), desc="local-fragments"):
            # Many fragment types have a .path; if not, you can pull from metadata or skip
            path = getattr(frag, "path", None)
            try:
                _, arid_from_name = extract_arid_and_domain(path)
            except ValueError as e:
                print(f"Can't determine arid of {path}: {e}", file=sys.stderr)
                arid_from_name = None

            if arid_from_name is not None and arid_from_name in skip_arids_in:
                print(f"Skipping existing ARID: {arid_from_name} ({path})")
                continue

            # Now actually read this fragment
            try:
                table = frag.to_table(columns=columns)
                yield table.to_pandas()
            except Exception as e:
                warn(path or "<fragment>", e)


def create_empty_db(
    db_path: str,
    parquet_source: str | Path | list,
    config_folder: str | Path | None = None,
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

    # Convert embedding config to new format and insert into the DB.
    if config_folder is not None:
        legacy_config = load_embedding_config(config_folder)
    else:
        legacy_config = load_baw_embedding_config(BAW_ANALYSIS_NAME)

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

    # #-- TODO: what is all this metadata doing? Might be possible to remove 
    # file_id_depth = legacy_config.embed_fn_config['file_id_depth']
    # audio_globs = []
    # for i, glob in enumerate(legacy_config.source_file_patterns):
    #   base_path, file_glob = glob.split('/')[-file_id_depth - 1 :]
    #   if i > 0:
    #     partial_dataset_name = f'{dataset_name}_{i}'
    #   else:
    #     partial_dataset_name = dataset_name
    #   audio_globs.append(
    #       source_info.AudioSourceConfig(
    #           dataset_name=partial_dataset_name,
    #           base_path=base_path,
    #           file_glob=file_glob,
    #           min_audio_len_s=legacy_config.embed_fn_config.min_audio_s,
    #           target_sample_rate_hz=legacy_config.embed_fn_config.get(
    #               'target_sample_rate_hz', -2
    #           ),
    #       )
    #   )

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
    def extract(source_id: str | None) -> int | None:
        try:
            return int(source_id.split("/")[-3]) if source_id else None
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

    total_inserted = 0


    for df in tqdm.tqdm(iter_parquet_dataframes(parquet_source, api_client=api_client, skip_arids_in=existing_arids), desc="ingest"):
        max_for_df = max_count - total_inserted if max_count > 0 else -1
        num_inserted = add_df_to_db(df, db, source_map_fn=source_map_fn, max_count=max_for_df, dataset_name=dataset_name)
        total_inserted += num_inserted
        if max_count > 0 and db.count_embeddings() >= max_count:
            break
        
    db.commit()
    num_embeddings = db.count_embeddings()
    print('\n\nTotal embeddings : ', num_embeddings)
    hours_equiv = num_embeddings / 60 / 60 * hop_size_s
    print(f'\n\nHours of audio equivalent : {hours_equiv:.2f}')

    return db


def add_df_to_db(df, db, source_map_fn=convert_source_to_original_url, max_count=-1, dataset_name='search_set'):
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
        if max_count > 0 and db.count_embeddings() >= max_count:
            break
    return num_inserted

