

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
import pyarrow.parquet as pq

from perch_hoplite.agile.convert_legacy import load_embedding_config

from recognizer_workshop.perch_hoplite_custom_modifications import sqlite_usearch_impl as sqlite_usearch_impl_custom

def convert_parquet(db_path: str,
    parquet_folder: str,
    db_type: str,
    dataset_name: str,
    parquet_filepaths: list = None,
    max_count: int = -1,
    max_size: int = None,
    source_map_fn = None):
   
    if Path(db_path).exists():
        db = add_parquet_to_existing_db(db_path, parquet_folder,
                                         dataset_name, parquet_filepaths, 
                                         max_count, max_size, source_map_fn)
    else:
        db = create_db(db_path, parquet_folder, db_type, dataset_name, 
                       parquet_filepaths, max_count, max_size, source_map_fn)
    return db


def create_db(
    db_path: str,
    parquet_folder: str,
    db_type: str,
    dataset_name: str,
    parquet_filepaths: list = None,
    max_count: int = -1,
    max_size: int = None,
    source_map_fn = None
    ):
    """Creates a hoplite database from a folder of parquet files"""

    # temp while dev
    return None

    if source_map_fn is None:
        source_map_fn = lambda x: x

    # glob all parquet files in the embeddings path.
    if parquet_filepaths is None:
        print("Collecting embeddings files...")
        parquet_filepaths = [f for f in Path(parquet_folder).rglob('*.parquet')]
        print("Found ", len(parquet_filepaths), " embeddings files.")
    
    if not parquet_filepaths:
        raise FileNotFoundError(f"No .parquet files found in '{parquet_folder}'.")

    # Peek at one parquet file to get the embedding dimension.
    schema = pq.read_schema(parquet_filepaths[0])
    emb_dim = len(schema) - 3  # Exclude 'source', 'channel', and 'offset' columns.

    # Convert embedding config to new format and insert into the DB.
    legacy_config = load_embedding_config(parquet_folder)
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

    #-- TODO: what is all this metadata doing? Might be possible to remove 
    file_id_depth = legacy_config.embed_fn_config['file_id_depth']
    audio_globs = []
    for i, glob in enumerate(legacy_config.source_file_patterns):
      base_path, file_glob = glob.split('/')[-file_id_depth - 1 :]
      if i > 0:
        partial_dataset_name = f'{dataset_name}_{i}'
      else:
        partial_dataset_name = dataset_name
      audio_globs.append(
          source_info.AudioSourceConfig(
              dataset_name=partial_dataset_name,
              base_path=base_path,
              file_glob=file_glob,
              min_audio_len_s=legacy_config.embed_fn_config.min_audio_s,
              target_sample_rate_hz=legacy_config.embed_fn_config.get(
                  'target_sample_rate_hz', -2
              ),
          )
      )

    audio_sources = source_info.AudioSources(audio_globs=tuple(audio_globs))
    db.insert_metadata('legacy_config', legacy_config)
    db.insert_metadata('audio_sources', audio_sources.to_config_dict())
    db.insert_metadata('model_config', model_config.to_config_dict())
    hop_size_s = model_config.model_config.hop_size_s

    db = add_parquet_to_db(db,
        parquet_folder=parquet_folder, dataset_name=dataset_name,
        parquet_filepaths=parquet_filepaths, max_count=max_count,
        max_size=max_size, source_map_fn=source_map_fn,
        hop_size_s=hop_size_s)
    
    return db








    # ds = pq.ParquetDataset(parquet_filepaths)
    # for fragment in tqdm.tqdm(ds.fragments):
    #     try:
    #         df = fragment.to_table().to_pandas()
    #         for i, row in df.iterrows():
    #             row_offset = np.array(row['offset'])
    #             row_source = source_map_fn(row['source'])
    #             # todo: check if it's find to hardcode to float16 or should we be getting this from the config?
    #             row_embedding = np.array(row[3:], dtype=np.float16) # Skip 'source', 'channel', and 'offset'
    #             source = interface.EmbeddingSource(dataset_name, row_source, row_offset)
    #             db.insert_embedding(row_embedding, source)
    #             if max_count > 0 and db.count_embeddings() >= max_count:
    #               break



    #     except Exception as e:
    #         print(f"Unexpected error on {fragment}: {e}")
    #         continue
        
    # db.commit()
    # num_embeddings = db.count_embeddings()
    # print('\n\nTotal embeddings : ', num_embeddings)
    # hours_equiv = num_embeddings / 60 / 60 * hop_size_s
    # print(f'\n\nHours of audio equivalent : {hours_equiv:.2f}')
        
    # return db 

def add_parquet_to_existing_db(
    db_path: str,
    parquet_folder: str,
    dataset_name: str,
    parquet_filepaths: list = None,
    max_count: int = -1,
    max_size: int = None,
    source_map_fn = None, 
    hop_size_s: float = 5.0
):
    # load the db 
    db = sqlite_usearch_impl.SQLiteUsearchDB.create(db_path)
    db_model_config = db.get_metadata('model_config')
    hop_size_s = db_model_config.model_config.hop_size_s
    db = add_parquet_to_db(
        db=db,
        parquet_folder=parquet_folder,
        dataset_name=dataset_name,
        parquet_filepaths=parquet_filepaths,
        max_count=max_count,
        max_size=max_size,
        source_map_fn=source_map_fn,
        hop_size_s=hop_size_s
    )
    return db


def add_parquet_to_db(
    db: interface.HopliteDBInterface,
    parquet_folder: str,
    dataset_name: str,
    parquet_filepaths: list = None,
    max_count: int = -1,
    max_size: int = None,
    source_map_fn = None, 
    hop_size_s: float = 5.0
):
    """Adds parquet files to an existing hoplite database."""
    
    if source_map_fn is None:
        source_map_fn = lambda x: x

    # glob all parquet files in the embeddings path.
    if parquet_filepaths is None:
        print("Collecting embeddings files...")
        parquet_filepaths = [f for f in Path(parquet_folder).rglob('*.parquet')]
        print("Found ", len(parquet_filepaths), " embeddings files.")

    
    # get all sources so we skip ones already in the db
    existing_sources = sqlite_usearch_impl_custom.get_all_sources(db)
    # extract the arid from the source where source is in the format 'https://api.ecosounds.org/audio_recordings/{arid}/original/'
    extract_arid_from_source_id = lambda source: source.split('/')[-3] if source else None
    existing_arids = [extract_arid_from_source_id(source.source_id) for source in existing_sources]

    # extract arid from parquet file names where filename is in the format /some/path/arid.parquet
    extract_arid_from_parquet = lambda filepath: Path(filepath).stem.split('_')[-1] if Path(filepath).stem else None

    new_parquet_filepaths = [f for f in parquet_filepaths if extract_arid_from_parquet(f) not in existing_arids]

    ds = pq.ParquetDataset(new_parquet_filepaths)
    for fragment in tqdm.tqdm(ds.fragments):
        try:
            df = fragment.to_table().to_pandas()
            for i, row in df.iterrows():
                row_offset = np.array(row['offset'])
                row_source = source_map_fn(row['source'])
                # todo: check if it's find to hardcode to float16 or should we be getting this from the config?
                row_embedding = np.array(row[3:], dtype=np.float16) # Skip 'source', 'channel', and 'offset'
                source = interface.EmbeddingSource(dataset_name, row_source, row_offset)
                db.insert_embedding(row_embedding, source)
                if max_count > 0 and db.count_embeddings() >= max_count:
                  break



        except Exception as e:
            print(f"Unexpected error on {fragment}: {e}")
            continue
        
    db.commit()
    num_embeddings = db.count_embeddings()
    print('\n\nTotal embeddings : ', num_embeddings)
    hours_equiv = num_embeddings / 60 / 60 * hop_size_s
    print(f'\n\nHours of audio equivalent : {hours_equiv:.2f}')

    return db
        