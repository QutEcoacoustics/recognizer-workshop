

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



def convert_parquet(
    parquet_folder: str,
    db_type: str,
    dataset_name: str,
    parquet_filepaths: list = None,
    max_count: int = -1,
    db_path: str = './db',
    max_size: int = None,
    source_map_fn = lambda x: x,
    ):
    """Creates a hoplite database from a folder of parquet files"""

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

    ds = pq.ParquetDataset(parquet_filepaths)
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

