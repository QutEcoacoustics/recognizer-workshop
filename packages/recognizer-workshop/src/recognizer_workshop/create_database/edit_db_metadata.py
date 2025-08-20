"""
edits the metadata of the database.

"""

import os
from ml_collections import config_dict
from perch_hoplite.db import sqlite_usearch_impl


def edit_tfhub_version(db_path, tfhub_version):
    """
    Edits the tfhub version in the database metadata.

    Perch version 4 is compatible with version 8, however at some point version 4 stopped being available on tf_hub. 
    Any database with perch embedding model version 4 in its metadata will crash on loading when perch-hoplite tries to fetch the embedding model from 
    tf_hub (which it does automatically, even though we don't necessarily need to embed anything new). The simplest fix for this is to just 
    change the metadata to have version 8, which is compatible with version 4 of the model.
    """

    db = sqlite_usearch_impl.SQLiteUsearchDB.create(db_path)
    model_config = db.get_metadata('model_config')
    model_config['model_config']['tfhub_version'] = tfhub_version

    # insert is an upsert
    db.insert_metadata('model_config', model_config)
    db.commit()

  
# just for reference
# example_config = {
#   "model_key": "taxonomy_model_tf", 
#   "embedding_dim": 1280, 
#   "model_config": 
#           {
#             "window_size_s": 5.0, 
#             "hop_size_s": 5.0, 
#             "sample_rate": 32000, 
#             "tfhub_version": 4, 
#             "model_path": ""}
#   }



