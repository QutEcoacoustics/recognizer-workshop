from perch_hoplite.db import sqlite_usearch_impl
from pathlib import Path

from baw_helpers import baw_helpers, baw_api
from recognizer_workshop.perch_hoplite_custom_modifications import sqlite_usearch_impl as sqlite_usearch_impl_custom

def debug_missing(db_path, 
                  db, 
                  region_id, 
                  source_ids):



    loaded_db = sqlite_usearch_impl.SQLiteUsearchDB.create(db_path)

    passed_db_sources = sqlite_usearch_impl_custom.get_all_sources(db)
    loaded_db_sources = sqlite_usearch_impl_custom.get_all_sources(loaded_db)

    print(f"Found {len(passed_db_sources)} sources in the supplied db {db}")
    print(f"Found {len(loaded_db_sources)} sources in the loaded db {loaded_db}")

    passed_db_sources = [source.source_id for source in passed_db_sources]
    loaded_db_sources = [source.source_id for source in loaded_db_sources]

    # Check if the loaded and passed source IDs match
    if not set(passed_db_sources) == set(loaded_db_sources):
        print("Source IDs do not match between the passed and loaded databases.")
    else :
        print("Source IDs match between the passed and loaded databases.")

    if not set(source_ids) == set(passed_db_sources):
        print("Source IDs do not match between the passed source_ids (obtained checked baw for recordings that have events) and the passed database sources.")
    else:
        print("Source IDs match between the passed source_ids and the passed database sources.")

    api = baw_api.baw_api()

    events =  baw_helpers.get_events_for_sites(api, {'region_id':[region_id]})
    arids_from_events = list(set([event['audio_recording_id'] for event in events]))
    print(f"Found {len(arids_from_events)} events for region {region_id}")


    arid_from_source_id = lambda source: int(source.split('/')[-2])

  

    passed_source_id_arids = [arid_from_source_id(source) for source in source_ids]
    passed_db_sources_arids = [arid_from_source_id(source) for source in passed_db_sources]
    loaded_db_sources_arids = [arid_from_source_id(source) for source in loaded_db_sources]

    if not set(passed_source_id_arids) == set(passed_db_sources_arids):
        print("passed_source_id_arids do not match passed_db_sources_arids.")

    if not set(passed_source_id_arids) == set(loaded_db_sources_arids):
        print("passed_source_id_arids do not match loaded_db_sources_arids.")


    missing_arids = set(arids_from_events) - set(passed_db_sources_arids)

    print(f"Missing audio_recording_ids in the passed database: {len(missing_arids)}")
    for arid in missing_arids:
        print(arid)


# db_path = Path("local_scripts/test_folder/db/116")
# db = sqlite_usearch_impl.SQLiteUsearchDB.create(db_path)
# debug_missing(db_path, db, region_id=116)