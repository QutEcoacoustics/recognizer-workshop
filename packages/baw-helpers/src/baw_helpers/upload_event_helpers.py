import logging
from baw_helpers import baw_helpers
from hashlib import sha256
import json
from pathlib import Path
import pandas as pd

log = logging.getLogger(__name__)


def lookup_arid(api, original_filename, filter_params):
    """
    Look up the Audio Recording ID (arid) based on the project ID and original filename.
    """
    filter_params = baw_helpers.simple_filter(**filter_params)
    filter_params["original_file_name"] = {"eq": original_filename}
    response = api.post('audio_recordings/filter', data={'filter': filter_params})
    
    data = response.get('data', [])
    if len(data) == 0:
        log.warning(f"No audio recording found for original filename: {original_filename}")
        return None
    elif len(data) > 1:
        log.warning(f"Warning: Multiple audio recordings found for original filename: {original_filename}. Returning None.")
        return None
    # Return the first audio recording ID found
    return data[0].get('id', None)


def make_arid_map(api, original_filenames, filter_params, cache=None):
    """
    Create a mapping of original filenames to audio recording IDs.
    Save the mapping to a cache file to avoid repeated lookups.
    
    """

    if cache is None:
        cache = Path('cache')
    cache.mkdir(parents=True, exist_ok=True)

    # Create a hash concatenated original filenames and filter parameters to use as a cache key
    arid_map_hash =  sha256(''.join([Path(original_filenames).name 
                                     for original_filenames in original_filenames] + [str(item) for pair in filter_params.items() for item in pair]).encode()).hexdigest()[0:7]
    if (cache / f'arid_map_{arid_map_hash}.json').exists():
        log.info(f"Loading ARID map from cache: {cache / f'arid_map_{arid_map_hash}.json'}")
        with open(cache / f'arid_map_{arid_map_hash}.json', 'r') as f:
            arid_map = json.load(f)
    else:
        arid_map = {}
        for original_filename in original_filenames:
            
            arid = lookup_arid(api, filter_params=filter_params, original_filename=original_filename)
            if arid:
                arid_map[str(original_filename)] = arid
            else:
                log.warning(f"Audio Recording ID not found for {original_filename}")
        # Save the ARID map to cache
        with open(cache / f'arid_map_{arid_map_hash}.json', 'w') as f:
            json.dump(arid_map, f, indent=2)
        log.info(f"ARID map saved to cache: {cache / f'arid_map_{arid_map_hash}.json'}")

    return arid_map


def get_original_filename_and_type(file_path, type=None, prefix=None):  
    """
    Tries to determine the type of file based on its filename pattern
    Returns the audio filename encoded in the file path
    """

    basename = Path(file_path).name

    # sometimes the filename is prefixed with something, e.g. 'YBG_'
    if prefix is not None and basename.startswith(prefix):
        basename = str(basename).replace(prefix, '', 1)

    if type is None:
        if basename.endswith('.data') or basename.endswith('.data.json'):
            type = 'avianz'
        elif 'selectiontable' in basename:
            type = 'raven'
        elif 'todo:arbimon' in basename:
            type = 'arbimon'
        else:
            type = 'unknown'
        
    original_filename = "file type not recognized"
    
    # remove the '.data' extension
    if type == 'avianz':
        original_filename=basename.rsplit('.data', 1)[0]

    log.info(f"Prepared file path: {basename}")

    return original_filename, type


def process_files(api, file_paths, filter_params, extra_labels=None, prefix=None, csv_path=None, type=None):
    """
    Process multiple files.
    For each file, determine the type, extract the original filename, and process the file accordingly.
    """
    
    #map the hash of the concatenated basenames to get a hash that we can save the arid_map to for cacheing
    #file_paths = file_paths[1:3]

    # list of tuples (original_filename, type, file_path)
    items = [get_original_filename_and_type(file_path, type=type, prefix=prefix) + (file_path,) for file_path in file_paths]

    # convert to list of dictionaries
    items = [{'original_filename': item[0], 'type': item[1], 'file_path': item[2]} for item in items]

    arid_map = make_arid_map(api, [item['original_filename'] for item in items], filter_params)    

    all_dfs = []
    for item in items:
        arid = arid_map.get(item['original_filename'], None)

        if item['type'] == 'avianz':
            df = process_avianz_file(item['file_path'], arid=arid, extra_labels=extra_labels)
        elif item['type'] == 'raven':
            df = process_raven_file(item['file_path'], arid=arid, extra_labels=extra_labels)
        elif item['type'] == 'arbimon':
            df = process_arbimon_file(item['file_path'], arid=arid, extra_labels=extra_labels)
        else:
            # warning for unknown file type
            log.warning(f"Unknown file type '{item['type']}' for file {item['file_path']}. Skipping.")
            continue

        all_dfs.append(df)
    
    # Concatenate all DataFrames
    all_events_df =  pd.concat(all_dfs, ignore_index=True)

    if csv_path:
        # Save to CSV if a path is provided
        # create the directory if it doesn't exist
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        all_events_df.to_csv(csv_path, index=False)
        log.info(f"Events saved to {csv_path}")

    return all_events_df




def process_avianz_file(file_path, 
                project_id=None,
                arid=None,
                extra_labels="AviaNZ"):
    """Prepare the file path for AviaNZ."""


    # read file_path as json
    with open(file_path, 'r') as f:
        avianz_data = json.load(f)
        log.debug(f"Data loaded from {file_path}.json: {avianz_data}")

    #pretty print the data
    #print(json.dumps(avianz_data, indent=2))
    # if arid is None:
    #     if project_id is None:
    #         raise ValueError("Either 'arid' or 'project_id' must be provided.")
    #     original_filename = get_original_filename_and_type(file_path, 'avianz')
    #     arid = lookup_arid(api, project_id=project_id, original_filename=original_filename)

    if arid:
        log.info(f"Audio Recording ID: {arid}")
    else:
        log.warning("Audio Recording ID not found for the given file path.")

    events = []
    
    # get all items except the first one
    # the first item is metadata about the annotator, reviewer etc
    avianz_events = avianz_data[1:]

    for avianz_item in avianz_events:
        
        # print(avianz_item)
        # exit(0)

    
        event = {
              "audio_recording_id": arid,
              "start": avianz_item[0],
              "end": avianz_item[1],
              "low_frequency_hertz": avianz_item[2],
              "high_frequency_hertz": avianz_item[3],
              "score": avianz_item[4][0]["certainty"], # if there are multiple annotations, use the score of the first one
              "tag": ";".join([annotation["species"] for annotation in avianz_item[4]]), # if there are multiple annotations, join them to multiple tags
        }
        event["tag"] = f"{event['tag']};{extra_labels}"
        events.append(event)

    
        

    # convert to pandas DataFrame
    df = pd.DataFrame(events)

    return df



def process_raven_file(file_path, 
                project_id=None,
                arid=None,
                extra_labels="RavenLite"):
    """Prepare the file path for Raven."""
    log.warning(f"Processing Raven file not implemented yet: {file_path}")
    return None

def process_arbimon_file(file_path, 
                project_id=None,
                arid=None,
                extra_labels="Arbimon"):
    """Prepare the file path for Arbimon."""
    log.warning(f"Processing Arbimon file not implemented yet: {file_path}")
    return None


