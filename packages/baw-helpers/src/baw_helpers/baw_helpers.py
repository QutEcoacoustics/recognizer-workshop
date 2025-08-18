"""
Helpers for BAW API interactions for agile modelling
"""
import random
from . import baw_api
import csv
import os
from pathlib import Path
import json
from typing import Any, Union

from baw_helpers.utils import stable_hash


def simple_filter(
    project_ids: int | list | None = None,
    region_ids: int | list | None = None,
    site_ids: int | list | None = None,
    arids: int | list | None = None,
    filter_params: dict | None = None):
    """Helper to create a simple filter for BAW API queries."""
    if filter_params is None:
        filter_params = {}

    def add_filter_param(param_name, param_value):
        if type(param_value) is int:
            filter_params[param_name] = {
                "eq": param_value
            }
        elif isinstance(param_value, (list, set, tuple)) and len(param_value) > 0:
            filter_params[param_name] = {
                "in": param_value
            }
            

    add_filter_param("projects.id", project_ids)
    add_filter_param("regions.id", region_ids)
    add_filter_param("sites.id", site_ids)
    add_filter_param("id", arids)
    if len(filter_params) == 0:
        return None
    
    return {
        "filter": filter_params
    }

def wrap_filter(filter_params: dict) -> dict:
    """Accept bare dict or {'filter': {...}}; return full payload."""
    return filter_params if "filter" in filter_params else {"filter": filter_params}


def get_filter_result(
    api,                       
    model: str,
    filter_params: dict,
    save_path: str | Path,        
    qsp: dict | None = None,
    use_cache: bool = True,
    *,
    include_qsp_in_hash: bool = True,
    pretty: bool = True,
) -> tuple[Any | None, Path]:
    """
    Get filter result with simple file cache.

    Returns:
        (data, path) where:
          - data is parsed JSON (dict/list) or None on failure
          - path is the full cache file path (even if not written due to failure)

    Notes:
      - Hash is computed over the *complete* 'filter' dict.
      - If include_qsp_in_hash=True, the qsp is included in the hash basis.
    """
    payload = wrap_filter(filter_params)
    basis = {"filter_payload": payload}
    if include_qsp_in_hash and qsp:
        basis["qsp"] = qsp
    # 3 bytes = 6 hex chars for ~16m possible values. 
    h = stable_hash(basis, digest_size=3)  # deterministic across key order/types
    out_dir = Path(save_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model}_{h}.json"

    if use_cache and out_file.exists():
        try:
            data = json.loads(out_file.read_text(encoding="utf-8"))
        except Exception:
            data = None  # corrupted cache; caller can decide what to do
        return data, out_file

    # Miss or bypass: call API
    data = api.baw_filter(model, payload, qsp=qsp)
    if data is None:
        # API layer already handled/logged; return path so caller knows where it would be
        return None, out_file

    with out_file.open("w", encoding="utf-8") as f:
        if pretty:
            json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

    return data, out_file


def get_random_audio_recordings(
    api,
    filter_params: dict,
    save_dir: str | Path,
    *,
    limit: int = 10,
    seed: int = 42,
    order_by: str = "created_at",
    order_direction: str = "desc",
    use_cache: bool = True,
) -> list[dict] | tuple[list[dict], Path]:
    """
    Fetch all recordings for the filter (disable paging), cache the JSON,
    then return a deterministic random subset.
    """

    payload = wrap_filter(filter_params)
    

    if "projection" not in payload:
        payload["projection"] = {
                "include": ["id", "site_id", "canonical_file_name", "sites.name", 
                            "regions.name", "regions.id", "duration_seconds"]
            }

    if "sorting" not in payload:
        payload["sorting"] = {
            "order_by": order_by,
            "direction": order_direction,
        }

    qsp = {"disable_paging": 'true'}
    data, cache_path = get_filter_result(
        api=api,
        model="audio_recordings",
        filter_params=payload,
        save_path=save_dir,
        qsp=qsp,
        use_cache=use_cache,
        include_qsp_in_hash=True
    )

    if not data or "data" not in data:
        raise ValueError("No data found in response (or cached file) for audio_recordings filter")

    recordings = data["data"]
    if not isinstance(recordings, list):
        raise ValueError("Unexpected response format: 'data' is not a list")

    # 3) Deterministic shuffle preserving ordering

    n = len(recordings)
    if limit >= n or limit < 0:
        return recordings, cache_path
    rng = random.Random(seed)
    idxs = sorted(rng.sample(range(n), limit))   # pick k unique positions, then sort them
    out = [recordings[i] for i in idxs]

    return out, cache_path


def get_embeddings(
    baw_api,
    audio_recording_id: list,
    analysis_job_id: int,
    destination_file=None,
    analysis_name="perch-embeddings_4"
    ):
    """
    There might be a better way to do this but:
    requests metatadata about an audio recording id for an analysis job
    which returns the path to the actual emebddings file. 
    we then download the embeddings file. 
    """

    metadata_endpoint = lambda ajid, arid: f'/analysis_jobs/{ajid}/results/{arid}/{analysis_name}'

    aj_meta = baw_api.get(metadata_endpoint(analysis_job_id, audio_recording_id))

    try:
      path = aj_meta.get("data").get("children", [])[0]['path']
    except (AttributeError, IndexError):
        return None

    result = baw_api.download(path, destination_file)

    print(result)

    return result


def get_events_for_sites(baw_api, filter_params, tags=None):
    """Get any audio recordings that have events for a specific region.
    region_index is the index of the region in the regions list.
    @param num: maximum number of audio recordings to get.

    This we can only use site_ids, so we need to get those if project or region ids are provided.
    """

    site_id = filter_params.get("site_id")
    project_id = filter_params.get("project_id")
    region_id = filter_params.get("region_id")

    if site_id is None:
        site_filter = {}
        if project_id is not None:
            site_filter["projects.id"] = {"in": project_id}
        elif region_id is not None:
            site_filter["regions.id"] = {"in": region_id}
        else:
            raise ValueError("Either site_id, project_id or region_id must be provided.")
        sites = baw_api.post("/sites/filter?disable_paging=true", data={"filter": site_filter})
        site_id = [site["id"] for site in sites.get("data", [])]
        
    # this is too slow and times out for lots of sites
    ae_filter = {
        "audio_recordings.site_id": {"in": site_id}
    }

    if tags is not None:
        if isinstance(tags, str):
            tags = [tags]
        ae_filter['tags.text'] = { "in": tags }

    response = baw_api.post("/audio_events/filter?disable_paging=true", data={"filter": ae_filter})
    # maybe site by site? or if not then paging, but that is complicated


    return response.get("data", [])


def get_verifications_for_sites(api, event_ids=None, filter_params=None):
    """Gets any audio recordings that have verifications so that we can exclude non-verified machine-generated events.
    """

    if event_ids is None:

        if filter_params is None:
            raise ValueError("Either event_ids or filter_params must be provided.")
        else:
            # first get all audio events for the target
            events = get_events_for_sites(api,filter_params)
            event_ids = [event["id"] for event in events]

    endpoint = f"/verifications/filter?disable_paging=true"
    filter_params = {
        "audio_event_id": {"in": event_ids}
    }
    response = api.post(endpoint, data={"filter": filter_params})

    return response.get("data", [])



def get_untagged_events(api, filter_params, save_path=None):

    events = get_events_for_sites(api, filter_params)

    # remove events that have no taggings
    events_without_taggings = [event for event in events if 'taggings' not in event or len(event['taggings']) == 0]

    # for each event dict in the events list of dicts, only keep the following keys: id, audio_recording_id, start_time_seconds, end_time_seconds
    events_without_taggings = [{key: event[key] for key in ['id', 'audio_recording_id', 'start_time_seconds', 'end_time_seconds']} for event in events_without_taggings]

    # for each item, add a link, which is https://www.ecosounds.org/listen/audio_recordings/{audio_recording_id}/events/{id}
    for event in events_without_taggings:
        # start is the start_time_seconds rounded down to the nearest 30 seconds
        start = int(event['start_time_seconds'] - (event['start_time_seconds'] % 30))
        end = start + 30
        event['link'] = f"https://www.ecosounds.org/listen/{event['audio_recording_id']}?start={start}&end={end}"

    # save as csv file
    if save_path is not None:

        with open(save_path, 'w', newline='') as csvfile:
            fieldnames = ['id', 'audio_recording_id', 'start_time_seconds', 'end_time_seconds', 'link']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for event in events_without_taggings:
                writer.writerow(event)
        print(f"Saved {len(events_without_taggings)} untagged events to {save_path}")
    
    return events_without_taggings


def canonical_file_name_to_source(file_name: str, base_url="https://api.ecosounds.org") -> str:
    """Convert a file name to a source URL."""
    # Assuming the file name is in the format 'file_id_offset.parquet'
    stem = Path(file_name).stem
    parts = stem.split('_')
    if len(parts) < 2:
        raise ValueError(f"Invalid file name format: {file_name}")
    arid = parts[-1]
    # no trailing slash!!
    return f"{base_url}/audio_recordings/{arid}/original"


def annotation_upload_csv(api, filepath_to_csv, provenance_id, import_name, commit='true'):

    event_import_endpoint = "/audio_event_imports"
    # check if they have a event_import we can use

    # todo sort by created_at desc
    existing_imports = api.baw_filter('audio_event_imports', filter_params={'name': {'eq': import_name}})
    existing_imports = existing_imports.get('data', [])
    if existing_imports and len(existing_imports) > 0:
        import_id = existing_imports[0]['id']
    else:
        # create a new import
        data = {
            "name": import_name,
            "description": "Import for recognizer workshop",
        }
        response = api.post(event_import_endpoint, data=data)
        if response and 'data' in response:
            import_id = response['data']['id']
        else:
            print("Failed to create a new audio_event_import record.")
            return

    event_file_upload_endpoint = f"/audio_event_imports/{import_id}/files"

    filepath_to_csv = Path(filepath_to_csv)  

    try:
        with open(filepath_to_csv, 'rb') as f:
            files_payload = {
                "audio_event_import_file[file]": (filepath_to_csv.name, f, 'text/csv')
            }
            params = {
                'provenance_id': provenance_id,
                'commit': commit
            }

            print(f"Uploading '{filepath_to_csv.name}' to the '{event_file_upload_endpoint}' endpoint...")
            
            # 5. Call the post_multipart method
            response = api.post_files(  
                endpoint=event_file_upload_endpoint,
                files=files_payload,
                params=params
            )

            print("\n--- API Response ---")
            if response:
                api.print_response(response)
            else:
                print("Upload failed. Check the error messages above.")

            return response

    except FileNotFoundError:
        print(f"ERROR: The file was not found at the path: {filepath_to_csv}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
