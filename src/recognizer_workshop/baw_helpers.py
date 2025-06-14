"""
Helpers for BAW API interactions for agile modelling
"""
import random
from . import baw_api
import csv
import os

def simple_filter(
    project_id=None,
    region_id=None,
    site_id=None,
    arid=None):
    """Helper to create a simple filter for BAW API queries."""
    filter_params = {}

    def add_filter_param(param_name, param_value):
        # if type(param_value) is int:
        #     filter_params[param_name] = {
        #         "eq": param_value
        #     }
        # elif isinstance(param_value, (list, set, tuple)) and len(param_value) > 0:
        #     filter_params[param_name] = {
        #         "in": param_value
        #     }
        if type(param_value) is int:
            param_value = [param_value]
            filter_params[param_name] = {
                "in": param_value
            }


    add_filter_param("projects.id", project_id)
    add_filter_param("regions.id", region_id)
    add_filter_param("sites.id", site_id)
    add_filter_param("id", arid)
    if len(filter_params) == 0:
        return None
    return filter_params


def get_random_audio_recordings(
        baw_api,
        filter,
        limit=10,
        seed=42,
        order_by="created_at",
        order_direction="desc",
    ):
    """Get random audio recordings from the BAW API."""

    filter_payload = {
        "filter": filter,
        "projection": {
            "include": ["id", "site_id", "canonical_file_name", "sites.name", 
                        "regions.name", "regions.id", "duration_seconds"]
        },
        "sorting": {
            "order_by": order_by,
            "direction": order_direction
        }
    }

    endpoint = "/audio_recordings/filter?disable_paging=true"

    response = baw_api.post(endpoint, data=filter_payload)

    if not response or "data" not in response:
        raise ValueError("No data found in response from BAW API")

    recordings = response.get("data")

    real_limit = min(limit, len(recordings))
    if real_limit != limit:
        print(f"Warning: Only {real_limit} recordings found, requested {limit}")
    
    # Shuffle the recordings to get a random selection
 
    random.seed(seed)
    random.shuffle(recordings)
    return recordings[:real_limit]
    


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



def get_events_for_sites(baw_api, filter_params):
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
    response = baw_api.post("/audio_events/filter?disable_paging=true", data={"filter": ae_filter})
    # maybe site by site? or if not then paging, but that is complicated

    # sites_data = []
    # for sid in site_id:
    #     event_filter = {"audio_recordings.site_id": {"eq": sid}}
    #     response = baw_api.post("/audio_events/filter?disable_paging=true", data={"filter": event_filter})
    #     if response and "data" in response:
    #         sites_data.extend(response["data"])

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
