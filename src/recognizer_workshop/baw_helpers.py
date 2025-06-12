"""
Helpers for BAW API interactions for agile modelling
"""
import random
from . import baw_api


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



def get_events_for_sites(baw_api, site_ids=None, project_ids=None, region_ids=None):
    """Get any audio recordings that have events for a specific region.
    region_index is the index of the region in the regions list.
    @param num: maximum number of audio recordings to get.
    """

    if site_ids is None:
        site_filter = {}
        if project_ids is not None:
            site_filter["projects.id"] = {"in": project_ids}
        elif region_ids is not None:
            site_filter["regions.id"] = {"in": region_ids}
        sites = baw_api.post("/sites/filter?disable_paging=true", data={"filter": site_filter})
        site_ids = [site["id"] for site in sites.get("data", [])]
        

    
    ae_filter = {
        "audio_recordings.site_id": {"in": site_ids}
    }

    response = baw_api.post("/audio_events/filter?disable_paging=true", data={"filter": ae_filter})

    return response.get("data", [])




