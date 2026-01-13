# packages/baw-helpers/src/baw_helpers/tag_cache.py
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List
from dateutil.parser import parse as parse_date

log = logging.getLogger(__name__)

class TagCache:
    def __init__(self, api_client, cache_file: str = "tags_cache.json"):
        self.api = api_client
        self.cache_file = Path(cache_file)

        # the tags in memory as dict with 'id' key
        self._tags: dict = {}

        self._load_cache_file()
        
        self.refresh()


    def _load_cache_file(self):
        """Loads the cache file from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._tags = self._index_by_id(data)
            except Exception as e:
                log.warning(f"Failed to read local tag cache: {e}")


    def _get_latest_server_timestamp(self):
        """Fetches the latest tag updated_at timestamp from the server"""

        return self.api.post("/tags/filter?disable_paging=true", data={
            "filter": {
                "projection": ["updated_at"],
                "sort": [{"order_by": "updated_at", "direction": "desc"}],
            }
        }).get('data', [{'updated_at': None}])[0]['updated_at']
    

    def _get_latest_local_timestamp(self):

        # iterate through _tags to find the latest updated_at
        if not self._tags:
            return None
        # updated at as a string should still be comparable
        latest = max([tag['updated_at'] for tag in self._tags.values()])
        return latest
        
       

    def refresh(self):
        """
        Smart refresh: checks if the server has newer data than our cache file.
        If server is newer (or cache missing), downloads ALL tags.
        """
        

        if not self._tags:
            self._download_all_tags()
            return

        
        try:
            latest_check = self.api.post("/tags/filter?disable_paging=true", data={
                "filter": {
                    "projection": ["updated_at"],
                    "sort": [{"order_by": "updated_at", "direction": "desc"}],
                }
            }).get('data', [{'updated_at': None}])
            
            server_latest = latest_check[0]['updated_at']

                
        except Exception as e:
            log.warning(f"Could not check server tag freshness: {e}")
            return

        local_timestamp = self._get_latest_local_timestamp()
        is_stale = (local_timestamp is None) or (server_latest > local_timestamp)
        
        if is_stale:
            log.info("Tag cache is stale")
            self._download_all_tags()
        else:
            log.info("Tag cache is up to date.")


    def _download_all_tags(self):
        """Fetches ALL tags and saves to disk."""
        log.info("Downloading all tags from server...")
        response = self.api.get("/tags", params={"disable_paging": "true"})
        
        if not response or 'data' not in response:
            log.error("Failed to download tags.")
            return

        all_tags = response['data']

        self._tags = self._index_by_id(all_tags)
        
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(all_tags, f)
            log.info(f"Saved {len(all_tags)} tags to {self.cache_file}")
        except Exception as e:
            log.error(f"Failed to write tag cache to disk: {e}")


    def _index_by_id(self, tag_list):
        return {int(tag['id']): tag for tag in tag_list}


    def get_by_ids(self, tag_ids: int | list[int]) -> Optional[dict]:
        if isinstance(tag_ids, int):
            tag_ids = [tag_ids]
        return {tag_id: self._tags.get(tag_id) for tag_id in tag_ids}


    def get_all_tag_texts(self) -> List[str]:
        tag_list = [tag['text'] for tag in self._tags.values()]
        tag_list.sort()
        return tag_list
