import logging

from recognizer_workshop.perch_hoplite_custom_modifications import sqlite_usearch_impl as sqlite_usearch_impl_mod
import baw_helpers.baw_utils as baw_utils
import pandas as pd

log = logging.getLogger(__name__)

def make_site_map(db, baw_client):
    """
    Creates a mapping from the hoplite db source_id to sites on ecoeounds
    """

    all_sources = sqlite_usearch_impl_mod.get_all_sources(db)
    # initialize empty site map
    site_map = {baw_utils.extract_arid_and_domain(src.source_id)[1]: {
        'source': src.source_id,
        'site_id': None,
        'site_name': None,
    } for src in all_sources}

    arids = [arid for arid in site_map.keys()]

    response = baw_client.post(
        '/audio_recordings/filter/?disable_paging=true',
        data={
            'filter': {
                'id': {
                    'in': arids
                }
            },
            'projection': { "include":['id', 'site_id', 'sites.name'] }
        }
    )

    recording_records = response.get('data', [])
    recording_records = {r['id']: r for r in recording_records}
    for arid, src in site_map.items():
        if arid in recording_records:
            rec = recording_records[arid]
            src['site_id'] = rec['site_id']
            src['site_name'] = rec['sites.name']
        else:
            log.warning(f'No recording found for arid {arid} in the BAW API')

    return site_map


def save_search_results(engine, path, append=False):
    """
    saves the search results to a csv file
    TODO: This is broken, search_results is a dict with a results set per label
    """
    rows = []
    for res in engine.search_results.search_results:
        source = engine.db.get_embedding_source(res.embedding_id)
        domain, arid = baw_utils.extract_arid_and_domain(source.source_id)
        offset = float(source.offsets[0])
        audio_url = baw_utils.make_baw_audio_url_from_arid(arid, offset, 5.0, domain)
        #source = self.db.get_source(embedding.source_id)
        row = {
          'embedding_id': res.embedding_id,
          'arid': arid,
          'offset': offset,
          'score': res.sort_score,
          'link': audio_url
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if append:
        df.to_csv(path, mode='a', header=False, index=False)
    else:
        df.to_csv(path, index=False)
