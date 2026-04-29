"""
Rebuild a usearch index from an existing SQLite hoplite database.

When the usearch.index file is missing or corrupted but the SQLite DB is intact,
this module re-downloads the parquet files for each source in the DB, matches
rows by (source, offset) to existing embedding IDs, and populates a fresh
usearch index — without modifying the SQLite tables.
"""

import functools
import logging
import threading
from pathlib import Path
from typing import Callable

import numpy as np
import pyarrow.parquet as pq
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from perch_hoplite.db.sqlite_usearch_impl import deserialize_embedding

from baw_helpers.baw_utils import extract_domain_and_arid
from baw_helpers.baw_api import AuthenticationError

from recognizer_workshop.app.create_database.create_database import (
    BAW_ANALYSIS_JOB_ID,
    DEFAULT_BAW_ANALYSIS_NAME,
    convert_source_to_original_url,
)

log = logging.getLogger(__name__)


def _get_source_embedding_map(db):
    """
    Build a mapping from (source_url, offset) -> embedding_id using SQLite data.

    Returns:
        dict[str, dict[float, int]]: {source_url: {offset: embedding_id}}
    """
    cursor = db._get_cursor()
    cursor.execute("""
        SELECT he.id, hs.source, he.offsets
        FROM hoplite_embeddings he
        JOIN hoplite_sources hs ON hs.id = he.source_idx
    """)

    source_map = {}
    for emb_id, source_url, offset_blob in cursor.fetchall():
        offsets = deserialize_embedding(offset_blob, db.offset_dtype)
        if source_url not in source_map:
            source_map[source_url] = {}
        for offset in offsets.tolist():
            source_map[source_url][offset] = int(emb_id)

    return source_map


def _get_arids_from_sources(source_urls):
    """
    Extract unique (domain, arid) pairs from source URLs.

    Returns:
        list[dict]: [{'id': arid, 'domain': domain, 'source_url': source_url}, ...]
    """
    seen = set()
    items = []
    for url in source_urls:
        domain, arid = extract_domain_and_arid(url)
        key = (domain, arid)
        if arid is not None and key not in seen:
            seen.add(key)
            items.append({'id': arid, 'domain': domain, 'source_url': url})
    return items


def repair_index(
    db,
    api_clients: dict,
    *,
    prefetch: int = 1,
    analysis_job_id: int = BAW_ANALYSIS_JOB_ID,
    analysis_name: str = DEFAULT_BAW_ANALYSIS_NAME,
    progress_callback: Callable[[dict], None] | None = None,
    stop_event: threading.Event | None = None,
):
    """
    Rebuild the usearch index for a database whose SQLite tables are intact.

    For each source in the DB, downloads the parquet file, extracts embedding
    vectors for offsets that exist in hoplite_embeddings, and adds them to the
    usearch index using the original embedding IDs. SQLite is not modified.

    Args:
        db: A ThreadedSQLiteUsearchDB instance (already opened; index may be empty).
        api_clients: A dict mapping domain URLs to baw_api client instances.
        prefetch: Number of parquet downloads to queue ahead.
        analysis_job_id: BAW analysis job ID for parquet lookup.
        analysis_name: BAW analysis name for parquet lookup.
        progress_callback: Optional callback for progress updates.
        stop_event: Optional threading.Event to signal early stop.

    Returns:
        dict with repair statistics.
    """
    if not api_clients:
        raise ValueError("At least one api_client is required to download parquet files for index repair.")

    # 1. Build the lookup: (source_url, offset) -> embedding_id
    log.info("Building source/offset -> embedding_id mapping from SQLite...")
    source_offset_map = _get_source_embedding_map(db)
    total_embeddings_expected = sum(len(offsets) for offsets in source_offset_map.values())
    log.info("Found %d embeddings across %d sources in SQLite",
             total_embeddings_expected, len(source_offset_map))

    # 2. Get unique ARIDs to download
    items = _get_arids_from_sources(source_offset_map.keys())
    log.info("Need to download %d parquet files", len(items))

    # Check which domains we need and warn about missing clients
    required_domains = {item['domain'] for item in items}
    available_domains = set(api_clients.keys())
    missing_domains = required_domains - available_domains
    if missing_domains:
        log.warning("No API client for domain(s): %s — sources from these domains will be skipped", missing_domains)
        items = [item for item in items if item['domain'] in available_domains]

    if not items:
        return {
            "sources": 0,
            "embeddings_expected": 0,
            "embeddings_added": 0,
            "embeddings_skipped": 0,
            "errors": 0,
        }

    # 3. Ensure usearch index is in memory for writing
    if not db._ui_loaded:
        db.ui.load()
        db._ui_loaded = True

    # Build per-domain source_map_fns to convert parquet source column values to DB source URLs
    source_map_fns = {
        domain: functools.partial(convert_source_to_original_url, domain=domain)
        for domain in available_domains
    }

    embeddings_added = 0
    embeddings_skipped = 0
    files_processed = 0
    errors = 0

    def _emit_progress(stage="repairing", force=False):
        if progress_callback is None:
            return
        if not force and files_processed % 5 != 0:
            return
        progress_callback({
            "stage": stage,
            "files_processed": files_processed,
            "total_files": len(items),
            "embeddings_added": embeddings_added,
            "embeddings_expected": total_embeddings_expected,
            "embeddings_skipped": embeddings_skipped,
            "errors": errors,
        })

    def fetch_one(item):
        client = api_clients[item['domain']]
        buf = client.analysis_result_file(
            arid=item['id'],
            analysis_job_id=analysis_job_id,
            analysis_name=analysis_name,
            to_memory=True,
        )
        if buf is None:
            raise RuntimeError(f"No parquet returned for ARID {item['id']}")
        table = pq.read_table(buf)
        return table.to_pandas()

    # 4. Download parquets and populate index
    _emit_progress("starting", force=True)

    with ThreadPoolExecutor(max_workers=max(1, prefetch + 1)) as executor:
        q: deque = deque()
        it = iter(items)

        def submit_next():
            if stop_event and stop_event.is_set():
                return False
            try:
                item = next(it)
                q.append((item, executor.submit(fetch_one, item)))
                return True
            except StopIteration:
                return False

        # Fill initial queue
        for _ in range(min(prefetch + 1, len(items))):
            if not submit_next():
                break

        while q:
            if stop_event and stop_event.is_set():
                log.info("Repair stopped by user.")
                break

            item, fut = q.popleft()
            submit_next()

            try:
                df = fut.result()
                files_processed += 1

                for _, row in df.iterrows():
                    # Convert parquet source value to the URL format stored in SQLite
                    # using the domain-specific source_map_fn
                    row_source = source_map_fns[item['domain']](row['source'])
                    row_offset = float(row['offset'])

                    offset_map = source_offset_map.get(row_source)
                    if offset_map is None:
                        continue

                    embedding_id = offset_map.get(row_offset)
                    if embedding_id is None:
                        continue

                    # Check if this ID is already in the index (e.g. partial repair)
                    if db.ui.contains(embedding_id):
                        embeddings_skipped += 1
                        continue

                    # Extract embedding vector (columns after source, channel, offset)
                    embedding = np.array(row.iloc[3:], dtype=np.float16)
                    db.ui.add(embedding_id, embedding.astype(db.embedding_dtype))
                    embeddings_added += 1

                log.info("Processed ARID %d (%d/%d), added %d embeddings so far",
                         item['id'], files_processed, len(items), embeddings_added)
                _emit_progress()

            except AuthenticationError as e:
                errors += 1
                log.error("Auth error for ARID %d: %s", item['id'], e)
                if errors >= 3:
                    log.error("Too many auth errors, aborting repair.")
                    break
            except Exception as e:
                errors += 1
                log.error("Error processing ARID %d: %s", item['id'], e)

    # 5. Save the index
    db.ui.save()
    db._ui_updated = False
    log.info("Index repair complete. Added %d embeddings, skipped %d, errors %d",
             embeddings_added, embeddings_skipped, errors)

    stats = {
        "sources": len(items),
        "embeddings_expected": total_embeddings_expected,
        "embeddings_added": embeddings_added,
        "embeddings_skipped": embeddings_skipped,
        "errors": errors,
    }

    _emit_progress("complete", force=True)
    return stats
