#!/usr/bin/env python3
import sqlite3
import logging
import argparse
import sys
from pathlib import Path

# Ensure the current directory is in the path
sys.path.append(str(Path.cwd()))

from baw_helpers.settings import baw_settings

try:
    from baw_helpers.baw_utils import extract_arid_from_filename, original_baw_audio_url_from_arid
except ImportError:
    print("Error: Could not import 'baw_helpers'. Make sure you run this from the project root (recognizer-workshop).")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("db_fixer")

def fix_database_sources(db_path: Path, apply_changes: bool = False, api_domain: str = None):

    if api_domain is None:
        api_domain = baw_settings.default_api
    
    # 1. Resolve Path
    if db_path.is_dir():
        potential_db = db_path / "hoplite.sqlite"
        if potential_db.exists():
            db_path = potential_db
        else:
            log.error(f"Directory found, but 'hoplite.sqlite' does not exist in: {db_path}")
            return

    # 2. Hard Check for Existence and Size
    if not db_path.exists():
        log.error(f"❌ File not found: {db_path}")
        return
    
    if db_path.stat().st_size == 0:
        log.error(f"❌ File exists but is EMPTY (0 bytes): {db_path}")
        return

    log.info(f"Opening database: {db_path}")
    
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=rw", uri=True)
    except sqlite3.OperationalError as e:
        log.error(f"❌ Could not connect to database: {e}")
        return

    cursor = conn.cursor()

    # 4. Verify Schema (CORRECTED TABLE NAME)
    TABLE_NAME = 'hoplite_sources'
    
    try:
        # Check table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (TABLE_NAME,))
        if not cursor.fetchone():
            log.error(f"❌ Table '{TABLE_NAME}' not found.")
            # Fallback check to see what tables ARE there
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [r[0] for r in cursor.fetchall()]
            log.info(f"Tables found: {tables}")
            conn.close()
            return

        # Check column names
        cursor.execute(f"PRAGMA table_info({TABLE_NAME})")
        columns = {row[1] for row in cursor.fetchall()}
        
        # Determine if the column is 'source' or 'source_id'
        if 'source' in columns:
            col_name = 'source'
        elif 'source_id' in columns:
            col_name = 'source_id'
        else:
            log.error(f"❌ Could not find a source column in '{TABLE_NAME}'. Columns: {columns}")
            conn.close()
            return

        log.info(f"Found table '{TABLE_NAME}' with source column '{col_name}'.")

        # Fetch data
        cursor.execute(f"SELECT id, {col_name} FROM {TABLE_NAME}")
        rows = cursor.fetchall()
        
    except sqlite3.DatabaseError as e:
        log.error(f"❌ Database error: {e}")
        conn.close()
        return

    # 5. Scan for fixes
    updates = []
    skipped = 0
    
    log.info(f"Scanning {len(rows)} sources...")

    for internal_id, current_source in rows:
        if str(current_source).startswith(("http://", "https://")):
            continue

        arid = extract_arid_from_filename(current_source)

        if arid:
            new_source_url = original_baw_audio_url_from_arid(
                arid=arid, 
                domain=api_domain
            )
            if new_source_url != current_source:
                updates.append((new_source_url, internal_id, current_source))
        else:
            skipped += 1

    if not updates:
        log.info(f"✅ Database is clean! (Skipped {skipped} valid/unrecognized records)")
        conn.close()
        return

    log.info(f"Found {len(updates)} sources that need fixing.")

    # 6. Apply or Dry Run
    if not apply_changes:
        log.warning("--- DRY RUN MODE (No changes applied) ---")
        limit = 5
        for new, uid, old in updates[:limit]:
            log.info(f"Would change ID {uid}: '{old}' -> '{new}'")
        if len(updates) > limit:
            log.info(f"...and {len(updates) - limit} more.")
        log.info(f"\nRun with '--apply' to execute changes.")
    else:
        log.info("Applying updates...")
        count = 0
        errors = 0
        for new_url, internal_id, old_name in updates:
            try:
                # Use the dynamic table and column name
                cursor.execute(f"UPDATE {TABLE_NAME} SET {col_name} = ? WHERE id = ?", (new_url, internal_id))
                count += 1
            except sqlite3.IntegrityError:
                log.warning(f"Skipping ID {internal_id}: URL already exists in DB.")
                errors += 1

        conn.commit()
        log.info(f"✅ Done. Updated {count} records.")

    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix BAW source IDs in a Hoplite/SQLite database.")
    parser.add_argument("db_path", type=Path, help="Path to sqlite file or dir.")
    parser.add_argument("--apply", action="store_true", help="Apply changes.")
    args = parser.parse_args()
    fix_database_sources(args.db_path, args.apply)