import asyncio
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import panel as pn

from recognizer_workshop.app.create_database.create_database import create_database_from_baw
from recognizer_workshop.ui.panel.components.native.filter_editor import build_filter_editor, resolve_filters_for_editor
from recognizer_workshop.ui.panel.utils import patch_tqdm

log = logging.getLogger(__name__)

def build(vm):
    if not hasattr(vm, 'config') or not vm.config or not vm.config.db_path:
        return pn.pane.Alert(
            "Configuration not available. Please configure your project on the Config tab first.",
            alert_type="warning"
        )

    # Create help buttons
    database_help_btn = vm.help_system.create_help_button("help-database-overview", "Help: Database Overview")
    db_settings_help_btn = vm.help_system.create_help_button("help-db-creation-settings", "Help: Database Creation Settings")
    recordings_filter_help_btn = vm.help_system.create_help_button("help-recordings-filter", "Help: Recordings Filter")

    page_state = {'filters': []}
    max_files = pn.widgets.IntInput(name="Max Recordings to Scan", value=1000, start=-1)
    max_embeddings = pn.widgets.IntInput(name="Max Total Embeddings in DB", value=200_000, start=-1)
    use_cached_list = pn.widgets.Checkbox(name="Use Cached Recording List", value=True, visible=False)
    cache_hint = pn.pane.Markdown("", visible=False)
    btn_create = pn.widgets.Button(name="Create Database", button_type="primary", icon="database")
    progress_bar = pn.widgets.Progress(name='Progress', active=True, visible=False, sizing_mode="stretch_width")
    status_display = pn.pane.Markdown("")  # Renamed from 'log' to avoid confusion
    progress_lines: list[str] = []
    
    # Alert shown when no auth token is available
    auth_warning = pn.pane.Alert(
        "⚠️ Authentication required: Database creation requires a valid API token. "
        "Please configure your token on the Config tab.",
        alert_type="warning",
        visible=False
    )
    
    db_path_display = pn.widgets.StaticText(
        name="Target Database Directory", 
        value=str(vm.config.db_path)
    )

    def _hash_filters(filters):
        if not filters or not filters[0]: return None
        filter_str = json.dumps(filters[0], sort_keys=True).encode('utf-8')
        return hashlib.sha256(filter_str).hexdigest()[:10]

    def _cached_file_for_filters(filters) -> Path | None:
        if not vm.config or not vm.config.db_path:
            return None
        filter_hash = _hash_filters(filters)
        if not filter_hash:
            return None
        db_dir = vm.config.db_path.expanduser()
        return db_dir / f"_source_arids_{filter_hash}.json"

    def _update_cache_toggle_state():
        cached_file = _cached_file_for_filters(page_state.get('filters'))
        has_cache = cached_file is not None and cached_file.exists()
        use_cached_list.visible = has_cache
        cache_hint.visible = has_cache
        if has_cache:
            cache_hint.object = f"ℹ️ Cached recording list found: `{cached_file.name}`"
        else:
            cache_hint.object = ""

    def get_page_filters():
        # Hydrate local page state from shared resolver if empty.
        if page_state.get('filters'):
            return page_state.get('filters')
        resolved = resolve_filters_for_editor(
            lambda: page_state.get('filters'),
            vm,
            dataset_priority=['search_set'],
            fallback_to_legacy=True,
        )
        if resolved:
            page_state['filters'] = resolved
        return page_state.get('filters')

    def set_page_filters(filters):
        page_state['filters'] = filters
        _update_cache_toggle_state()

    filter_editor = build_filter_editor(
        get_page_filters,
        set_page_filters,
        vm,
        dataset_priority=['search_set'],
        fallback_to_legacy=True,
    )

    def _update_button_state():
        """Enable/disable create button based on whether we have a valid token."""
        has_token = vm.eng and vm.eng.active_client and vm.eng.active_client.token
        btn_create.disabled = not has_token
        auth_warning.visible = not has_token
        
        if not has_token:
            log.warning("Database creation disabled: No authentication token available")

    async def _run_creation_process(use_cache=False):
        db_dir = vm.config.db_path.expanduser()
        filters = get_page_filters()
        filter_hash = _hash_filters(filters)
        filelist_path = db_dir / f"_source_arids_{filter_hash}.json"
        btn_create.disabled = True
        progress_bar.visible = True
        progress_bar.value = 0
        status_display.object = f"🚀 Starting database creation in `{db_dir}`...\n\n"
        if use_cache:
            status_display.object += "📦 Using cached recording list for current filters.\n\n"
        else:
            status_display.object += "🌐 Fetching a fresh recording list for current filters.\n\n"
        
        try:
            status_display.object += f"📥 Downloading and processing parquet files...\n"
            status_display.object += f"⏳ This may take several minutes. Watch the logs below for progress.\n\n"

            def _append_progress_line(msg: str):
                ts = datetime.now().strftime("%H:%M:%S")
                progress_lines.append(f"- `{ts}` {msg}")
                if len(progress_lines) > 20:
                    del progress_lines[:-20]
                status_display.object = (
                    status_display.object.split("\n\n### Progress\n", 1)[0]
                    + "\n\n### Live Progress\n"
                    + "\n".join(progress_lines)
                )

            def _progress_callback(event: dict):
                def _apply_update():
                    stage = event.get("stage")
                    if stage == "selection_start":
                        max_files_msg = event.get("max_files")
                        cache_mode = "cache" if event.get("use_cache") else "fresh fetch"
                        _append_progress_line(f"Selecting recordings (max_files={max_files_msg}, mode={cache_mode})")
                    elif stage == "selection_complete":
                        _append_progress_line(f"Selected {event.get('selected_recordings', 0)} recordings")
                    elif stage == "ingest_start":
                        total_candidates = event.get("total_candidates")
                        max_count = event.get("max_count")
                        current_embeddings = event.get("current_embeddings")
                        if total_candidates is not None:
                            _append_progress_line(f"Starting ingestion for {total_candidates} candidate recordings")
                        else:
                            _append_progress_line("Starting ingestion")
                        if max_count and max_count > 0 and current_embeddings is not None:
                            _append_progress_line(
                                f"Current DB size is {current_embeddings} embeddings (limit {max_count})"
                            )
                    elif stage == "source_progress":
                        candidates_seen = event.get("candidates_seen", 0)
                        total_candidates = event.get("total_candidates", 0)
                        skipped_existing = event.get("skipped_existing", 0)
                        files_processed = event.get("files_processed", 0)
                        if total_candidates:
                            _append_progress_line(
                                f"Checked {candidates_seen}/{total_candidates} recordings "
                                f"(downloaded {files_processed}, skipped existing {skipped_existing})"
                            )
                            progress_bar.value = int(min(100, (candidates_seen / total_candidates) * 100))
                        else:
                            _append_progress_line(
                                f"Checked {candidates_seen} recordings "
                                f"(downloaded {files_processed}, skipped existing {skipped_existing})"
                            )
                    elif stage == "ingest_progress":
                        files_processed = event.get("files_processed", 0)
                        total_candidates = event.get("total_candidates")
                        added_this_file = event.get("added_this_file", 0)
                        total_inserted = event.get("total_inserted", 0)
                        if total_candidates:
                            _append_progress_line(
                                f"Processed {files_processed}/{total_candidates} files (+{added_this_file}, total added {total_inserted})"
                            )
                            progress_bar.value = int(min(100, (files_processed / total_candidates) * 100))
                        else:
                            _append_progress_line(
                                f"Processed {files_processed} files (+{added_this_file}, total added {total_inserted})"
                            )
                    elif stage == "ingest_complete":
                        if event.get("stop_reason") == "max_count_already_reached":
                            _append_progress_line(
                                f"Skipped ingestion: DB already at limit ({event.get('final_embeddings', 0)} embeddings)"
                            )
                        else:
                            _append_progress_line(
                                f"Ingestion complete: {event.get('files_processed', 0)} files, {event.get('total_inserted', 0)} embeddings added"
                            )

                if pn.state.curdoc is not None:
                    pn.state.curdoc.add_next_tick_callback(_apply_update)
                else:
                    _apply_update()
            
            log.info(f"Starting database creation: max_files={max_files.value}, max_embeddings={max_embeddings.value}")
            
            with patch_tqdm.patch_tqdm(progress_bar):
                await asyncio.to_thread(
                    create_database_from_baw, 
                    db_path=db_dir, 
                    filelist_path=filelist_path,
                    client=vm.eng.active_client if vm.eng else None, 
                    filter=filters[0],
                    max_files=max_files.value, 
                    max_embeddings=max_embeddings.value,
                    use_cache=use_cache,
                    progress_callback=_progress_callback,
                )
            status_display.object += "\n\n🎉 **Database creation complete!**"
            vm.notify("Database created successfully!", "success")
            log.info("Database creation completed successfully")
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            status_display.object = f"### 💥 Error:\n```\n{e}\n\nFull traceback:\n{tb}\n```"
            vm.notify(f"DB creation failed: {e}", "error")
            log.error(f"Database creation failed: {e}", exc_info=True)
        finally:
            btn_create.disabled = False
            progress_bar.visible = False

    async def _create_db(_):
        db_dir = vm.config.db_path.expanduser()
        if db_dir.exists() and any(db_dir.iterdir()) and not list(db_dir.glob("_source_arids_*.json")):
            log.warning(f"Directory '{db_dir}' is not empty.")
             #return vm.notify(f"Directory '{db_dir}' is not empty.", "error")
        filters = get_page_filters()
        if not filters or not filters[0]: return vm.notify("Please define at least one filter.", "warn")
        cached_file = _cached_file_for_filters(filters)
        selected_use_cache = bool(use_cached_list.value and cached_file is not None and cached_file.exists())
        await _run_creation_process(use_cache=selected_use_cache)
    
    @pn.depends(vm.obs.ensure("config_state").param.version, watch=True)
    def _update_db_path_display(_):
        """Reacts to config changes and updates the static text."""
        if vm.config and vm.config.db_path:
            db_path_display.value = str(vm.config.db_path)
        else:
            # This case shouldn't happen due to the check at the top, but it's safe to handle
            db_path_display.value = "Not set in config."
        _update_cache_toggle_state()

    # Watch for config changes instead of engine changes
    @pn.depends(vm.obs.ensure("config_state").param.version, watch=True)
    @pn.depends(vm.obs.ensure("engine_state").param.version, watch=True)
    def _update_on_config_change(_):
        """Reacts to config changes and updates button availability."""
        _update_button_state()

        # Rebuild to ensure the editor reflects latest metadata when engine/db state changes.
        filter_editor.objects = build_filter_editor(
            get_page_filters,
            set_page_filters,
            vm,
            dataset_priority=['search_set'],
            fallback_to_legacy=True,
        ).objects

    btn_create.on_click(_create_db)
    
    # Initialize button state
    _update_button_state()
    _update_cache_toggle_state()

    return pn.Column(
        pn.Row("## Create New Database from BAW", pn.Spacer(), database_help_btn),
        pn.pane.Alert("This tool will build a new database in the directory specified in your current project configuration.", alert_type="info"),
        auth_warning,  # Show auth warning when no token
        pn.Row(
            pn.Column(
                pn.Row("Creation Settings", pn.Spacer(), db_settings_help_btn),
                db_path_display,
                pn.Row(max_files, max_embeddings, sizing_mode="stretch_width"),
                use_cached_list,
                cache_hint,
                sizing_mode="stretch_width",
            ),
            pn.Column(
                pn.Row("Recording Filters", pn.Spacer(), recordings_filter_help_btn),
                filter_editor,
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        ),
        pn.layout.Divider(),
        pn.Row(btn_create, align="start"),
        progress_bar,
        status_display,
        sizing_mode="stretch_width"
    )