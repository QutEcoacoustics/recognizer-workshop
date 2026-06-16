import asyncio
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import panel as pn
from panel_material_ui import Dialog

from recognizer_workshop.app.create_database.database_creation_task import (
    DatabaseCreationTask,
    format_progress_event,
)
from recognizer_workshop.app.services.database_status import (
    database_exists_at,
    format_size,
    USEARCH_INDEX_FILENAME,
)
from recognizer_workshop.ui.panel.components.native.filter_editor import FilterEditor
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

    page_state: dict = {'task': None}
    max_files = pn.widgets.IntInput(name="Max Recordings to Scan", value=1000, start=-1)
    max_embeddings = pn.widgets.IntInput(name="Max Total Embeddings in DB", value=200_000, start=-1)
    use_cached_list = pn.widgets.Checkbox(name="Use Cached Recording List", value=True, visible=False)
    cache_hint = pn.pane.Markdown("", visible=False)
    btn_create = pn.widgets.Button(name="Create Database", button_type="primary", icon="database")
    btn_stop = pn.widgets.Button(name="⏹ Stop", button_type="danger", visible=False)
    progress_bar = pn.widgets.Progress(name='Progress', active=True, visible=False, sizing_mode="stretch_width")
    status_pane = pn.pane.Markdown("")
    progress_pane = pn.pane.Markdown("", visible=False)
    
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
        cached_file = _cached_file_for_filters(filter_editor.value)
        has_cache = cached_file is not None and cached_file.exists()
        use_cached_list.visible = has_cache
        cache_hint.visible = has_cache
        if has_cache:
            cache_hint.object = f"ℹ️ Cached recording list found: `{cached_file.name}`"
        else:
            cache_hint.object = ""

    # The filter editor owns the recording filter. Read it via `filter_editor.value`;
    # it seeds itself from existing DB metadata and refreshes the cache toggle on save.
    filter_editor = FilterEditor(
        vm,
        dataset_priority=['search_set'],
        fallback_to_legacy=True,
        on_save=lambda _f: _update_cache_toggle_state(),
    )

    def _db_exists() -> bool:
        """Return True if a database already exists at the configured path."""
        if not vm.config or not vm.config.db_path:
            return False
        return database_exists_at(vm.config.db_path)

    def _update_button_state():
        """Enable/disable create button and update its label based on DB existence and token."""
        has_token = vm.eng and vm.eng.active_client and vm.eng.active_client.token
        btn_create.disabled = not has_token
        auth_warning.visible = not has_token

        btn_create.name = "Add Embeddings to Database" if _db_exists() else "Create Database"

        if not has_token:
            log.warning("Database creation disabled: No authentication token available")

    async def _run_creation_process(use_cache=False):
        db_dir = vm.config.db_path.expanduser()
        filters = filter_editor.value
        filter_hash = _hash_filters(filters)
        filelist_path = db_dir / f"_source_arids_{filter_hash}.json"
        btn_create.disabled = True
        progress_bar.visible = True
        progress_bar.value = 0
        progress_lines: list[str] = []
        cache_msg = "📦 Using cached recording list." if use_cache else "🌐 Fetching a fresh recording list."
        status_pane.object = (
            f"🚀 Starting database creation in `{db_dir}`...\n\n"
            f"{cache_msg}\n\n"
            f"📥 Downloading and processing parquet files...\n"
            f"⏳ This may take several minutes.\n"
        )
        progress_pane.object = ""
        progress_pane.visible = True

        try:
            def _append_progress_line(msg: str):
                ts = datetime.now().strftime("%H:%M:%S")
                progress_lines.append(f"- `{ts}` {msg}")
                if len(progress_lines) > 20:
                    del progress_lines[:-20]
                progress_pane.object = "### Live Progress\n" + "\n".join(progress_lines)

            def _progress_callback(event: dict):
                messages, pct = format_progress_event(event)

                def _apply_update():
                    for msg in messages:
                        _append_progress_line(msg)
                    if pct is not None:
                        progress_bar.value = pct

                if pn.state.curdoc is not None:
                    pn.state.curdoc.add_next_tick_callback(_apply_update)
                else:
                    _apply_update()
            log.info("Starting database creation: max_files=%s, max_embeddings=%s", max_files.value, max_embeddings.value)

            task = DatabaseCreationTask(
                db_path=db_dir,
                filelist_path=filelist_path,
                client=vm.eng.active_client if vm.eng else None,
                filter=filters[0],
                max_files=max_files.value,
                max_embeddings=max_embeddings.value,
                use_cache=use_cache,
            )
            page_state['task'] = task
            btn_stop.visible = True
            btn_stop.disabled = False
            btn_stop.name = "⏹ Stop"

            with patch_tqdm.patch_tqdm(progress_bar):
                await asyncio.to_thread(task.run, _progress_callback)

            if task.was_stopped:
                status_pane.object += "\n\n⏹ **Database creation stopped.** Partial data has been safely committed."
                vm.notify("Database creation stopped.", "warning")
            else:
                status_pane.object += "\n\n🎉 **Database creation complete!**"
                vm.notify("Database created successfully!", "success")
            log.info("Database creation completed successfully")
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            status_pane.object = f"### 💥 Error:\n```\n{e}\n\nFull traceback:\n{tb}\n```"
            vm.notify(f"DB creation failed: {e}", "error")
            log.error(f"Database creation failed: {e}", exc_info=True)
        finally:
            btn_stop.visible = False
            page_state['task'] = None
            progress_bar.visible = False
            progress_pane.visible = False
            _update_button_state()  # re-check: DB may now exist

    async def _create_db(_):
        db_dir = vm.config.db_path.expanduser()
        if db_dir.exists() and any(db_dir.iterdir()) and not list(db_dir.glob("_source_arids_*.json")):
            log.warning(f"Directory '{db_dir}' is not empty.")
             #return vm.notify(f"Directory '{db_dir}' is not empty.", "error")
        filters = filter_editor.value
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
        _update_button_state()

    # Watch for config changes instead of engine changes
    @pn.depends(vm.obs.ensure("config_state").param.version, watch=True)
    @pn.depends(vm.obs.ensure("engine_state").param.version, watch=True)
    def _update_on_config_change(_):
        """Reacts to config changes and updates button availability."""
        _update_button_state()

        # Seed the editor from the latest DB metadata when engine/db state changes.
        # This only fills the value if the user hasn't already set one.
        filter_editor.seed_from_db()

        # Re-render the status box and button visibility for the (re)loaded DB.
        _refresh_status_section()

    btn_create.on_click(_create_db)

    def _on_stop(_):
        task = page_state.get('task')
        if task is not None:
            task.stop()
            btn_stop.name = "⏹ Stopping..."
            btn_stop.disabled = True

    btn_stop.on_click(_on_stop)
    
    # Initialize button state
    _update_button_state()
    _update_cache_toggle_state()

    # --- Health Check ---
    dialogs_container = pn.Column()

    btn_health_check = pn.widgets.Button(name="Check DB Health", button_type="warning", icon="stethoscope")

    # The dark status box: always shows the lightweight status (embeddings, index
    # size, datasets, label counts). After a health check it also shows the
    # consistency verdict. The dark background comes from the ``` code fence.
    status_box = pn.pane.Markdown("", sizing_mode="stretch_width")

    # --- Repair Index ---
    btn_repair = pn.widgets.Button(name="Repair Index", button_type="danger", icon="tool", visible=False)
    repair_status = pn.pane.Markdown("", sizing_mode="stretch_width")

    # --- Delete Index ---
    btn_delete_index = pn.widgets.Button(
        name="Delete Usearch Index",
        button_type="danger",
        icon="trash",
    )
    delete_index_status = pn.pane.Markdown("", sizing_mode="stretch_width")

    def _update_action_buttons(status):
        """Set visibility of the three action buttons from the current status.

        - Check Health: visible whenever a database is loaded.
        - Delete Index: visible only when the usearch index file is present.
        - Repair Index: visible whenever the status is known to be unhealthy
          (``is_healthy is False``): index missing, count mismatch, or a health
          check that found missing embeddings.
        """
        loaded = bool(vm.eng and vm.eng.db)
        index_present = bool(status and status.index_present)

        btn_health_check.visible = loaded
        btn_delete_index.visible = index_present
        btn_repair.visible = bool(status) and status.is_healthy is False

    def _refresh_status_section():
        """Render the lightweight status box and refresh button visibility.

        Called on page build, engine/DB state change, and after delete/repair.
        """
        if not (vm.eng and vm.eng.db):
            status_box.object = "_No database loaded._"
            _update_action_buttons(None)
            return
        try:
            status = vm.eng.get_database_status()
        except Exception as e:
            status_box.object = f"_(Could not read database status: {e})_"
            _update_action_buttons(None)
            return
        status_box.object = f"```\n{status}\n```"
        _update_action_buttons(status)

    async def _run_health_check(_):
        if not (vm.eng and vm.eng.db):
            status_box.object = "_No database loaded. Initialize the engine on the Config tab first._"
            return
        btn_health_check.disabled = True
        btn_health_check.name = "Checking..."
        repair_status.object = ""
        try:
            status = await asyncio.get_running_loop().run_in_executor(
                None, lambda: vm.eng.get_database_status(include_health=True)
            )
            status_box.object = f"```\n{status}\n```"
            _update_action_buttons(status)
        except Exception as e:
            status_box.object = f"❌ Health check failed: {e}"
        finally:
            btn_health_check.disabled = False
            btn_health_check.name = "Check DB Health"

    btn_health_check.on_click(_run_health_check)

    def _on_delete_index(_):
        """Show a confirmation dialog before deleting the usearch index."""
        if not vm.eng or not vm.eng.db:
            vm.notify("No database loaded.", "error")
            return

        try:
            status = vm.eng.get_database_status()
        except Exception as e:
            vm.notify(f"Could not read index info: {e}", "error")
            return

        if not status.index_present:
            vm.notify("Usearch index file does not exist.", "warn")
            return

        btn_cancel = pn.widgets.Button(name="Cancel", button_type="default")
        btn_confirm = pn.widgets.Button(name="Yes, delete it", button_type="danger")

        dialog_content = pn.Column(
            pn.pane.Markdown(
                f"### Delete usearch index?\n\n"
                f"This will permanently delete the usearch index file "
                f"(`{USEARCH_INDEX_FILENAME}`, **{status.index_size_str()}**) from disk.\n\n"
                f"The SQLite tables are untouched. You can rebuild the index "
                f"afterwards using **Repair Index**."
            ),
            pn.Row(btn_cancel, btn_confirm),
        )

        dialog = Dialog(dialog_content, title="Confirm index deletion")
        dialogs_container[:] = [dialog]
        dialog.open = True

        def _cancel(_):
            dialog.open = False
            dialogs_container.clear()

        async def _confirm(_):
            dialog.open = False
            dialogs_container.clear()
            btn_delete_index.disabled = True
            delete_index_status.object = "⏳ Deleting index..."
            try:
                freed_bytes = await asyncio.get_running_loop().run_in_executor(
                    None, vm.eng.delete_usearch_index
                )
                delete_index_status.object = (
                    f"🗑 Index deleted ({format_size(freed_bytes)} freed). "
                    f"Use **Repair Index** to rebuild it."
                )
                _refresh_status_section()
                vm.notify("Usearch index deleted.", "success")
            except Exception as e:
                delete_index_status.object = f"❌ Delete failed: {e}"
                vm.notify(f"Delete failed: {e}", "error")
            finally:
                btn_delete_index.disabled = False

        btn_cancel.on_click(_cancel)
        btn_confirm.on_click(_confirm)

    btn_delete_index.on_click(_on_delete_index)

    async def _run_repair(_):
        btn_repair.disabled = True
        btn_repair.name = "Repairing..."
        repair_status.object = "⏳ Downloading parquet files and rebuilding index..."

        def _progress_callback(event):
            stage = event.get("stage", "")
            if stage in ("starting", "complete"):
                return
            files_done = event.get("files_processed", 0)
            total_files = event.get("total_files", 0)
            added = event.get("embeddings_added", 0)
            expected = event.get("embeddings_expected", 0)
            skipped = event.get("embeddings_skipped", 0)
            errs = event.get("errors", 0)
            text = (
                f"⏳ **Repairing index...**\n\n"
                f"- Files processed: {files_done} / {total_files}\n"
                f"- Embeddings added: {added} / {expected}\n"
                f"- Skipped (already present): {skipped}\n"
                f"- Errors: {errs}"
            )
            def _apply():
                repair_status.object = text
            try:
                doc = pn.state.curdoc
                if doc and doc.session_context:
                    doc.add_next_tick_callback(_apply)
                else:
                    _apply()
            except Exception:
                _apply()

        try:
            stats = await asyncio.get_running_loop().run_in_executor(
                None, lambda: vm.eng.repair_db_index(progress_callback=_progress_callback)
            )
            lines = [
                "**Repair complete:**",
                f"- Sources processed: {stats['sources']}",
                f"- Embeddings added: {stats['embeddings_added']}",
                f"- Embeddings skipped (already present): {stats['embeddings_skipped']}",
                f"- Errors: {stats['errors']}",
            ]
            repair_status.object = "\n".join(lines)
            await _run_health_check(None)
        except Exception as e:
            repair_status.object = f"❌ Repair failed: {e}"
            _refresh_status_section()
        finally:
            btn_repair.disabled = False
            btn_repair.name = "Repair Index"

    btn_repair.on_click(_run_repair)

    # Populate the status box and button visibility on load.
    _refresh_status_section()

    return pn.Column(
        pn.Row("## Build Database", pn.Spacer(), database_help_btn),
        auth_warning,  # Show auth warning when no token
        pn.Row(
            pn.Column(
                db_path_display,
                pn.Row(max_files, max_embeddings, sizing_mode="stretch_width"),
                use_cached_list,
                cache_hint,
                sizing_mode="stretch_width",
            ),
            pn.Column(
                filter_editor.view,
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        ),
        pn.layout.Divider(),
        pn.Row(btn_create, btn_stop, align="start"),
        progress_bar,
        status_pane,
        progress_pane,
        pn.layout.Divider(),
        pn.Row("## Database Status"),
        pn.Row(
            pn.Column(
                status_box,
                repair_status,
                delete_index_status,
                sizing_mode="stretch_width",
                styles={"flex": "3"},
            ),
            pn.Column(
                btn_health_check,
                btn_delete_index,
                btn_repair,
                sizing_mode="stretch_width",
                styles={"flex": "1"},
            ),
            sizing_mode="stretch_width",
        ),
        dialogs_container,
        sizing_mode="stretch_width"
    )