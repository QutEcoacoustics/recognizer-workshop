import asyncio
import hashlib
import json
import logging
from pathlib import Path

import panel as pn
from panel_material_ui import Dialog

from recognizer_workshop.app.create_database.create_database import create_database_from_baw
from recognizer_workshop.ui.panel.components.native.filter_editor import build_filter_editor
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
    max_embeddings = pn.widgets.IntInput(name="Max Embeddings to Ingest", value=200_000, start=-1)
    btn_create = pn.widgets.Button(name="Create Database", button_type="primary", icon="database")
    progress_bar = pn.widgets.Progress(name='Progress', active=True, visible=False, sizing_mode="stretch_width")
    status_display = pn.pane.Markdown("")  # Renamed from 'log' to avoid confusion
    dialog_container = pn.Column()
    
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

    def get_page_filters(): return page_state.get('filters')
    def set_page_filters(filters): page_state['filters'] = filters
    filter_editor = build_filter_editor(get_page_filters, set_page_filters, vm)

    def _update_button_state():
        """Enable/disable create button based on whether we have a valid token."""
        has_token = vm.eng and vm.eng.baw_api and vm.eng.baw_api.token
        btn_create.disabled = not has_token
        auth_warning.visible = not has_token
        
        if not has_token:
            log.warning("Database creation disabled: No authentication token available")

    def _hash_filters(filters):
        if not filters or not filters[0]: return None
        filter_str = json.dumps(filters[0], sort_keys=True).encode('utf-8')
        return hashlib.sha256(filter_str).hexdigest()[:10]
    async def _run_creation_process(use_cache=False):
        db_dir = vm.config.db_path.expanduser()
        filters = get_page_filters()
        filter_hash = _hash_filters(filters)
        filelist_path = db_dir / f"_source_arids_{filter_hash}.json"
        btn_create.disabled = True
        progress_bar.visible = True
        progress_bar.value = 0
        status_display.object = f"🚀 Starting database creation in `{db_dir}`...\n\n"
        
        try:
            status_display.object += f"📥 Downloading and processing parquet files...\n"
            status_display.object += f"⏳ This may take several minutes. Watch the logs below for progress.\n\n"
            
            log.info(f"Starting database creation: max_files={max_files.value}, max_embeddings={max_embeddings.value}")
            
            with patch_tqdm.patch_tqdm(progress_bar):
                await asyncio.to_thread(
                    create_database_from_baw, 
                    db_path=db_dir, 
                    filelist_path=filelist_path,
                    client=vm.eng.baw_api if vm.eng else None, 
                    filter=filters[0],
                    max_files=max_files.value, 
                    max_embeddings=max_embeddings.value
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
            dialog_container.clear()

    async def _create_db(_):
        db_dir = vm.config.db_path.expanduser()
        if db_dir.exists() and any(db_dir.iterdir()) and not list(db_dir.glob("_source_arids_*.json")):
            log.warning(f"Directory '{db_dir}' is not empty.")
             #return vm.notify(f"Directory '{db_dir}' is not empty.", "error")
        filters = get_page_filters()
        if not filters or not filters[0]: return vm.notify("Please define at least one filter.", "warn")
        filter_hash = _hash_filters(filters)
        cached_file = db_dir / f"_source_arids_{filter_hash}.json"
        if cached_file.exists():
            btn_use_cache = pn.widgets.Button(name="Use Cached List", button_type="primary")
            btn_fetch_new = pn.widgets.Button(name="Fetch New List")
            btn_use_cache.on_click(lambda _: asyncio.create_task(_run_creation_process(use_cache=True)))
            btn_fetch_new.on_click(lambda _: asyncio.create_task(_run_creation_process(use_cache=False)))
            dialog = Dialog(pn.Column(pn.pane.Markdown(f"Found a cached file list for these filters. Use it or fetch a new list?"), pn.Row(btn_use_cache, btn_fetch_new)), title="Cached File List Found")
            dialog_container.objects = [dialog]; dialog.open = True
        else:
            await _run_creation_process(use_cache=False)
    
    @pn.depends(vm.obs.ensure("config_state").param.version, watch=True)
    def _update_db_path_display(_):
        """Reacts to config changes and updates the static text."""
        if vm.config and vm.config.db_path:
            db_path_display.value = str(vm.config.db_path)
        else:
            # This case shouldn't happen due to the check at the top, but it's safe to handle
            db_path_display.value = "Not set in config."

    # Watch for config changes instead of engine changes
    @pn.depends(vm.obs.ensure("config_state").param.version, watch=True)
    def _update_on_config_change(_):
        """Reacts to config changes and updates button availability."""
        _update_button_state()

    btn_create.on_click(_create_db)
    
    # Initialize button state
    _update_button_state()

    settings_card = pn.Card(
        db_path_display, # Use the widget we defined
        pn.Row(max_files, max_embeddings),
        title="Creation Settings"
    )

    return pn.Column(
        pn.Row("## Create New Database from BAW", pn.Spacer(), database_help_btn),
        pn.pane.Alert("This tool will build a new database in the directory specified in your current project configuration.", alert_type="info"),
        auth_warning,  # Show auth warning when no token
        pn.Row(
            pn.Column(
                pn.Row("Creation Settings", pn.Spacer(), db_settings_help_btn),
                settings_card
            )
        ),
        pn.Row(
            pn.Column(
                pn.Row("Recording Filters", pn.Spacer(), recordings_filter_help_btn),
                filter_editor
            )
        ),
        pn.layout.Divider(),
        pn.Row(btn_create),
        dialog_container,
        progress_bar,
        status_display,
        sizing_mode="stretch_width"
    )