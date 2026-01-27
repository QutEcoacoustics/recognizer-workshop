"""Configuration page for the recognizer workshop UI."""

from functools import partial
from pathlib import Path
import logging
import panel as pn

from recognizer_workshop.app.agile_engine import AgileConfig
from recognizer_workshop.app.agile_engine import agile_state
from recognizer_workshop.app.services.app_data import get_recent_files
from recognizer_workshop.ui.panel.utils.delayed_init import run_after_render

log = logging.getLogger(__name__)

# TODO: tidy this up
try:
    from panel_material_ui import Dialog
except ImportError:
    # Provide a helpful error if the library isn't installed.
    raise ImportError(
        "The panel-material-ui library is not installed. "
        "Please run 'pip install panel-material-ui' in your environment."
    )


def build(vm):
    """Build the configuration page UI.
    
    Args:
        vm: The view model containing application state
        
    Returns:
        Panel Column containing the configuration interface
    """

    print("[DEBUG] Building configuration page...")

    dialogs_container = pn.Column()  

    recent_files = get_recent_files()
    if recent_files:
        try:
            # Try to load the most recent file
            vm.config = AgileConfig.load(recent_files[0])
            is_new_unsaved_config = False
            should_auto_start_engine = True
        except FileNotFoundError:
            # If the recent file was moved/deleted, start with a new one
            vm.config = AgileConfig()
            is_new_unsaved_config = True
            should_auto_start_engine = False
    else:
        # If no recent files, start with a new, default config
        vm.config = AgileConfig()
        is_new_unsaved_config = True
        should_auto_start_engine = False
    
    save_status = pn.pane.Markdown("")

    btn_new = pn.widgets.Button(name="New", icon='new-document')
    btn_open = pn.widgets.Button(name="Open...", icon='folder-open')
    btn_save = pn.widgets.Button(name="Save", icon='save') # maybe not necessary, because we autosave
    btn_save_as = pn.widgets.Button(name="Save As...")
    current_config_path_display = pn.widgets.StaticText()

    btn_start_engine = pn.widgets.Button(name="🚀 Start Engine", button_type="primary")
    
    def update_engine_button():
        """Update the engine button text and state based on current engine status."""
        if vm.eng:
            btn_start_engine.name = "🔄 Restart Engine"
            btn_start_engine.button_type = "warning"
        else:
            btn_start_engine.name = "🚀 Start Engine"
            btn_start_engine.button_type = "primary"

        # button is disabled until a config path is set
        btn_start_engine.disabled = not vm.config.config_path

    # Label editor populated only after the engine has started successfully.
    labels_editor_placeholder = pn.Column(sizing_mode="stretch_width")

    # Default values for new configurations
    default_paths = {
        'db_path': Path("db"),
        'labeled_examples_folder': Path("labeled_examples"),
        'models_folder': Path("models"),
        'predictions_folder': Path("predictions"),
    }
    
    default_api_domain = "https://api.ecosounds.org"

    # text inputs for config fields, with values coming from the loaded config
    annotator_id = pn.widgets.TextInput(name="Annotator", value=vm.config.annotator_id)
    domain = pn.widgets.TextInput(name="Workbench API domain", value=vm.config.baw_config.get("domain", ""))
    auth_token = pn.widgets.TextInput(name="Workbench API token", value=vm.config.baw_config.get("auth_token", ""))

    # Create help buttons for key fields
    annotator_help_btn = vm.help_system.create_help_button("help-annotator-id", "Help: Annotator ID")
    domain_help_btn = vm.help_system.create_help_button("help-api-domain", "Help: API Domain")
    token_help_btn = vm.help_system.create_help_button("help-api-token", "Help: API Token")
    folder_help_btn = vm.help_system.create_help_button("help-folder-config", "Help: Folder Configuration")
    engine_help_btn = vm.help_system.create_help_button("help-engine", "Help: Engine")
    project_help_btn = vm.help_system.create_help_button("help-project-file", "Help: Project Files")
    labels_help_btn = vm.help_system.create_help_button("help-target-labels", "Help: Target Labels")

    # Advanced path configuration (hidden by default)
    db_path = pn.widgets.TextInput(name="DB path (Directory)", value=str(vm.config.db_path or ""))
    labeled_dir = pn.widgets.TextInput(name="Labeled examples folder", value=str(vm.config.labeled_examples_folder or ""))
    models_dir = pn.widgets.TextInput(name="Models folder", value=str(vm.config.models_folder or ""))
    preds_dir = pn.widgets.TextInput(name="Predictions folder", value=str(vm.config.predictions_folder or ""))
    
    # Advanced section toggle
    show_advanced = pn.widgets.Checkbox(name="Show Advanced Path Settings", value=False)
    advanced_section = pn.Column(visible=False, sizing_mode="stretch_width")

    def _print_paths(prefix=""):
        log.debug(f'{prefix} Config paths: {vm.config.db_path}, {vm.config.labeled_examples_folder}, {vm.config.models_folder}, {vm.config.predictions_folder}')

    def set_default_paths_if_empty():
        """Set default relative paths and other default values if they're not already set."""
        # Set relative paths as defaults - these will be resolved to absolute when saved/loaded
        if not vm.config.db_path:
            vm.config.db_path = default_paths['db_path']
        if not vm.config.labeled_examples_folder:
            vm.config.labeled_examples_folder = default_paths['labeled_examples_folder']
        if not vm.config.models_folder:
            vm.config.models_folder = default_paths['models_folder']
        if not vm.config.predictions_folder:
            vm.config.predictions_folder = default_paths['predictions_folder']
            
        # Set default API domain if not already set
        if not vm.config.baw_config.get("domain"):
            vm.config.baw_config["domain"] = default_api_domain

    def toggle_advanced_visibility(*_):
        """Toggle visibility of advanced path settings."""
        advanced_section.visible = show_advanced.value

    def _has_custom_paths():
        """Check if any custom paths are set in the config."""
        return bool(
            str(vm.config.db_path) == str(default_paths['db_path']) or
            str(vm.config.labeled_examples_folder) == str(default_paths['labeled_examples_folder']) or
            str(vm.config.models_folder) == str(default_paths['models_folder']) or
            str(vm.config.predictions_folder) == str(default_paths['predictions_folder'])
        )

    def sync_config_to_ui():
        """Updates all UI widgets from the current vm.config object."""
        # Use a flag to prevent autosave feedback loop
        if hasattr(sync_config_to_ui, '_syncing'):
            return  # Prevent recursive calls
        vm.obs.bump("config_state")
        
        sync_config_to_ui._syncing = True
        try:
            annotator_id.value = vm.config.annotator_id
            domain.value = vm.config.baw_config.get("domain", default_api_domain)
            auth_token.value = vm.config.baw_config.get("auth_token", "")
            _print_paths("1.")
            
            # Capture all path values at once to avoid side effects from individual access
            db_path_val = str(vm.config.db_path or "")
            labeled_dir_val = str(vm.config.labeled_examples_folder or "")
            models_dir_val = str(vm.config.models_folder or "")
            preds_dir_val = str(vm.config.predictions_folder or "")
            _print_paths("2.")
            
            # Update advanced path fields with captured values
            db_path.value = db_path_val
            labeled_dir.value = labeled_dir_val
            models_dir.value = models_dir_val
            preds_dir.value = preds_dir_val
            
            # Show advanced section if any paths are customized (non-default)
            has_custom_paths = _has_custom_paths()
            show_advanced.value = has_custom_paths
            advanced_section.visible = has_custom_paths
            
            path_str = str(vm.config.config_path) if vm.config.config_path else "Unsaved Configuration"
            current_config_path_display.value = f"<b>Current File:</b> {path_str}"
            update_engine_button()
        finally:
            del sync_config_to_ui._syncing

    def sync_ui_to_config():
        """Updates the vm.config object from all UI widgets."""
        # will trigger watchers of the config_state
        vm.obs.bump("config_state")
        vm.config.annotator_id = annotator_id.value
        vm.config.baw_config["domain"] = domain.value
        vm.config.baw_config["auth_token"] = auth_token.value
        
        # Always update paths from widgets (they contain default values when not advanced)
        vm.config.db_path = Path(db_path.value) if db_path.value else None
        vm.config.labeled_examples_folder = Path(labeled_dir.value) if labeled_dir.value else None
        vm.config.models_folder = Path(models_dir.value) if models_dir.value else None
        vm.config.predictions_folder = Path(preds_dir.value) if preds_dir.value else None


    def _autosave(*_events):
        """Syncs UI to the config object and saves it if a config path exists."""
        # Don't autosave if we're currently syncing config to UI (prevents feedback loop)
        if hasattr(sync_config_to_ui, '_syncing'):
            return
        
        if not vm.config.config_path:
            vm.notify("Autosave skipped: No configuration file path set.", "warn")
            return  # No path to autosave to

        # will trigger watchers of the config_state
        vm.obs.bump("config_state")

        sync_ui_to_config()
        if vm.config.config_path:
            try:
                vm.config.save()
                vm.notify("Autosaved.", "info")
            except (FileNotFoundError, PermissionError, OSError) as e:
                vm.notify(f"Autosave failed: {e}", "error")



    def _on_new_click(_event):
        """Resets the current configuration to a blank, unsaved state."""
        vm.config.reset()
        set_default_paths_if_empty()  # Set default relative paths for new config
        sync_config_to_ui()
        vm.notify("New configuration created. Fill in the details and 'Save As'.", "info")


    def _on_open_click(_):
        """Opens a dialog to select a configuration file to load."""
        if Dialog is None: return vm.notify("panel-material-ui not installed for file dialog.", "warn")
        
        recent_files = get_recent_files()
        file_selector = pn.widgets.FileSelector(Path.home(), only_files=True)
        recent_selector = pn.widgets.Select(name="Recent Files", options=recent_files)
        
        dialog_content = pn.Column(
            "### Open from disk", 
            file_selector, 
            "### Open recent", 
            recent_selector
        )


        btn_cancel = pn.widgets.Button(name="Cancel", button_type="danger")
        btn_open   = pn.widgets.Button(name="Open",   button_type="primary")

        def _on_cancel(event):
            dialog.open = False


        def _on_open(event):
            path = recent_selector.value or (file_selector.value[0] if file_selector.value else None)
            if path:
                try:
                    vm.config = AgileConfig.load(path)
                    sync_config_to_ui()
                    vm.notify(f"Loaded config from {Path(path).name}", "success")
                    # Auto-start engine after loading config
                    _make_engine(None)
                except Exception as ex:
                    vm.notify(f"Failed to load config: {ex}", "error")
            dialog.open = False

        
        btn_cancel.on_click(_on_cancel)
        btn_open.on_click(_on_open)

        dialog = Dialog(
            pn.Column(dialog_content, 
                      pn.Row(btn_cancel, btn_open)),
            title="Open Configuration File"
        )
        dialogs_container[:] = [dialog]
        dialog.open = True


    def _on_save_as_click(_):
        """Save current config to a chosen path."""
        if Dialog is None:
            return vm.notify("panel-material-ui not installed for file dialog.", "warn")

        # Directory picker + filename input makes "Save As" UX clearer.
        dir_selector = pn.widgets.FileSelector(Path.home(), only_files=False)
        filename_input = pn.widgets.TextInput(name="Filename", value="agile_config.json")

        btn_cancel = pn.widgets.Button(name="Cancel", button_type="danger")
        btn_save   = pn.widgets.Button(name="Save",   button_type="primary")

        def _on_cancel(event):
            dialog.open = False
            dialogs_container.clear()

        def _on_save(event):
            # Prefer selected directory (first selected item). If a file is selected, use its parent.
            if dir_selector.value:
                selected = Path(dir_selector.value[0])
                save_dir = selected if selected.is_dir() else selected.parent
            else:
                save_dir = Path.home()

            save_path = save_dir / (filename_input.value.strip() or "agile_config.json")

            try:
                sync_ui_to_config()
                vm.config.save_as(save_path)
                set_default_paths_if_empty()  # Set default paths after saving
                sync_config_to_ui()  # Update path display and enable engine button
                vm.notify(f"Saved new file: {save_path.name}", "success")
            except Exception as ex:
                vm.notify(f"Save As failed: {ex}", "error")
            finally:
                dialog.open = False

        btn_cancel.on_click(_on_cancel)
        btn_save.on_click(_on_save)

        dialog = Dialog(
            pn.Column(
                "### Choose a folder (or a file to use its parent)",
                dir_selector,
                pn.Row(filename_input),
                pn.Row(btn_cancel, btn_save),
            ),
            title="Save Configuration As..."
        )
        dialogs_container[:] = [dialog]
        dialog.open = True


    def open_file_selector(target_widget, is_directory, event):
        """Opens a file selector dialog and updates the target widget."""
        def update_target(value):
            target_widget.value = str(Path(value))
        return open_file_selector_dialog(update_target, is_directory, target_widget.value, event)


    def open_file_selector_dialog(select_callback, 
                           is_directory,
                           start_value="",
                           event=None):
        """Opens a file selector dialog and updates the target widget."""
        start_val = start_value.strip()
        start_path = Path(start_val).expanduser() if start_val else Path.home()
        start_dir = start_path if start_path.is_dir() else start_path.parent

        file_selector = pn.widgets.FileSelector(
            directory=str(start_dir),
            only_files=not is_directory
        )

        btn_cancel = pn.widgets.Button(name="Cancel", button_type="danger")
        btn_select = pn.widgets.Button(name="Select", button_type="primary")

        def _on_cancel(evt):
            dialog.open = False

        def _on_select(evt):
            if file_selector.value:
                # For directories, take the selected directory; for files, take the selected file.
                select_callback(file_selector.value[0])
            dialog.open = False
            

        btn_cancel.on_click(_on_cancel)
        btn_select.on_click(_on_select)

        dialog_content = pn.Column(
            file_selector,
        )

        dialog = Dialog(
            pn.Column(dialog_content, 
                      pn.Row(btn_cancel, btn_select)),
            title=f"Select {'Directory' if is_directory else 'File'}",
        )


        dialogs_container[:] = [dialog]
        dialog.open = True


    # Set up advanced section with path widgets now that open_file_selector is defined
    advanced_section.objects = [
        pn.pane.Markdown("Configure custom paths for database and folders."),
        pn.Row(
            db_path, 
            pn.widgets.Button(
                name="📂", 
                width=60,
                on_click=partial(open_file_selector, db_path, True)
            ),
            sizing_mode="stretch_width"
        ),
        pn.Row(
            labeled_dir, 
            pn.widgets.Button(
                name="📂", 
                width=70,
                on_click=partial(open_file_selector, labeled_dir, True)
            ),
            sizing_mode="stretch_width"
        ),
        pn.Row(
            models_dir, 
            pn.widgets.Button(
                name="📂", 
                width=60,
                on_click=partial(open_file_selector, models_dir, True)
            ),
            sizing_mode="stretch_width"
        ),
        pn.Row(
            preds_dir, 
            pn.widgets.Button(
                name="📂", 
                width=70,
                on_click=partial(open_file_selector, preds_dir, True)
            ),
            sizing_mode="stretch_width"
        ),
    ]

    # Connect the toggle checkbox to show/hide advanced section
    show_advanced.param.watch(toggle_advanced_visibility, 'value')


    def _make_engine(_):

        # anything observing engine_state will check for updates
        vm.obs.bump("engine_state")

        try:

            vm.eng = agile_state(vm.config)

            vm.notify("Engine created. Initializing...", "info")

            # initialise the API connection
            try:
                vm.eng.init_workbench(allow_prompt=False)
                vm.notify("API connection successful", "success")
            except Exception as e:
                vm.notify(f"API connection failed {e}", "error")

            # initialise the database
            try:
                db_success = vm.eng.init_db()
            except Exception as e:
                db_success = False
                log.exception("Database initialization error: %s", e)

            if db_success:
                vm.notify("Database initialized", "success")
            else:
                vm.notify("Database initialization failed", "error")

            tags_success = vm.eng.fetch_workbench_tags()
            if tags_success:
                vm.notify("Tags fetched successfully", "success")
            else:
                vm.notify("Failed to fetch tags", "error")

            # Create labels editor regardless of tag fetch success
            labels_editor_placeholder.objects = [vm.ui.view_target_labels(vm.eng)]

            # this will trigger anything observing these observables to update
            vm.obs.bump("engine_state")
            vm.obs.bump("target_labels")
            
            # Update button state after successful engine creation
            update_engine_button()

        except Exception as e:
            vm.notify(f"Failed to start engine: {e}", "error")


    # --- BINDINGS ---
    config_widgets = [annotator_id, domain, auth_token, show_advanced, db_path, labeled_dir, models_dir, preds_dir]
    for widget in config_widgets:
        widget.param.watch(_autosave, 'value')
    btn_new.on_click(_on_new_click)
    btn_open.on_click(_on_open_click)
    btn_save.on_click(_autosave) 
    btn_save_as.on_click(_on_save_as_click)
    btn_start_engine.on_click(_make_engine)
    
    # Set up advanced section toggle
    show_advanced.param.watch(toggle_advanced_visibility, 'value')

    # Initialize default paths for new configurations
    _print_paths("0.0")
    if is_new_unsaved_config:
        set_default_paths_if_empty()
    _print_paths("0.1")
    sync_config_to_ui()

    def _init_config_page():

        if not pn.state.curdoc or not pn.state.curdoc.session_context:
            return

        if is_new_unsaved_config:
            vm.notify("Welcome! Please configure your project and click 'Save As' to begin.", "warn")
        else:
            # Safe access to config_path.name
            config_name = vm.config.config_path.name if vm.config.config_path else "Unknown"
            vm.notify(f"Loaded recent project: {config_name}", "success")
            
            # Auto-start engine for loaded configurations
            if should_auto_start_engine:
                _make_engine(None)

    run_after_render(_init_config_page)

    # --- LAYOUT ---
    file_buttons = pn.Row(btn_new, btn_open, btn_save, btn_save_as, sizing_mode="fixed")
    config_status = pn.Row(current_config_path_display, save_status, sizing_mode="fixed")

    return pn.Column(
        pn.pane.Markdown(
            "Configure your project settings, database paths, and API connections. "
            "Click 'Start Engine' when ready to begin."
        ),
        
        # File operations section
        pn.Row("### Project File", pn.Spacer(), project_help_btn),
        file_buttons,
        config_status,
        
        # Configuration sections
        pn.Row(
            # Left column - Workbench connection
            pn.Column(
                pn.Row("### 🌐 Workbench Connection", pn.Spacer()),
                pn.Row(annotator_id, annotator_help_btn, sizing_mode="stretch_width"),
                pn.Row(domain, domain_help_btn, sizing_mode="stretch_width"),
                pn.Row(auth_token, token_help_btn, sizing_mode="stretch_width"),
                pn.Row(show_advanced, folder_help_btn, sizing_mode="stretch_width"),
                advanced_section,
                width=300,
                sizing_mode="fixed"
            ),
            
            # Right column - Basic settings
            pn.Column(
                pn.Row("### 🎯 Target Labels", pn.Spacer(), labels_help_btn),
                labels_editor_placeholder,
                sizing_mode="stretch_width"
            ),
            sizing_mode="stretch_width"
        ),
        
        # Engine initialization
        pn.Row("### 🚀 Engine", pn.Spacer(), engine_help_btn),
        pn.Row(btn_start_engine, sizing_mode="stretch_width"),
        
       
        
        # Hidden dialogs container
        dialogs_container,
        sizing_mode="stretch_both",
    )

