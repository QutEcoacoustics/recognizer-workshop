from functools import partial
from pathlib import Path
from tkinter import dialog
import panel as pn

from datetime import datetime


from recognizer_workshop.app.agile_engine import AgileConfig, agile_state, get_recent_files
from recognizer_workshop.ui.panel.adaptors import PanelUI
from recognizer_workshop.ui.panel.utils import ppath

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
    # --- WIDGETS ---

    # config_path_widget = pn.widgets.TextInput(
    #     name="Config Save Path",
    #     value=str(ppath("~/agile_config.json"))
    # )

    dialogs_container = pn.Column()  

    recent_files = get_recent_files()
    if recent_files:
        try:
            # Try to load the most recent file
            vm.config = AgileConfig.load(recent_files[0])
            is_new_unsaved_config = False
        except FileNotFoundError:
            # If the recent file was moved/deleted, start with a new one
            vm.config = AgileConfig()
            is_new_unsaved_config = True
    else:
        # If no recent files, start with a new, default config
        vm.config = AgileConfig()
        is_new_unsaved_config = True
    
    save_status = pn.pane.Markdown("")

    btn_new = pn.widgets.Button(name="New", icon='new-document')
    btn_open = pn.widgets.Button(name="Open...", icon='folder-open')
    btn_save = pn.widgets.Button(name="Save", icon='save') # maybe not necessary, because we autosave
    btn_save_as = pn.widgets.Button(name="Save As...")
    current_config_path_display = pn.widgets.StaticText()



    # text inputs for config fields, with values coming from the loaded config
    annotator_id = pn.widgets.TextInput(name="Annotator", value=vm.config.annotator_id)
    db_path = pn.widgets.TextInput(name="DB path (Directory)", value=str(vm.config.db_path or ""))
    labeled_dir = pn.widgets.TextInput(name="Labeled examples folder", value=str(vm.config.labeled_examples_folder or ""))
    models_dir = pn.widgets.TextInput(name="Models folder", value=str(vm.config.models_folder or ""))
    preds_dir = pn.widgets.TextInput(name="Predictions folder", value=str(vm.config.predictions_folder or ""))
    domain = pn.widgets.TextInput(name="Workbench API domain", value=vm.config.baw_config.get("domain", ""))
    auth_token = pn.widgets.TextInput(name="Workbench API token", value=vm.config.baw_config.get("auth_token", ""))

    # --- BUTTONS ---
    btn_start_engine = pn.widgets.Button(name="Start Engine", button_type="primary")

    # Label editor populated only after the engine has started successfully.
    labels_editor_placeholder = pn.Column(sizing_mode="stretch_width")

    def sync_config_to_ui():
        """Updates all UI widgets from the current vm.config object."""
        annotator_id.value = vm.config.annotator_id
        db_path.value = str(vm.config.db_path or "")
        labeled_dir.value = str(vm.config.models_folder)
        models_dir.value = str(vm.config.models_folder)
        preds_dir.value = str(vm.config.predictions_folder)
        path_str = str(vm.config.config_path) if vm.config.config_path else "Unsaved Configuration"
        current_config_path_display.value = f"<b>Current File:</b> {path_str}"
        btn_start_engine.disabled = not vm.config.config_path

    def sync_ui_to_config():
        """Updates the vm.config object from all UI widgets."""
        vm.config.annotator_id = annotator_id.value
        vm.config.db_path = Path(db_path.value) if db_path.value else None
        vm.config.labeled_examples_folder = Path(labeled_dir.value) if labeled_dir.value else None
        vm.config.models_folder = Path(models_dir.value) if models_dir.value else None
        vm.config.predictions_folder = Path(preds_dir.value) if preds_dir.value else None
        vm.config.baw_config["domain"] = domain.value
        vm.config.baw_config["auth_token"] = auth_token.value


    def _autosave(*events):
        """Syncs UI to the config object and saves it if a path exists."""
        sync_ui_to_config()
        if vm.config.config_path:
            try:
                vm.config.save()
                vm.notify("Autosaved.", "info")
            except Exception as e:
                vm.notify(f"Autosave failed: {e}", "error")


    def _on_new_click(_):
        """Resets the current configuration to a blank, unsaved state."""
        vm.config.reset()
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
                except Exception as ex:
                    vm.notify(f"Failed to load config: {ex}", "error")
            dialog.open = False

        
        btn_cancel.on_click(_on_cancel)
        btn_open.on_click(_on_open)

        dialog = Dialog(
            pn.Column(dialog_content, pn.Row(btn_cancel, btn_open)),
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
        start_val = (target_widget.value or "").strip()
        start_path = Path(start_val).expanduser() if start_val else Path.home()
        start_dir = start_path if start_path.is_dir() else start_path.parent

        file_selector = pn.widgets.FileSelector(
            directory=str(start_dir),
            only_files=not is_directory,
            height=300,
        )

        btn_cancel = pn.widgets.Button(name="Cancel", button_type="danger")
        btn_select = pn.widgets.Button(name="Select", button_type="primary")

        def _on_cancel(evt):
            dialog.open = False

        def _on_select(evt):
            if file_selector.value:
                # For directories, take the selected directory; for files, take the selected file.
                target_widget.value = file_selector.value[0]
            dialog.open = False

        btn_cancel.on_click(_on_cancel)
        btn_select.on_click(_on_select)

        dialog = Dialog(
            pn.Column(
                pn.pane.Markdown(f"### Select {'Directory' if is_directory else 'File'}"),
                file_selector,
                pn.Row(btn_cancel, btn_select),
            ),
            title=f"Select {'Directory' if is_directory else 'File'}",
        )
        dialog.open = True


    def _make_engine(_):

        try:

            vm.eng = agile_state(vm.config)

            vm.notify("Engine created. Initializing...", "info")

            # initialise the API connection
            token_success = vm.eng.init_workbench(allow_prompt=False)
            if token_success:
                vm.notify("API connection successful", "success")
            else:
                vm.notify("API connection failed", "error")

            # initialise the database
            db_success = vm.eng.init_db()
            if db_success:
                vm.notify("Database initialized", "success")
            else:
                vm.notify("Database initialization failed", "error")

            tags_success = vm.eng.fetch_workbench_tags()
            if tags_success:
                vm.notify("Tags fetched successfully", "success")
                labels_editor_placeholder.objects = [vm.ui.view_target_labels(vm.eng)]
            else:
                vm.notify("Failed to fetch tags", "error")

            labels_editor_placeholder.objects = [vm.ui.view_target_labels(vm.eng)]

            print("before", vm.obs)

            # this will trigger anything observing these observables to update
            vm.obs.bump("engine_state")
            vm.obs.bump("target_labels")

            print("after", vm.obs)

        except Exception as e:
            vm.notify(f"Failed to start engine: {e}", "error")


    # --- BINDINGS ---
    config_widgets = [annotator_id, db_path, labeled_dir, models_dir, preds_dir, domain, auth_token]
    for widget in config_widgets:
        widget.param.watch(_autosave, 'value')
    btn_new.on_click(_on_new_click)
    btn_open.on_click(_on_open_click)
    btn_save.on_click(_autosave) 
    btn_save_as.on_click(_on_save_as_click)
    btn_start_engine.on_click(_make_engine)

    sync_config_to_ui()

    if is_new_unsaved_config:
        vm.notify("Welcome! Please configure your project and click 'Save As' to begin.", "warn")
    else:
        vm.notify(f"Loaded recent project: {vm.config.config_path.name}", "success")

    # --- LAYOUT ---

    file_buttons = pn.Row(btn_new, btn_open, btn_save, btn_save_as, current_config_path_display)

    return pn.Column(
        "## Config",
        pn.Row(file_buttons, save_status, annotator_id, sizing_mode="stretch_width"),
        pn.Row(
            pn.Column("## Workbench Connection", domain, auth_token, 
                      #btn_test_connection
                      ),
            pn.Column(
                "## Database",
                pn.Row(db_path, pn.widgets.Button(name="Select...", on_click=partial(open_file_selector, db_path, False))),
                #btn_init_db,
                "## Output Folders",
                pn.Row(labeled_dir, pn.widgets.Button(name="Select...", on_click=partial(open_file_selector, labeled_dir, False))),
                pn.Row(models_dir, pn.widgets.Button(name="Select...", on_click=partial(open_file_selector, models_dir, False))),
                pn.Row(preds_dir, pn.widgets.Button(name="Select...", on_click=partial(open_file_selector, preds_dir, False)))
            )
        ),
        pn.Row(btn_start_engine, sizing_mode="stretch_width"),
        pn.layout.Divider(),
        pn.pane.Markdown("## 3. Define Target Labels"),
        pn.pane.Alert(
            "Define one or more target species. For each species, you can optionally specify distinct call types. "
            "If no call types are given, all sounds for that species will be treated as a single class.",
            alert_type="info"
        ),
        labels_editor_placeholder,
        dialogs_container,
        sizing_mode="stretch_both",
    )

