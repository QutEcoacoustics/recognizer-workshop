from functools import partial
from pathlib import Path
import panel as pn
import json
from datetime import datetime

# Assuming your classes are now in these locations after refactoring
from recognizer_workshop.app.agile_engine import AgileConfig, agile_state, LabelName
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
    config_path_widget = pn.widgets.TextInput(
        name="Config Save Path",
        value=str(ppath("~/agile_config.json"))
    )
    save_status = pn.pane.Markdown("")

    # this will create with default values if the path does not exist
    vm.config = AgileConfig.from_json(config_path_widget.value)

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

    labels_container = pn.Column(sizing_mode="stretch_width")
    btn_add_label = pn.widgets.Button(name="Add Label", icon="plus", button_type="success", disabled=True)
    current_labels_display = pn.pane.Markdown("*(Engine not started. No labels defined.)*")



    # update the config object and immediately save to json 
    # triggered whenever the user makes a change to the UI fields
    def sync_ui_to_config_and_save(*events):
        """Syncs UI widget values back to the central vm.config object and then saves it."""
        try:
            # Update the config object from the UI
            vm.config.annotator_id = annotator_id.value
            vm.config.db_path = Path(db_path.value) if db_path.value else None
            vm.config.labeled_examples_folder = Path(labeled_dir.value) if labeled_dir.value else None
            vm.config.models_folder = Path(models_dir.value) if models_dir.value else None
            vm.config.predictions_folder = Path(preds_dir.value) if preds_dir.value else None
            vm.config.baw_config["domain"] = domain.value
            vm.config.baw_config["auth_token"] = auth_token.value

            # Save the updated config object to JSON using its own method
            vm.config.to_json(config_path_widget.value)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            save_status.object = f"✅ *Saved at {timestamp}*"
        except Exception as e:
            vm.notify(f"Config save failed: {e}", "error")
            save_status.object = f"❌ *Save failed!*"



    def open_file_selector(target_widget, is_directory, event):
        
        #TODO: this is broken and doesn't work
        """Opens a file selector dialog and updates the target widget."""
        start_dir = Path(target_widget.value).parent if Path(target_widget.value).exists() else Path.home()
        
        file_selector = pn.widgets.FileSelector(
            directory=str(start_dir), 
            only_files=not is_directory,
            height=300
        )
        
        dialog = Dialog(
            title=f"Select {'Directory' if is_directory else 'File'}",
            content=file_selector,
            buttons=[
                {'name': 'Cancel', 'button_type': 'danger'},
                {'name': 'Select', 'button_type': 'primary'}
            ],
            width=600,
        )

        def _on_select(event):
            if dialog.value == 'Select' and file_selector.value:
                target_widget.value = file_selector.value[0]
            dialog.open = False # Close the dialog

        dialog.on_click = _on_select
        dialog.open = True # Open the dialog


    def sync_and_update_display(*events):
        """A wrapper to save the config and update the human-readable display text."""
        sync_ui_to_config_and_save()
        if not vm.eng: return
        try:
            description = vm.eng.get_target_label_description()
            current_labels_display.object = f"**Summary:** {description}" if description else "*(No labels defined)*"
        except Exception:
            current_labels_display.object = "*(No labels defined)*"


    def _remove_label(label_to_remove, event):
        """Callback to remove a label from the list and rebuild the UI."""
        if label_to_remove in vm.config.target_labels:
            vm.config.target_labels.remove(label_to_remove)
            _build_labels_ui()
            sync_and_update_display()

    def _build_labels_ui():
        """Clears and rebuilds the UI for editing labels."""
        labels_container.clear()
                
        for i, label in enumerate(vm.config.target_labels):
            species_widget = pn.widgets.AutocompleteInput(
                name=f"Species {i+1}",
                options=vm.eng.workbench_tags,
                value=label.species,
                placeholder="Type to search for a species...",
                case_sensitive=False
            )
            call_types_widget = pn.widgets.TextInput(
                name="Call Type",
                value=label.call_type,
                placeholder="(optional) e.g., song"
            )
            remove_btn = pn.widgets.Button(name="Remove", icon="trash", button_type="danger", width=100, align="end")
            remove_btn.on_click(partial(_remove_label, label))

            def _update_species(event, target_label=label):
                target_label.species = event.new
                sync_and_update_display()
            
            def _update_call_type(event, target_label=label):
                target_label.call_type = event.new.strip()
                sync_and_update_display()

            species_widget.param.watch(_update_species, 'value')
            call_types_widget.param.watch(_update_call_type, 'value')

            labels_container.append(
                pn.Row(species_widget, call_types_widget, remove_btn, sizing_mode="stretch_width")
            )

    def _add_label(event):
        """Callback to add a new empty label and rebuild the UI."""
        vm.config.target_labels.append(LabelName(species= "", call_type=None))
        _build_labels_ui()
        sync_and_update_display()

    btn_add_label.on_click(_add_label)


    def _make_engine(_):
        vm.eng = agile_state(vm.config)
        # create the UI adaptor that 'implements' the AgileUI protocol
        vm.ui  = PanelUI(annotator_id=vm.config.annotator_id)

        vm.notify("Engine started", "success")

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
            btn_add_label.disabled = False
            _build_labels_ui()
            
        else:
            vm.notify("Failed to fetch tags", "error")

        sync_and_update_display()



    # --- BINDINGS ---
    config_widgets = [annotator_id, db_path, labeled_dir, models_dir, preds_dir, domain, auth_token]
    for widget in config_widgets:
        widget.param.watch(sync_ui_to_config_and_save, 'value')

    btn_start_engine.on_click(_make_engine)
   

    # --- LAYOUT ---
    return pn.Column(
        "## Config",
        pn.Row(config_path_widget, save_status, annotator_id, sizing_mode="stretch_width"),
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
        "---",
        pn.pane.Markdown("## 3. Define Target Labels"),
        pn.pane.Alert(
            "Define one or more target species. For each species, you can optionally specify distinct call types. "
            "If no call types are given, all sounds for that species will be treated as a single class.",
            alert_type="info"
        ),
        current_labels_display,
        labels_container,
        btn_add_label,
        sizing_mode="stretch_both",
    )

