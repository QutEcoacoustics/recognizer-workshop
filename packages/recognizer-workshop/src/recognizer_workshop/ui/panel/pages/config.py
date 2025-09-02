from functools import partial
from pathlib import Path
import panel as pn
import json
from datetime import datetime

# Assuming your classes are now in these locations after refactoring
from recognizer_workshop.app.agile_engine import AgileConfig, agile_state
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
    api_token = pn.widgets.TextInput(name="Workbench API token", value=vm.config.baw_config.get("auth_token", ""))

    # --- BUTTONS ---
    btn_start_engine = pn.widgets.Button(name="Start Engine", button_type="primary")
    btn_test_connection = pn.widgets.Button(name="Test Connection", button_type="primary", disabled=True)
    btn_init_db = pn.widgets.Button(name="Initialize DB", button_type="success", disabled=True)
    btn_pick_label = pn.widgets.Button(name="Pick Target Label(s)", button_type="primary", disabled=True)
    current_label = pn.pane.Markdown("*(no label picked yet)*")

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
            vm.config.baw_config["api_token"] = api_token.value

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


    def _make_engine(_):
        vm.eng = agile_state(vm.config)
        # create the UI adaptor that 'implements' the AgileUI protocol
        vm.ui  = PanelUI(annotator_id=vm.config.annotator_id)

        vm.notify("Engine started", "success")

        # initialise the API connection
        token_success = vm.eng.init_workbench()
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

        # btn_test_connection.disabled = False
        # btn_init_db.disabled    = False
        # btn_pick_label.disabled = True


    # def _load_token(_):
    #     if not vm.eng: return vm.notify("Create engine first.", "error")
    #     try:
    #         vm.eng.init_workbench()
    #         vm.notify("Token step finished.", "success")
    #     except Exception as e:
    #         vm.notify(f"Token step failed: {e}", "error")

    # def _init_db(_):
    #     if not vm.eng: return vm.notify("Create engine first.", "error")
    #     try:
    #         vm.eng.init_db()
    #         btn_pick_label.disabled = False
    #         vm.notify("DB initialized. You can now pick labels and use other tabs.", "success")
    #     except Exception as e:
    #         vm.notify(f"DB init failed: {e}", "error")

    def _pick_label(_):
        if not (vm.eng and vm.ui): return vm.notify("Initialize first.", "error")
        vm.ui.pick_label(vm.eng)
        try:
            current_label.object = f"**Target labels:** {', '.join(vm.eng.get_target_labels())}"
        except Exception:
            current_label.object = "*(no label picked yet)*"

    # --- BINDINGS ---
    all_widgets = [annotator_id, db_path, labeled_dir, models_dir, preds_dir, domain, config_path_widget]
    for widget in all_widgets:
        widget.param.watch(sync_ui_to_config_and_save, 'value')

    btn_start_engine.on_click(_make_engine)
    # btn_test_connection.on_click(_load_token)
    # btn_init_db.on_click(_init_db)
    btn_pick_label.on_click(_pick_label)
    

    # --- LAYOUT ---
    return pn.Column(
        "## Config",
        pn.Row(config_path_widget, save_status, annotator_id, sizing_mode="stretch_width"),
        pn.Row(
            pn.Column("## Workbench Connection", domain, api_token, 
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
        pn.Row(btn_pick_label, current_label),
        sizing_mode="stretch_both",
    )

