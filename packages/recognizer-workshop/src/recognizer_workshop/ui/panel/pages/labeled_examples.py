import copy
import json

import pandas as pd
import panel as pn

from recognizer_workshop.app.services.db_labels.add_db_labels import add_labels_to_db

def build(vm):

    # Create help buttons
    labeled_examples_help_btn = vm.help_system.create_help_button("help-labeled-examples-overview", "Help: Labeled Examples")
    filter_help_btn = vm.help_system.create_help_button("help-recordings-filter-labelled-examples", "Help: Recording Filters")

    btn_fetch = pn.widgets.Button(name="Fetch labels from server", button_type="primary")
    btn_add   = pn.widgets.Button(name="Add fetched labels to DB", button_type="success", disabled=True)
    log       = pn.pane.Markdown("")
    # Initialize tables with empty DataFrames to prevent None value errors
    summary_table = pn.pane.DataFrame(pd.DataFrame(), sizing_mode="stretch_width")
    outcomes_table = pn.pane.DataFrame(pd.DataFrame(), sizing_mode="stretch_width")
    
    # Details table - shows ALL outcomes, sortable and filterable
    # Initialize with empty DataFrame that has the expected columns
    empty_details_df = pd.DataFrame(columns=['Filter', 'Species', 'Type', 'Source', 'Offset', 'Outcome'])
    details_table = pn.widgets.Tabulator(
        value=empty_details_df,
        sizing_mode="stretch_width", 
        pagination='remote', 
        page_size=50,
        sorters=[{'field': 'Outcome', 'dir': 'asc'}],
        show_index=False,
        disabled=True  # Make read-only
    )
    details_title = pn.pane.Markdown("")
    stash     = {"result_sets": None, "outcomes": None}
    
    project_ids_input = pn.widgets.TextInput(name="Project IDs (comma-separated)", placeholder="e.g., 123, 456")
    region_ids_input = pn.widgets.TextInput(name="Region IDs (comma-separated)", placeholder="e.g., 789")
    site_ids_input = pn.widgets.TextInput(name="Site IDs (comma-separated)", placeholder="e.g., 101, 102")
    btn_save = pn.widgets.Button(name="Save Filters", button_type="primary", icon='save')
    btn_cancel = pn.widgets.Button(name="Cancel")


    filter_display = pn.pane.Markdown(sizing_mode="stretch_width")
    btn_edit = pn.widgets.Button(name="Edit", icon='edit')

    editor_container = pn.Column(sizing_mode="stretch_width")

    display_view = pn.Column(
        pn.pane.Markdown("#### Ecosounds Recording Settings"),
        pn.Row(filter_display, btn_edit, align="center")
    )
    edit_view = pn.Column(
        pn.pane.Markdown("#### Edit Ecosounds Recording Settings"),
        project_ids_input,
        region_ids_input,
        site_ids_input,
        pn.Row(btn_save, btn_cancel)
    )


    def _update_filter_display():
        """Decides whether to show the display_view or edit_view."""
        if not (vm.eng and vm.eng.db):
            editor_container.objects = [
                pn.pane.Alert("Initialize the engine on the 'Config' tab to set up filters.", alert_type="warning")
            ]
            return

        filters = vm.eng.get_baw_filters()
        filter_display.object = _format_filters_for_display(filters)
        # If filters exist and are not empty, show the display view.
        if filters and filters[0]:
            editor_container.objects = [display_view]
        else: # Otherwise, show the edit view to prompt the user.
            _populate_editors_from_filters()
            editor_container.objects = [edit_view]


    def _format_filters_for_display(filters: list) -> str:
        """Human-readable formatting of the filters"""
        if not filters or not filters[0]:
            return "No filters set. Click 'Edit' to add one."

        filter_dict = filters[0]
        description_lines = []
        
        # Define the order and labels for the filters
        filter_keys = {
            "projects.id": "Projects",
            "regions.id": "Regions",
            "sites.id": "Sites"
        }

        for key, label in filter_keys.items():
            if key in filter_dict and "in" in filter_dict[key]:
                id_list = filter_dict[key]["in"]
                if id_list:
                    ids_str = ', '.join(map(str, id_list))
                    description_lines.append(f"**{label}:** `{ids_str}`")

        if not description_lines:
            return "No filters set. Click 'Edit' to add one."

        # Join the lines with an "OR"
        return " <br> **OR** <br> ".join(description_lines)


    def _format_filter_params_for_display(filter_params: dict, fallback_name: str = "Filter") -> str:
        """Convert filter parameters to human-readable description."""
        filter_parts = []
        
        if 'projects.id' in filter_params and 'in' in filter_params['projects.id']:
            project_ids = ', '.join(map(str, filter_params['projects.id']['in']))
            filter_parts.append(f"Projects: {project_ids}")
        if 'regions.id' in filter_params and 'in' in filter_params['regions.id']:
            region_ids = ', '.join(map(str, filter_params['regions.id']['in']))
            filter_parts.append(f"Regions: {region_ids}")
        if 'sites.id' in filter_params and 'in' in filter_params['sites.id']:
            site_ids = ', '.join(map(str, filter_params['sites.id']['in']))
            filter_parts.append(f"Sites: {site_ids}")
        
        return ' + '.join(filter_parts) if filter_parts else fallback_name


    def _create_fetch_summary_table(result_sets):
        """Create a summary table from the fetched results."""
        if not result_sets:
            return None
        
        rows = []
        for i, result_set in enumerate(result_sets):
            filter_params = result_set.get('filter_params', {})
            counts = result_set.get('counts', {})
            
            # Use the abstracted function to get filter description
            filter_desc = _format_filter_params_for_display(filter_params, f"Filter {i+1}")
            
            # Add rows for each species and label type
            for species, label_counts in counts.items():
                if 'positive' in label_counts:
                    rows.append({
                        'Filter': filter_desc,
                        'Species': species,
                        'Type': 'Positive',
                        'Count': label_counts['positive']
                    })
                if 'negative' in label_counts:
                    rows.append({
                        'Filter': filter_desc,
                        'Species': species,
                        'Type': 'Negative', 
                        'Count': label_counts['negative']
                    })
        
        if not rows:
            return None
            
        df = pd.DataFrame(rows)
        # Reset index to ensure clean integer sequence starting from 0 and hide it
        df = df.reset_index(drop=True)
        return df


    def _create_outcomes_summary_table(outcomes_by_filter):
        """Create a table showing the outcomes of adding labels to the database."""
        if not outcomes_by_filter:
            return None
        
        rows = []
        
        # Process each filter's outcomes
        for filter_data in outcomes_by_filter:
            filter_desc = filter_data['filter_desc']
            outcomes_data = filter_data['outcomes']
            
            # Aggregate outcomes by species and type for this filter
            species_type_outcomes = {}
            
            for source_id, labels_data in outcomes_data.items():
                for label_str, type_data in labels_data.items():
                    for label_type_str, outcome_list in type_data.items():
                        # Create a key for this species/type combination
                        key = (label_str, label_type_str)
                        
                        if key not in species_type_outcomes:
                            species_type_outcomes[key] = {
                                'added': 0, 'already_exists': 0, 'embedding_not_found': 0, 
                                'replaced_type': 0, 'error': 0
                            }
                        
                        # Count outcomes for this species/type
                        for outcome_item in outcome_list:
                            outcome = outcome_item.get('outcome', 'error')
                            if outcome in species_type_outcomes[key]:
                                species_type_outcomes[key][outcome] += 1
            
            # Create rows for this filter
            for (label_str, label_type_str), outcome_counts in species_type_outcomes.items():
                # Calculate total count (should match the original count if everything went right)
                total_count = sum(outcome_counts.values())
                
                type_display = 'Positive' if label_type_str == 'pos' else 'Negative'
                
                rows.append({
                    'Filter': filter_desc,
                    'Species': label_str,
                    'Type': type_display,
                    'Count': total_count,
                    'Added': outcome_counts['added'],
                    'Already Exists': outcome_counts['already_exists'],
                    'Embedding Not Found': outcome_counts['embedding_not_found'],
                    'Replaced Type': outcome_counts['replaced_type'],
                    'Error': outcome_counts['error']
                })
        
        if not rows:
            return None
            
        df = pd.DataFrame(rows)
        df = df.reset_index(drop=True)
        return df


    def _create_outcomes_details_table(outcomes_by_filter):
        """Create a comprehensive table showing ALL outcome details."""
        if not outcomes_by_filter:
            return None
        
        rows = []
        
        # Process each filter's outcomes
        for filter_data in outcomes_by_filter:
            filter_desc = filter_data['filter_desc']
            outcomes = filter_data['outcomes']
            
            for source_id, labels_data in outcomes.items():
                for species, type_data in labels_data.items():
                    for label_type_str, outcome_list in type_data.items():
                        type_display = 'Positive' if label_type_str == 'pos' else 'Negative'
                        
                        for outcome_item in outcome_list:
                            rows.append({
                                'Filter': filter_desc,
                                'Species': species,
                                'Type': type_display,
                                'Source': source_id,
                                'Offset': outcome_item.get('offset', 'N/A'),
                                'Outcome': outcome_item.get('outcome', 'N/A')
                            })
        
        if not rows:
            return None
            
        df = pd.DataFrame(rows)
        df = df.reset_index(drop=True)
        return df
        """Automatically show embedding_not_found details if any exist."""
        if not outcomes_by_filter:
            return
            
        # Check if there are any embedding_not_found cases
        has_missing = False
        for filter_data in outcomes_by_filter:
            outcomes = filter_data['outcomes']
            for source_id, labels_data in outcomes.items():
                for species, type_data in labels_data.items():
                    for label_type_str, outcome_list in type_data.items():
                        if any(item.get('outcome') == 'embedding_not_found' for item in outcome_list):
                            has_missing = True
                            break
        
        if has_missing:
            # Show details for embedding_not_found
            _show_outcome_details('', '', '', 'embedding_not_found')


    def _populate_editors_from_filters():
        """Helper to parse existing filters and fill the text inputs."""
        filters = vm.eng.get_baw_filters()
        project_ids_input.value, region_ids_input.value, site_ids_input.value = "", "", ""

        if not filters or not isinstance(filters, list) or not filters[0]:
            return
            
        filter_dict = filters[0]

        def get_ids(key):
            if key in filter_dict and "in" in filter_dict[key]:
                return ", ".join(map(str, filter_dict[key]["in"]))
            return ""

        project_ids_input.value = get_ids("projects.id")
        region_ids_input.value = get_ids("regions.id")
        site_ids_input.value = get_ids("sites.id")


    def _toggle_to_edit_view(_):
        """Swaps the container's content to the edit_view."""
        _populate_editors_from_filters()
        editor_container.objects = [edit_view]


    def _save_filters(_):
        """Saves filters and then switches back to the display view."""
        if not vm.eng: return vm.notify("Initialize on Config tab first.", "error")

        inputs = {"projects.id": project_ids_input.value, "regions.id": region_ids_input.value, "sites.id": site_ids_input.value}
        if not any(val.strip() for val in inputs.values()):
            vm.notify("Please enter IDs in at least one field.", "warn"); return
        new_filter = {}
        try:
            for key, value_str in inputs.items():
                if not value_str.strip(): continue
                id_list = [int(s.strip()) for s in value_str.split(',') if s.strip()]
                if id_list: new_filter[key] = {"in": id_list}
        except ValueError:
            vm.notify("Invalid input. Please ensure all IDs are comma-separated numbers.", "error"); return
        
        try:
            vm.eng.set_baw_filters([new_filter])
            vm.notify("BAW filters saved successfully!", "success")
            _update_filter_display() 
        except Exception as e:
            vm.notify(f"Could not save filters: {e}", "error")


    def _fetch(_):
        if not vm.eng: return vm.notify("Initialize on Config tab first.", "error")
        try:
            result = vm.eng.fetch_labels()
            stash["result_sets"] = result
            
            # Create and display summary table
            summary_df = _create_fetch_summary_table(result)
            if summary_df is not None:
                # Configure the DataFrame to not show index
                summary_table.object = summary_df.style.hide(axis='index')
            else:
                summary_table.object = pd.DataFrame()  # Use empty DataFrame instead of None
            
            total_labels = sum(len(rs['labels']) for rs in result)
            log.object = f"Successfully fetched **{total_labels}** labels across {len(result)} result sets."
            btn_add.disabled = False
            vm.notify("Fetch successful. Ready to add to DB.", "success")
        except Exception as e:
            log.object = f"Error: {e}"
            summary_table.object = pd.DataFrame()  # Use empty DataFrame instead of None
            vm.notify(f"Fetch failed: {e}", "error")

    def _add(_):
        if not vm.eng: return vm.notify("Initialize first.", "error")
        if not stash["result_sets"]: return vm.notify("Fetch labels first.", "warn")

        try:
            outcomes_list = vm.eng.commit_labels(stash["result_sets"])

            # Show the results as a table with the outcome of adding each label
            # each item in the outcomes list corresponds to a result_set, which corresponds to an audio recording filter. 
            # we display the filter in a human-readable way
            for i, item in enumerate(outcomes_list):
                item['filter_desc'] = _format_filter_params_for_display(
                    item['filter_params'], 
                    fallback_name=f"Filter {i+1}"
                )                

            stash["outcomes"] = outcomes_list
            outcomes_df = _create_outcomes_summary_table(outcomes_list)
            
            if outcomes_df is not None:
                outcomes_table.object = outcomes_df.style.hide(axis='index')
                details_df = _create_outcomes_details_table(outcomes_list)
                if details_df is not None:
                    details_table.value = details_df
                    details_title.object = f"### All Outcome Details ({len(details_df)} records)"
                else:
                    details_table.value = pd.DataFrame(columns=['Filter', 'Species', 'Type', 'Source', 'Offset', 'Outcome'])
                    details_title.object = "### No outcome details available"
            else:
                outcomes_table.object = pd.DataFrame()  # Use empty DataFrame instead of None
                details_table.value = pd.DataFrame(columns=['Filter', 'Species', 'Type', 'Source', 'Offset', 'Outcome'])
                details_title.object = ""

                    
            vm.notify("Labels added to DB.", "success")
        except Exception as e:
            vm.notify(f"Insert failed: {e}", "error")
            return
        




    btn_edit.on_click(_toggle_to_edit_view)
    btn_save.on_click(_save_filters)
    btn_cancel.on_click(lambda _: _update_filter_display())
    btn_fetch.on_click(_fetch)
    btn_add.on_click(_add)
    
    # Set up click event for outcomes table
    # outcomes_table.param.watch(_on_outcomes_table_click, 'clicks')

    _update_filter_display()

    @pn.depends(vm.obs.ensure("engine_state").param.version, watch=True)
    def _on_engine_state_change(_):
        _update_filter_display()

    return pn.Column(
        pn.Row("## Labelled examples", pn.Spacer(), labeled_examples_help_btn),
        pn.pane.Alert(
            "Define a set of Ecosounds recordings by Project, Region, or Site IDs. "
            "Labels will be fetched from annotations within these recordings.",
            alert_type="info"
        ),

        pn.Row(
            pn.Column(
                pn.Row("Recording Filter Settings", pn.Spacer(), filter_help_btn),
                editor_container
            )
        ),
        pn.layout.Divider(),
        pn.Row(btn_fetch, btn_add),
        log,
        pn.layout.Divider(),
        pn.pane.Markdown("### Label Summary"),
        summary_table,
        pn.layout.Divider(),
        pn.pane.Markdown("### Database Insert Outcomes"),
        outcomes_table,
        pn.layout.Divider(),
        details_title,
        details_table,
        sizing_mode="stretch_both",
    )