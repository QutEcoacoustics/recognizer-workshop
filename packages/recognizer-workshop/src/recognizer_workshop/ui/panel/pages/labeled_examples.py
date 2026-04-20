import threading

import pandas as pd
import panel as pn

from recognizer_workshop.app.services.db_labels.add_db_labels import add_labels_to_db
from recognizer_workshop.ui.panel.components.native.filter_editor import build_filter_editor

def build(vm):
    # Create help buttons
    labeled_examples_help_btn = vm.help_system.create_help_button("help-labeled-examples-overview", "Help: Labeled Examples")
    filter_help_btn = vm.help_system.create_help_button("help-recordings-filter-labelled-examples", "Help: Recording Filters")

    btn_fetch = pn.widgets.Button(name="Fetch labels from server", button_type="primary", sizing_mode="fixed", width=220)
    btn_add   = pn.widgets.Button(name="Add fetched labels to DB", button_type="success", disabled=True, sizing_mode="fixed", width=220)
    log       = pn.pane.Markdown("")
    summary_table = None
    outcomes_table = None
    details_table = None

    summary_placeholder = pn.pane.Markdown("_No summary yet._")
    outcomes_placeholder = pn.pane.Markdown("_No outcomes yet._")
    details_placeholder = pn.pane.Markdown("_No outcome details yet._")

    summary_container = pn.Column(summary_placeholder, sizing_mode="stretch_width")
    outcomes_container = pn.Column(outcomes_placeholder, sizing_mode="stretch_width")
    details_title = pn.pane.Markdown("")
    details_container = pn.Column(details_title, details_placeholder, sizing_mode="stretch_width")
    stash     = {"result_sets": None, "outcomes": None}
    
    def _get_filters():
        if not (vm.eng and vm.eng.db):
            return None
        return vm.eng.get_baw_filters()

    def _set_filters(filters):
        if not (vm.eng and vm.eng.db):
            raise RuntimeError("Initialize the engine on the Config tab first.")
        vm.eng.set_baw_filters(filters)

    engine_warning = pn.pane.Alert(
        "Initialize the engine on the 'Config' tab to set up filters.",
        alert_type="warning"
    )
    filter_section = pn.Column(sizing_mode="stretch_width")

    def _build_filter_editor_widget():
        """Build a fresh filter editor each time the engine state changes."""
        return build_filter_editor(
            _get_filters,
            _set_filters,
            vm,
            dataset_priority=['fetched_labels', 'search_set'],
            fallback_to_legacy=True,
        )

    def _update_filter_section():
        """Show filter editor when engine is ready, warning otherwise."""
        if not (vm.eng and vm.eng.db):
            filter_section.objects = [engine_warning]
        else:
            # Rebuild the filter editor to ensure it reflects latest DB metadata
            filter_editor = _build_filter_editor_widget()
            filter_section.objects = [filter_editor]


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


    def _ensure_tables():
        nonlocal summary_table, outcomes_table, details_table
        if summary_table is not None:
            return
        summary_table = pn.widgets.Tabulator(
            value=pd.DataFrame(),
            sizing_mode="stretch_width",
            min_height=180,
            show_index=False,
            disabled=True,
        )
        outcomes_table = pn.widgets.Tabulator(
            value=pd.DataFrame(),
            sizing_mode="stretch_width",
            min_height=220,
            show_index=False,
            disabled=True,
        )
        empty_details_df = pd.DataFrame(columns=['Filter', 'Species', 'Type', 'Source', 'Offset', 'Outcome'])
        details_table = pn.widgets.Tabulator(
            value=empty_details_df,
            sizing_mode="stretch_width",
            min_height=260,
            pagination='remote',
            page_size=50,
            sorters=[{'field': 'Outcome', 'dir': 'asc'}],
            show_index=False,
            disabled=True,
        )
        summary_container.objects = [summary_table]
        outcomes_container.objects = [outcomes_table]
        details_container.objects = [details_title, details_table]


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



    def _fetch(_):
        if not vm.eng: return vm.notify("Initialize on Config tab first.", "error")
        btn_fetch.disabled = True
        btn_add.disabled = True
        log.object = "Fetching labels…"
        doc = pn.state.curdoc

        def _schedule(fn):
            if doc:
                doc.add_next_tick_callback(fn)
            else:
                # Fallback when no document is available; run directly.
                fn()

        def _do_fetch():
            try:
                result = vm.eng.fetch_labels()

                def _apply_success():
                    _ensure_tables()
                    stash["result_sets"] = result

                    summary_df = _create_fetch_summary_table(result)
                    summary_table.value = summary_df if summary_df is not None else pd.DataFrame()

                    total_labels = sum(len(rs['labels']) for rs in result)
                    log.object = (
                        f"Successfully fetched **{total_labels}** labels across {len(result)} result sets."
                    )
                    btn_add.disabled = False
                    vm.notify("Fetch successful. Ready to add to DB.", "success")

                _schedule(_apply_success)
            except Exception as e:
                error_message = str(e)

                def _apply_error():
                    log.object = f"Error: {error_message}"
                    _ensure_tables()
                    summary_table.value = pd.DataFrame()
                    vm.notify(f"Fetch failed: {error_message}", "error")

                _schedule(_apply_error)
            finally:

                def _enable_fetch():
                    btn_fetch.disabled = False

                _schedule(_enable_fetch)

        threading.Thread(target=_do_fetch, daemon=True).start()

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
            _ensure_tables()
            
            if outcomes_df is not None:
                outcomes_table.value = outcomes_df
                details_df = _create_outcomes_details_table(outcomes_list)
                if details_df is not None:
                    details_table.value = details_df
                    details_title.object = f"### All Outcome Details ({len(details_df)} records)"
                else:
                    details_table.value = pd.DataFrame(columns=['Filter', 'Species', 'Type', 'Source', 'Offset', 'Outcome'])
                    details_title.object = "### No outcome details available"
            else:
                outcomes_table.value = pd.DataFrame()  # Use empty DataFrame instead of None
                details_table.value = pd.DataFrame(columns=['Filter', 'Species', 'Type', 'Source', 'Offset', 'Outcome'])
                details_title.object = ""

                    
            vm.notify("Labels added to DB.", "success")
        except Exception as e:
            vm.notify(f"Insert failed: {e}", "error")
            return
        




    btn_fetch.on_click(_fetch)
    btn_add.on_click(_add)

    _update_filter_section()

    @pn.depends(vm.obs.ensure("engine_state").param.version, watch=True)
    def _on_engine_state_change(_):
        _update_filter_section()

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
                filter_section,
                sizing_mode="stretch_width",
            ),
            pn.Column(
                pn.pane.Markdown("### Actions"),
                btn_fetch,
                btn_add,
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        ),
        pn.layout.Divider(),
        log,
        pn.layout.Divider(),
        pn.pane.Markdown("### Label Summary"),
        summary_container,
        pn.layout.Divider(),
        pn.pane.Markdown("### Database Insert Outcomes"),
        outcomes_container,
        pn.layout.Divider(),
        details_container,
        sizing_mode="stretch_both",
    )