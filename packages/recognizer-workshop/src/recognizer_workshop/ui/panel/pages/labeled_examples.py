import threading

import pandas as pd
import panel as pn

from recognizer_workshop.app.services.baw_filters import format_filter_params_for_display
from recognizer_workshop.ui.panel.components.native.filter_editor import FilterEditor

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

    def _has_ready_labels() -> bool:
        return bool(vm.eng and vm.eng.has_fetched_labels_ready)

    def _sync_status_from_engine():
        if not vm.eng:
            return
        if vm.eng.labels_commit_state == 'error' and vm.eng.labels_commit_error:
            log.object = f"Error: {vm.eng.labels_commit_error}"
            return
        if vm.eng.labels_fetch_state == 'error' and vm.eng.labels_fetch_error:
            log.object = f"Error: {vm.eng.labels_fetch_error}"
            return
        if vm.eng.labels_commit_message:
            log.object = vm.eng.labels_commit_message
            return
        if vm.eng.labels_fetch_message:
            log.object = vm.eng.labels_fetch_message

    def _sync_add_button_state():
        is_busy = bool(
            vm.eng
            and (
                vm.eng.labels_fetch_state == 'running'
                or vm.eng.labels_commit_state == 'running'
            )
        )
        btn_fetch.disabled = is_busy
        btn_add.disabled = is_busy or not _has_ready_labels()

    engine_warning = pn.pane.Alert(
        "Initialize the engine on the 'Config' tab to set up filters.",
        alert_type="warning"
    )
    filter_section = pn.Column(sizing_mode="stretch_width")
    # The currently-mounted filter editor. Its ``value`` is the source of truth
    # for which filter the Fetch step uses.
    filter_editor = None

    def _update_filter_section():
        """Show filter editor when engine is ready, warning otherwise.

        The editor is rebuilt fresh each time the engine state changes so it
        reflects the latest DB metadata. It seeds its value from the most recent
        ``fetched_labels`` filter, then ``search_set``. Whatever is shown in the
        editor is what the Fetch step uses; nothing is persisted on save.
        """
        nonlocal filter_editor
        if not (vm.eng and vm.eng.is_db_loaded):
            filter_editor = None
            filter_section.objects = [engine_warning]
        else:
            filter_editor = FilterEditor(
                vm,
                dataset_priority=['fetched_labels', 'search_set'],
                fallback_to_legacy=True,
            )
            filter_section.objects = [filter_editor.view]

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
            filter_desc = format_filter_params_for_display(filter_params, f"Filter {i+1}")
            
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

    def _format_add_progress_event(event: dict) -> str | None:
        """Format add-label progress events into concise user-facing status text."""
        stage = event.get('stage')
        if event.get('message'):
            return str(event['message'])

        if stage == 'source_progress':
            seen = event.get('candidates_seen', 0)
            total = event.get('total_candidates', 0)
            downloaded = event.get('files_processed', 0)
            skipped = event.get('skipped_existing', 0)
            return f"Checking recordings {seen}/{total} (downloaded {downloaded}, skipped {skipped})"

        if stage == 'ingest_progress':
            processed = event.get('files_processed', 0)
            total = event.get('total_candidates')
            added_this = event.get('added_this_file', 0)
            total_added = event.get('total_inserted', 0)
            if total:
                return f"Ingesting parquet {processed}/{total} (+{added_this}, total added {total_added})"
            return f"Ingesting parquet ({processed} files, total added {total_added})"

        if stage == 'ingest_start':
            total = event.get('total_candidates')
            return f"Starting parquet ingest for {total} recordings..." if total else "Starting parquet ingest..."

        return None



    def _fetch(_):
        if not (vm.eng and vm.eng.is_db_loaded) or filter_editor is None:
            return vm.notify("Initialize the engine on the Config tab first.", "error")
        baw_filters = filter_editor.value
        if not baw_filters:
            return vm.notify("Set a recording filter before fetching.", "warn")
        log.object = "Fetching labels..."
        _sync_add_button_state()
        doc = pn.state.curdoc

        def _schedule(fn):
            if doc:
                doc.add_next_tick_callback(fn)
            else:
                # Fallback when no document is available; run directly.
                fn()

        def _do_fetch():
            try:
                result = vm.eng.fetch_labels(baw_filters)
                result_sets = result if isinstance(result, list) else []

                def _apply_success():
                    _ensure_tables()

                    summary_df = _create_fetch_summary_table(result_sets)
                    summary_table.value = summary_df if summary_df is not None else pd.DataFrame()

                    total_labels = vm.eng.fetched_labels_count if vm.eng else 0
                    _sync_status_from_engine()
                    _sync_add_button_state()
                    if total_labels > 0:
                        vm.notify("Fetch successful. Ready to add to DB.", "success")
                    else:
                        vm.notify("Fetch completed but no labels were generated for the current filters.", "warn")

                _schedule(_apply_success)
            except Exception as e:
                error_message = str(e)

                def _apply_error():
                    _sync_status_from_engine()
                    _ensure_tables()
                    summary_table.value = pd.DataFrame()
                    _sync_add_button_state()
                    vm.notify(f"Fetch failed: {error_message}", "error")

                _schedule(_apply_error)
            finally:
                _schedule(_sync_add_button_state)

        threading.Thread(target=_do_fetch, daemon=True).start()

    def _add(_):
        if not vm.eng: return vm.notify("Initialize first.", "error")
        if not _has_ready_labels():
            return vm.notify("No fetched labels are ready to add. Fetch labels first.", "warn")

        log.object = "Adding fetched labels to DB..."
        _sync_add_button_state()
        doc = pn.state.curdoc

        def _schedule(fn):
            if doc:
                doc.add_next_tick_callback(fn)
            else:
                fn()

        def _do_add():
            try:
                def _progress_callback(event: dict):
                    msg = _format_add_progress_event(event)
                    if not msg:
                        return

                    def _apply_progress():
                        log.object = msg
                        _sync_add_button_state()

                    _schedule(_apply_progress)

                outcomes_list = vm.eng.commit_fetched_labels(progress_callback=_progress_callback)

                def _apply_success():
                    # Show the results as a table with the outcome of adding each label
                    # each item in the outcomes list corresponds to a result_set, which corresponds to an audio recording filter.
                    # we display the filter in a human-readable way
                    for i, item in enumerate(outcomes_list):
                        item['filter_desc'] = format_filter_params_for_display(
                            item['filter_params'],
                            fallback_name=f"Filter {i+1}"
                        )

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
                    _sync_status_from_engine()
                    _sync_add_button_state()

                _schedule(_apply_success)
            except Exception as e:
                error_message = str(e)

                def _apply_error():
                    _sync_status_from_engine()
                    _sync_add_button_state()
                    vm.notify(f"Insert failed: {error_message}", "error")

                _schedule(_apply_error)
            finally:
                _schedule(_sync_add_button_state)

        threading.Thread(target=_do_add, daemon=True).start()
        




    btn_fetch.on_click(_fetch)
    btn_add.on_click(_add)

    _update_filter_section()
    _sync_status_from_engine()
    _sync_add_button_state()

    @pn.depends(vm.obs.ensure("engine_state").param.version, watch=True)
    def _on_engine_state_change(_):
        _update_filter_section()
        _sync_status_from_engine()
        _sync_add_button_state()

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