import html

import pandas as pd
import panel as pn

from recognizer_workshop.app.create_database.db_health_check import DatabaseStatus
from recognizer_workshop.app.services.version import get_version_status_string
from recognizer_workshop.ui.panel.view_model import ViewModel

def _build_status_view(vm: ViewModel):
    """Helper function to construct the main status view based on the current vm state."""
    
    if not vm.eng:
        return pn.pane.Markdown("### ℹ️ Engine not initialized. Please create the engine in the 'Config' tab first.")

    status_data = vm.eng.get_status()

    version_status_display = get_version_status_string(include_git_hash=True)

    # General Status Section
    general_status_view = pn.Column(
        pn.Row(
            pn.pane.HTML("<strong>Database Loaded:</strong>", width=200, margin=(0, 0)),
            pn.pane.HTML("✅ Yes" if status_data['db_loaded'] else "❌ No", margin=(0, 0)),
            margin=(0, 0),
        ),
        pn.Row(
            pn.pane.HTML("<strong>Embedding Runtime Ready:</strong>", width=200, margin=(0, 0)),
            pn.pane.HTML(
                "✅ Yes" if status_data.get('embedding_runtime_ready', status_data.get('embedding_model_loaded', False)) else "❌ No",
                margin=(0, 0)
            ),
            margin=(0, 0),
        ),
        pn.Row(
            pn.pane.HTML("<strong>Version:</strong>", width=200, margin=(0, 0)),
            pn.pane.HTML(version_status_display, margin=(0, 0)),
            margin=(0, 0),
        ),
    )

    # Configuration Section
    config_str = str(status_data['config'])
    config_view = pn.Column(
        "### Configuration",
        pn.pane.Markdown(f"```\n{config_str}\n```", sizing_mode="stretch_width"),
        pn.layout.Divider()
    )

    # Database Summary Section
    db_status: DatabaseStatus = status_data.get('database')
    db_metadata = status_data.get('db_metadata')
    if db_status:
        # Create a table for the label counts
        labels_dict = db_status.label_counts
        if labels_dict:
            # Reformat dict for better display in DataFrame
            table_data = []
            for _label, counts in labels_dict.items():
                
                table_data.append({
                    'Label': counts.label_name,
                    'Positive': counts.positive,
                    'Negative': counts.negative
                })
            df = pd.DataFrame(table_data)
            table_html = df.to_html(index=False, classes="status-table", border=0)
            labels_table = pn.pane.HTML(
                f"""
                <style>
                .status-table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 0;
                    font-size: 0.9rem;
                }}
                .status-table th,
                .status-table td {{
                    border: 1px solid #ddd;
                    padding: 4px 8px;
                    text-align: left;
                }}
                .status-table th {{
                    font-weight: 600;
                }}
                </style>
                {table_html}
                """,
                sizing_mode="stretch_width",
                margin=(0, 0),
            )
        else:
            labels_table = pn.pane.Markdown("*No labels found in the database.*", margin=(0, 0))

        db_summary_view = pn.Column(
            "### Database Summary",
            pn.pane.HTML(f"<strong>Total Embeddings:</strong> {db_status.total_embeddings}", margin=(0, 0)),
            pn.pane.HTML(f"<strong>Datasets:</strong> {', '.join(db_status.datasets) if db_status.datasets else 'None'}", margin=(0, 0)),
            pn.pane.HTML(
                f"<strong>Usearch Index:</strong> {db_status.index.format_size() if db_status.index else 'unknown'}",
                margin=(0, 0),
            ),
            labels_table,
            pn.pane.Markdown("#### Database Metadata"),
            pn.pane.HTML(
                f"""
                <pre style=\"max-height: 150px; overflow-y: auto; margin: 0; padding: 8px; border: 1px solid #ddd; border-radius: 4px;\">{html.escape(db_metadata.to_json(indent=2) if db_metadata else '{}')}</pre>
                """,
                sizing_mode="stretch_width",
                margin=(0, 0),
            ),
            margin=(0, 0),
        )
    else:
        db_summary_view = pn.Column("### Database Summary", pn.pane.Markdown("*Database not loaded.*"))

    top_summary_row = pn.Row(
        pn.Column(general_status_view, sizing_mode="stretch_width"),
        pn.Column(db_summary_view, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )

    return pn.Column(
        top_summary_row,
        config_view,
        sizing_mode="stretch_width"
    )

def build(vm: ViewModel):
    """
    Builds the Status tab, which displays information from the agile_engine.
    Includes a refresh button to update the view.
    """
    
    # A button to manually trigger a refresh of the status information
    refresh_button = pn.widgets.Button(
        name="🔄 Refresh", 
        button_type="primary", 
        width=120, 
        align='end'
    )
    
    content_pane = pn.Column(_build_status_view(vm))

    def refresh_view(_event):
        """Callback for the refresh button."""
        content_pane.objects = [_build_status_view(vm)]

    refresh_button.on_click(refresh_view)

    return pn.Column(
        refresh_button,
        content_pane
    )
