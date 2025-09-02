import panel as pn
import pandas as pd
from ..view_model import ViewModel

from recognizer_workshop.app.services.db_labels.db_labels import LabelCounts, DatabaseSummary

def _build_status_view(vm: ViewModel):
    """Helper function to construct the main status view based on the current vm state."""
    
    if not vm.eng:
        return pn.pane.Markdown("### ℹ️ Engine not initialized. Please create the engine in the 'Config' tab first.")

    status_data = vm.eng.get_status()

    # --- 1. General Status Section ---
    general_status_view = pn.Column(
        "### General Status",
        pn.Row(
            pn.pane.Markdown("**Database Loaded:**", width=200),
            pn.pane.Markdown("✅ Yes" if status_data['db_loaded'] else "❌ No")
        ),
        pn.Row(
            pn.pane.Markdown("**Embedding Model Loaded:**", width=200),
            pn.pane.Markdown("✅ Yes" if status_data['embedding_model_loaded'] else "❌ No")
        ),
        pn.layout.Divider()
    )

    # --- 2. Configuration Section ---
    # Use the __str__ method we created for the AgileConfig dataclass
    config_str = str(status_data['config'])
    config_view = pn.Column(
        "### Configuration",
        pn.pane.Markdown(f"```\n{config_str}\n```", sizing_mode="stretch_width"),
        pn.layout.Divider()
    )

    # --- 3. Database Summary Section ---
    db_summary: DatabaseSummary = status_data.get('db_summary')
    if db_summary:
        # Create a table for the label counts
        labels_dict = db_summary.label_counts
        if labels_dict:
            # Reformat dict for better display in DataFrame
            table_data = []
            for label, counts in labels_dict.items():
                
                table_data.append({
                    'Label': counts.label_name,
                    'Positive': counts.positive,
                    'Negative': counts.negative
                })
            df = pd.DataFrame(table_data)
            labels_table = pn.widgets.DataFrame(df, sizing_mode="stretch_width", show_index=False)
        else:
            labels_table = pn.pane.Markdown("*No labels found in the database.*")

        db_summary_view = pn.Column(
            "### Database Summary",
            pn.pane.Markdown(f"**Total Embeddings:** {db_summary.total_embeddings}"),
            pn.pane.Markdown(f"**Datasets:** {', '.join(db_summary.datasets) if db_summary.datasets else 'None'}"),
            pn.layout.Divider(),
            "#### Label Counts",
            labels_table
        )
    else:
        db_summary_view = pn.Column("### Database Summary", pn.pane.Markdown("*Database not loaded.*"))

    # --- Assemble all sections ---
    return pn.Column(
        general_status_view,
        config_view,
        db_summary_view,
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
    
    # A container to hold the dynamically generated status view
    content_pane = pn.Column(_build_status_view(vm))

    def refresh_view(event):
        """Callback for the refresh button."""
        # Replace the contents of the pane with a freshly built view
        content_pane.objects = [_build_status_view(vm)]

    refresh_button.on_click(refresh_view)

    # The final layout for the tab is the button and the content pane
    return pn.Column(
        refresh_button,
        content_pane
    )
