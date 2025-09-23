import asyncio
from contextlib import contextmanager

import numpy as np
import pandas as pd
import panel as pn
import tqdm

from recognizer_workshop.ui.panel.utils.patch_tqdm import patch_tqdm


def build(vm):
    # Create help buttons
    train_help_btn = vm.help_system.create_help_button("help-train-overview", "Help: Training")
    algorithm_params_help_btn = vm.help_system.create_help_button("help-training-algorithm-params", "Help: Algorithm Parameters")
    labeled_data_params_help_btn = vm.help_system.create_help_button("help-labeled-data-params", "Help: Labeled Data Parameters")

    learning_rate = pn.widgets.FloatInput(name="learning_rate", value=1e-3, step=1e-4)
    num_steps     = pn.widgets.IntInput(name="num_steps", value=101, start=1)
    batch_size    = pn.widgets.IntInput(name="batch_size", value=32, start=1)
    train_ratio   = pn.widgets.FloatInput(name="train_ratio", value=0.9, start=0.1, end=0.99, step=0.01)
    weak_neg_weight = pn.widgets.FloatInput(name="weak_neg_weight", value=0.05, step=0.01)
    weak_negs_batch = pn.widgets.IntInput(name="weak_negatives_batch_size", value=16, start=0)

    btn_train = pn.widgets.Button(name="Train classifier", button_type="success")
    
    progress_bar = pn.widgets.Progress(name='Training Progress', active=True, visible=False, sizing_mode="stretch_width")

    log = pn.pane.Markdown("")

    results_view = pn.Column(
        pn.pane.Markdown("Training results will be shown here."),
        sizing_mode="stretch_width"
    )

    cls_name = pn.widgets.TextInput(name="Classifier name", value="my_classifier")
    btn_save = pn.widgets.Button(name="Save classifier", button_type="primary")

    async def _train(_):
        if not vm.eng: return vm.notify("Initialize first.", "error")

        # Disable the button and show the progress bar to give user feedback
        btn_train.disabled = True
        progress_bar.visible = True
        results_view.objects = [pn.pane.Markdown("⏳ Training in progress... please wait.")]

        def run_training():
            return vm.eng.train_classifier(
                learning_rate=learning_rate.value,
                weak_neg_weight=weak_neg_weight.value,
                num_steps=num_steps.value,
                train_ratio=train_ratio.value,
                batch_size=batch_size.value,
                weak_negatives_batch_size=weak_negs_batch.value,
            )
        

        try:

            progress_bar.bar_color = 'success'
            progress_bar.value = 0 
            with patch_tqdm(progress_bar):
                scores = await asyncio.to_thread(run_training)

            scores_df = format_scores_df(scores)

            # Display the DataFrame in our results view
            results_view.objects = [pn.widgets.DataFrame(scores_df, sizing_mode="stretch_width")]
            vm.notify("Training complete.", "success")
            
        except Exception as e:
            results_view.objects = [pn.pane.Alert(f"**Error:** {e}", alert_type="danger")]
            vm.notify(f"Training failed: {e}", "error")
        finally:
            # ALWAYS re-enable the button and hide the progress bar
            btn_train.disabled = False
            progress_bar.visible = False


    def _save(_):
        if not vm.eng: return vm.notify("Initialize first.", "error")
        if not hasattr(vm.eng, 'classifier') or vm.eng.classifier is None:
            return vm.notify("No classifier has been trained yet.", "warn")
        try:
            out = (vm.eng.config.models_folder / f"{cls_name.value}.json").absolute()
            out.parent.mkdir(parents=True, exist_ok=True)
            vm.eng.save_classifier(out)
            vm.notify(f"Classifier saved: {out}", "success")
        except Exception as e:
            vm.notify(f"Save classifier failed: {e}", "error")

    btn_train.on_click(_train)
    btn_save.on_click(_save)

    # training params
    hyperparameters = pn.Card(
        pn.Row("Algorithm Parameters", pn.Spacer(), algorithm_params_help_btn),
        pn.Row(learning_rate, num_steps, batch_size, train_ratio),
        pn.Row("Labeled Data Parameters", pn.Spacer(), labeled_data_params_help_btn),
        pn.Row(weak_neg_weight, weak_negs_batch),
        title="Hyperparameters",
        collapsible=False
    )

    return pn.Column(
        pn.Row("## Train Classifier", pn.Spacer(), train_help_btn),
        hyperparameters,
        pn.Row(btn_train),
        progress_bar,
        pn.layout.Divider(),
        pn.pane.Markdown("### Results"),
        results_view,
        pn.layout.Divider(),
        pn.pane.Markdown("### Save Model"),
        pn.Row(cls_name, btn_save),
        sizing_mode="stretch_both",
    )





def format_scores_df(scores: dict) -> pd.DataFrame:
    """
    Filters, formats, and prepares the training scores dictionary for display.
    """

    metrics_to_show = [
        'top1_acc',
        'roc_auc',
        'cmap',
        'roc_auc_individual',
        'cmap_individual'
    ]

    filtered_scores = {key: scores[key] for key in metrics_to_show if key in scores}

    # Some metrics are overall, and some per class. 
    # We don't need the per class ones if there is only 1 class. 
    if len(filtered_scores.get('cmap_individual', [])) <= 1:
        filtered_scores.pop('roc_auc_individual', None)
        filtered_scores.pop('cmap_individual', None)

    df = pd.DataFrame.from_dict(filtered_scores, orient='index', columns=['Score'])
    df.index.name = 'Metric'


    def format_value(value):
        if isinstance(value, (int, float, np.number)):
            return f'{value:.3f}'
        elif isinstance(value, (list, np.ndarray)):
            return ', '.join([f'{v:.3f}' for v in value])
        return str(value)

    df['Score'] = df['Score'].apply(format_value)
    return df




