import panel as pn

def build(vm):
    learning_rate = pn.widgets.FloatInput(name="learning_rate", value=1e-3, step=1e-4)
    num_steps     = pn.widgets.IntInput(name="num_steps", value=101, start=1)
    batch_size    = pn.widgets.IntInput(name="batch_size", value=32, start=1)
    train_ratio   = pn.widgets.FloatInput(name="train_ratio", value=0.9, start=0.1, end=0.99, step=0.01)
    weak_neg_weight = pn.widgets.FloatInput(name="weak_neg_weight", value=0.05, step=0.01)
    weak_negs_batch = pn.widgets.IntInput(name="weak_negatives_batch_size", value=16, start=0)

    btn_train = pn.widgets.Button(name="Train classifier", button_type="success")
    log = pn.pane.Markdown("")

    cls_name = pn.widgets.TextInput(name="Classifier name", value="my_classifier")
    btn_save = pn.widgets.Button(name="Save classifier", button_type="primary")

    def _train(_):
        if not vm.eng: return vm.notify("Initialize first.", "error")
        try:
            log.object = "Training…"
            scores = vm.eng.train_classifier(
                learning_rate=learning_rate.value,
                weak_neg_weight=weak_neg_weight.value,
                num_steps=num_steps.value,
                train_ratio=train_ratio.value,
                batch_size=batch_size.value,
                weak_negatives_batch_size=weak_negs_batch.value,
            )
            log.object = f"top1={scores['top1_acc']:.3f}, roc_auc={scores['roc_auc']:.3f}, cmap={scores['cmap']:.3f}"
            vm.notify("Training complete.", "success")
        except Exception as e:
            log.object = f"Error: {e}"
            vm.notify(f"Training failed: {e}", "error")

    def _save(_):
        if not vm.eng: return vm.notify("Initialize first.", "error")
        try:
            out = (vm.eng.config.models_folder / f"{cls_name.value}.json").absolute()
            out.parent.mkdir(parents=True, exist_ok=True)
            vm.eng.classifier.save(out)
            vm.notify(f"Classifier saved: {out}", "success")
        except Exception as e:
            vm.notify(f"Save classifier failed: {e}", "error")

    btn_train.on_click(_train)
    btn_save.on_click(_save)

    return pn.Column(
        "## Train",
        pn.Row(learning_rate, num_steps, batch_size, train_ratio),
        pn.Row(weak_neg_weight, weak_negs_batch),
        pn.Row(btn_train),
        log,
        pn.Spacer(height=10),
        pn.Row(cls_name, btn_save),
        sizing_mode="stretch_both",
    )
