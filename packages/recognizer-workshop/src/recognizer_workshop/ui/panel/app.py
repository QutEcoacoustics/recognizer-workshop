import panel as pn
from recognizer_workshop.ui.panel.view_model import ViewModel
from recognizer_workshop.ui.panel.pages import config, labeled_examples, search, train, refine, inference, status

import logging
from recognizer_workshop.ui.panel.utils import PanelLogHandler


pn.extension()

def build_app():
    vm = ViewModel()
    tabs = pn.Tabs(
        ("Config",            config.build(vm)),
        ("Labelled examples", labeled_examples.build(vm)),
        ("Search",            search.build(vm)),
        ("Train",             train.build(vm)),
        ("Refine",            refine.build(vm)),
        ("Inference",         inference.build(vm)),
        ("Status",            status.build(vm)), 
        ("Logs",              vm.log_pane), 
        sizing_mode="stretch_both",
    )
    return pn.Column(
        "# Agile Modelling",
        tabs,
        sizing_mode="stretch_both",
    )

app = build_app()
app.servable(title="Recognizer Workshop")
