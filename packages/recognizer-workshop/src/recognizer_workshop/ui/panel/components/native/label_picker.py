import logging
import panel as pn

log = logging.getLogger(__name__)

def build_active_label_picker(vm) -> pn.Row:
    """
    A reactive picker that:
      - lists vm.eng.config.target_labels (List[LabelName])
      - stores/returns LabelName instances (not strings)
      - mirrors vm.eng.active_label
      - reacts to vm.obs['target_labels'] and vm.obs['active_label']
    """
    select = pn.widgets.Select(
        name="Active label",
        options={},        # {display_str -> LabelName}
        value=None,        # LabelName or None
        disabled=True,
        sizing_mode="stretch_width",
        width=280,
    )
    
    def _rebuild():
        eng = getattr(vm, "eng", None)
        if not eng or not getattr(eng, "config", None):
            select.options = {}
            select.value = None
            select.disabled = True
            return

        pool = eng.config.target_labels or []   # List[LabelName]
        log.debug(f"Rebuilding active label picker with {len(pool)} target labels...")
        # Options map friendly text -> LabelName
        opts = {str(ln): ln for ln in pool}
        select.options = opts
        select.disabled = not bool(opts)

        # Sync current selection
        current = getattr(eng, "active_label", None)  # LabelName or None
        if current in opts.values():
            # Select compares by identity; pick key for same object if present
            select.value = current
        elif opts:
            # default to first
            first_ln = next(iter(opts.values()))
            select.value = first_ln
            eng.active_label = first_ln
            vm.obs.bump("active_label")
        else:
            select.value = None

    # React to changes in the available labels
    @pn.depends(vm.obs.ensure("target_labels").param.version, watch=True)
    def _on_target_labels_changed(_version):
        log.info("Rebuilding active label picker due to target_labels change...")
        _rebuild()

    # React to external changes of the active label
    @pn.depends(vm.obs.ensure("active_label").param.version, watch=True)
    def _on_active_label_changed(_version):
        log.info("Active label changed externally, updating picker...")
        eng = getattr(vm, "eng", None)
        if not eng:
            return
        
        # if there is an active label and it's not what is already selected
        if eng.active_label is not None and select.value is not eng.active_label:
            # If it's one of the options, sync it into the Select
            if eng.active_label in select.options.values():
                select.value = eng.active_label


    # User changed selection -> write back LabelName and bump
    def _on_user_change(event):
        eng = getattr(vm, "eng", None)
        if not eng:
            return
        
        if event.new is event.old:
            return
        
        eng.active_label = event.new    # LabelName
        # this will trigger our own _on_active_label_changed, which is fine. It should do nothing
        vm.obs.bump("active_label") 

    select.param.watch(_on_user_change, "value")

    # Initial build
    _rebuild()

    return pn.Row(select, sizing_mode="stretch_width")
