import ipywidgets as widgets
from IPython.display import display

def label_selector(baw_api, binding, initial_value=None):
    """
    Value is a dict with a single key 'value'. The value of this item
    should be updated to the label selected by the user whenever it change. 
    It should also be used to initialize the label_selector widget.
    """
    
    response = baw_api.baw_filter('/tags', {'is_taxonomic': {'eq': True}}, qsp="disable_paging=true").get('data', [])
    possible_labels = [l['text'] for l in response]
    # Sort them for a better user experience
    possible_labels.sort()
    
    def on_value_change(change):
        binding['value'] = change['new']
        print(f"Label is now: {binding['value']}")

    # initial value is set to these in order of priority
    # initial_value if valid
    # binding['value'] if it exists and is valid
    # empty string if neither is valid
    if initial_value and initial_value in possible_labels:
        value = initial_value
    elif binding.get('value') in possible_labels:
        value = binding['value']
    else:
        value = ''

    #    - 'ensure_option=True': This is important! It forces the user to select
    #      a valid option from the list. It won't let them submit a custom value.
    label_selector_widget = widgets.Combobox(
        value=value, # The default starting value
        placeholder='Start typing to find a label ...',
        options=possible_labels,
        description='Classifier Label:',
        ensure_option=True,
        disabled=False,
        style={'description_width': 'initial'} # Helps the description fit
    )

    label_selector_widget.observe(on_value_change, names='value')

    # 3. Display the widget in the notebook output.
    print("Please select a label from the list below:")
    display(label_selector_widget)
    return label_selector_widget