import ipywidgets as widgets
from IPython.display import display, clear_output

def label_selector(baw_api, binding, initial_species=None, initial_call_types=None):
    """
    Displays a form with a species selector and a text field for call types.
    On button click, it updates the 'binding' dictionary with the submitted values.

    Args:
        baw_api: The API object to fetch species labels.
        binding (dict): A dictionary that will be updated with form values on submit.
          - 'species' (str)
          - 'call_types' (list)
        initial_species (str, optional): The initial value for the species selector.
        initial_call_types (str, optional): The initial value for the call types text field.

    The initial values are mainly for debugging and running the notebook as a non-interactive script.
    """
    
    response = baw_api.baw_filter('/tags', {'is_taxonomic': {'eq': True}}, qsp="disable_paging=true").get('data', [])
    possible_labels = sorted([l['text'] for l in response])
    
    # Determine the starting value for the species selector
    if initial_species and initial_species in possible_labels:
        species_value = initial_species
    elif binding.get('species') in possible_labels:
        species_value = binding.get('species')
    else:
        species_value = ''

    species_selector_widget = widgets.Combobox(
        value=species_value,
        placeholder='Start typing to find a species...',
        options=possible_labels,
        description='Species Label:',
        ensure_option=True,
        disabled=False,
        style={'description_width': 'initial'}
    )

    initial_call_types_str = ''
    if initial_call_types:
        if isinstance(initial_call_types, list):
            # Join the list into a string to pre-populate the text field
            initial_call_types_str = ', '.join(initial_call_types)
        else:
            initial_call_types_str = initial_call_types
    elif 'call_types' in binding and isinstance(binding.get('call_types'), list):
        # Join the list into a string to pre-populate the text field
        initial_call_types_str = ', '.join(binding['call_types'])

    call_types_widget = widgets.Text(
        value=initial_call_types_str,
        placeholder='Enter call types (e.g., call, song)...',
        description='Call Types:',
        disabled=False,
        style={'description_width': 'initial'}
    )

    submit_button = widgets.Button(
        description='Submit',
        disabled=False,
        button_style='success',
        tooltip='Submit the selected species and call types',
        icon='check'
    )
    
    output_widget = widgets.Output()

    def on_button_clicked(b):
        # Update the species value
        binding['species'] = species_selector_widget.value
        
        # NEW: Get the string from the text widget and convert it to a list
        call_types_str = call_types_widget.value
        if call_types_str:

            call_types = [item.strip() for item in call_types_str.split(',') if item.strip()]
            # convert to lower case except for the first letter which should be capitalized
            call_types = [ct.lower().capitalize() for ct in call_types]
            binding['call_types'] = call_types
        else:
            # If the input is empty, store an empty list
            binding['call_types'] = []
        
        # Provide feedback to the user in the output area
        with output_widget:
            clear_output(wait=True)
            print("✔️ Submitted successfully!")
            print(f"  - Species: {binding['species']}")
            # Corrected print statement to show the list of call types
            print(f"  - Call Types: {binding['call_types']}")
            
    # Register the click handler with the button
    submit_button.on_click(on_button_clicked)

    # --- 6. Arrange all widgets vertically in a VBox container ---
    form_container = widgets.VBox([
        species_selector_widget, 
        call_types_widget, 
        submit_button, 
        output_widget
    ])

    # Display the entire form
    print("Please select a species, add a comma-separated list of call types if applicable, and submit:")
    display(form_container)

    if initial_species or initial_call_types:
        on_button_clicked(None)
    
    return form_container