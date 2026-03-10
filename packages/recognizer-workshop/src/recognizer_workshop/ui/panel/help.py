from pathlib import Path

import panel as pn
import param


class HelpSystem(pn.custom.JSComponent):
    """Manages the application's help sidebar."""
    
    visible = param.Boolean(default=False, doc="Whether the help panel is visible.")
    scroll_to = param.String(default="", doc="The ID of the help section to scroll to.")
    #sidebar_open = param.Boolean(default=True, doc="Whether the sidebar is open.")
    sidebar_bump = param.Integer(default=0, doc="Bump to trigger sidebar open/close.")
        

    _esm = """
    export function render({ model }) {
        console.log('HelpSystem JSComponent initialized');
        model.on('sidebar_bump', () => {
            // Always open the sidebar when bumped
            openNav();
        });

        model.on('scroll_to', () => {
            console.log('scroll_to changed to:', model.scroll_to);
            if (model.scroll_to) {
                // Wait a bit for the sidebar to finish opening
                setTimeout(() => {
                    // First try normal document
                    let element = document.getElementById(model.scroll_to);
                    console.log('Found in main document:', element);
                    
                    // If not found, look in shadow roots
                    if (!element) {
                        console.log('Looking in shadow roots...');
                        const htmlElements = document.querySelectorAll('.bk-panel-models-markup-HTML');
                        
                        for (const htmlEl of htmlElements) {
                            if (htmlEl.shadowRoot) {
                                console.log('Checking shadow root:', htmlEl.shadowRoot);
                                element = htmlEl.shadowRoot.getElementById(model.scroll_to);
                                if (element) {
                                    console.log('Found in shadow root:', element);
                                    break;
                                }
                            }
                        }
                    }
                    
                    if (element) {
                        element.scrollIntoView({ 
                            behavior: 'smooth', 
                            block: 'start',
                            inline: 'nearest'
                        });
                        
                        // Optional: highlight the section briefly
                        element.style.transition = 'background-color 0.5s';
                        element.style.backgroundColor = '#fff3cd';
                        setTimeout(() => { 
                            element.style.backgroundColor = ''; 
                        }, 2000);
                    } else {
                        console.log('Element not found anywhere with ID:', model.scroll_to);
                    }
                }, 300);
            }
        });
    }
    """

    
    def __init__(self, **params):
        super().__init__(**params)
        self.content = self._create_help_content()

    def _load_help_css(self):
        """Load the CSS styles for the help content."""
        css_file_path = Path(__file__).parent / "help_content.css"
        try:
            with open(css_file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return ""
        except Exception as e:
            print(f"Warning: Error loading help CSS: {e}")
            return ""

    def _load_help_html(self):
        """Load the help content from the HTML file."""
        help_file_path = Path(__file__).parent / "help_content.html"
        try:
            with open(help_file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Load CSS and wrap it in a style tag
            css_content = self._load_help_css()
            if css_content:
                styled_content = f"<style>{css_content}</style>{html_content}"
                return styled_content
            else:
                return html_content
                
        except FileNotFoundError:
            return "<div>Help content not found. Please check the help_content.html file.</div>"
        except Exception as e:
            return f"<div>Error loading help content: {e}</div>"
    
    def _create_help_content(self):
        help_html = self._load_help_html()
        return pn.pane.HTML(help_html, sizing_mode="stretch_width")
        
    
    def show_help_section(self, section_id):
        self.visible = True
        self.scroll_to = section_id

    def create_help_button(self, section_id, tooltip="Get help"):
        """
        Creates a ? button that opens the sidebar and triggers a scroll.
        args: section_id (str): the element id of the help section to scroll to when clicked
        """
        button = pn.widgets.Button(name="?", width=40, button_type='light', description=tooltip)
        
        def open_and_scroll(_):
            # Tell the ViewModel which section to scroll to
            self.scroll_to = section_id
            self.sidebar_bump += 1
         
        
        button.on_click(open_and_scroll)
        return button