from pathlib import Path

import panel as pn
import param


class HelpSystem(pn.custom.JSComponent):
    """Manages the application's help sidebar."""
    
    visible = param.Boolean(default=False, doc="Whether the help panel is visible.")
    scroll_to = param.String(default="", doc="The ID of the help section to scroll to.")
    #sidebar_open = param.Boolean(default=True, doc="Whether the sidebar is open.")
    sidebar_bump = param.Integer(default=0, doc="Bump to trigger sidebar open/close.")
        

    _esm = (Path(__file__).parent / "assets" / "js" / "help_system.js").read_text(
        encoding="utf-8"
    )

    
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