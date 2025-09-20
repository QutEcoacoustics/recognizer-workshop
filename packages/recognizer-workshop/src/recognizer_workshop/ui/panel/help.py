"""Help system for the recognizer workshop UI."""

import panel as pn
import param
from .help_content import HELP_HTML


class HelpSystem(param.Parameterized):
    """Global help system for the application."""
    
    # Global state for help panel visibility
    visible = param.Boolean(default=False, doc="Whether the help panel is visible")
    current_section = param.String(default="", doc="Current section to scroll to")
    
    def __init__(self, **params):
        super().__init__(**params)
        self._help_content = self._create_help_content()
        self._js_executor = None
    
    def _create_help_content(self):
        """Create the main help content with proper HTML IDs for scrolling."""
        # Create scrollable content with JavaScript injection
        content_pane = pn.pane.HTML(HELP_HTML, sizing_mode="stretch_width")
        
        # JavaScript to handle scrolling - improved to work with Panel's scroll containers
        scroll_js = """
        <script>
        window.scrollToSection = function(sectionId) {
            console.log('Attempting to scroll to:', sectionId);
            setTimeout(function() {
                const element = document.getElementById(sectionId);
                if (!element) {
                    console.log('Section not found:', sectionId);
                    return;
                }
                
                // Find the scrollable container (Panel creates divs with overflow)
                let scrollContainer = element.closest('[style*="overflow"]');
                if (!scrollContainer) {
                    // Fallback: look for common Panel scroll container classes/attributes
                    scrollContainer = element.closest('.bk-root') || 
                                    element.closest('[data-root-id]') ||
                                    document.querySelector('.bk-root') ||
                                    window;
                }
                
                // Calculate position relative to the scroll container
                const elementRect = element.getBoundingClientRect();
                const containerRect = scrollContainer === window ? 
                    {top: 0, left: 0} : scrollContainer.getBoundingClientRect();
                
                if (scrollContainer === window) {
                    // Standard document scroll
                    element.scrollIntoView({ behavior: 'smooth', block: 'start' });
                } else {
                    // Scroll within the container
                    const targetScrollTop = scrollContainer.scrollTop + 
                        (elementRect.top - containerRect.top) - 20; // 20px offset
                    
                    scrollContainer.scrollTo({
                        top: targetScrollTop,
                        behavior: 'smooth'
                    });
                }
                
                // Add highlight effect
                element.style.transition = 'background-color 0.5s ease';
                element.style.backgroundColor = '#fff3cd';
                element.style.padding = '10px';
                element.style.borderRadius = '8px';
                
                // Remove highlight after 3 seconds
                setTimeout(function() {
                    element.style.backgroundColor = '';
                    element.style.padding = '';
                }, 3000);
                
            }, 300); // Increased delay to ensure Panel rendering is complete
        };
        </script>
        """
        
        self._js_executor = pn.pane.HTML(scroll_js, width=0, height=0)
        
        return pn.Column(self._js_executor, content_pane, sizing_mode="stretch_width")

    def show_help_section(self, section_id):
        """Show the help panel and scroll to a specific section."""
        # Make help visible first
        self.visible = True
        
        # Set the current section for tracking
        self.current_section = section_id
        
        # Create a new JavaScript execution pane to trigger the scroll
        js_code = f"""
        <script>
        setTimeout(function() {{
            if (window.scrollToSection) {{
                window.scrollToSection('{section_id}');
            }} else {{
                console.log('scrollToSection function not available yet');
                // Fallback: try direct scrollIntoView
                setTimeout(function() {{
                    const element = document.getElementById('{section_id}');
                    if (element) {{
                        element.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                    }}
                }}, 100);
            }}
        }}, 500); // Longer delay to ensure help panel is fully rendered
        </script>
        """
        
        # Create a temporary execution pane that will run the JavaScript
        temp_js = pn.pane.HTML(js_code, width=0, height=0)
        
        # Add it temporarily to trigger execution, then remove it
        if hasattr(self, '_help_content') and self._help_content:
            self._help_content.append(temp_js)
            # Remove it after a short delay
            def remove_temp():
                if temp_js in self._help_content:
                    self._help_content.remove(temp_js)
            pn.state.add_periodic_callback(remove_temp, period=1000, count=1)

    def _toggle_help(self, event=None):
        """Toggle the help panel visibility."""
        self.visible = not self.visible
    
    def get_panel(self):
        """Get the help panel - returns empty Column when hidden, full panel when visible."""
        if not self.visible:
            # Return a minimal toggle button when hidden
            toggle_button = pn.widgets.Button(
                name="❓ Help",
                button_type="primary",
                width=100,
                margin=(5, 5)
            )
            toggle_button.on_click(self._toggle_help)
            
            return pn.Column(
                toggle_button,
                width=110,
                margin=(5, 5)
            )
        else:
            # Return full help panel when visible
            toggle_button = pn.widgets.Button(
                name="❌ Hide Help",
                button_type="primary",
                width=120,
                margin=(5, 5)
            )
            toggle_button.on_click(self._toggle_help)
            
            return pn.Column(
                toggle_button,
                pn.Spacer(height=10),
                pn.pane.HTML("<h3>Workshop Help</h3>", margin=(0, 10)),
                self._help_content,
                width=400,
                margin=(5, 10),
                scroll=True,
                height=600
            )
    
    def create_help_button(self, section_id, tooltip="Get help"):
        """Create a small help button that scrolls to a specific help section."""
        button = pn.widgets.Button(
            name="❓",
            width=30,
            height=30,
            button_type="light",
            description=tooltip
        )
        
        def on_click(event):
            self.show_help_section(section_id)
        
        button.on_click(on_click)
        return button
