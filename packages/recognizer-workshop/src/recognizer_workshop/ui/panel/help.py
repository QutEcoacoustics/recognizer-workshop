"""Help system for the recognizer workshop UI."""

import panel as pn
import param


class HelpSystem(param.Parameterized):
    """Global help system for the application."""
    
    # Global state for help panel visibility
    visible = param.Boolean(default=False, doc="Whether the help panel is visible")
    
    def __init__(self, **params):
        super().__init__(**params)
        self._help_content = self._create_help_content()
        self._help_panel = self._create_help_panel()
    
    def _create_help_content(self):
        """Create the main help content with proper HTML IDs for scrolling."""
        help_html = """
        <div style="padding: 20px; font-family: Arial, sans-serif;">
            <h2>🚀 Recognizer Workshop Help</h2>
            <p>Welcome to the comprehensive help guide for the Recognizer Workshop. Use the navigation below or click the ? buttons throughout the app to jump to specific sections.</p>
            
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin: 20px 0;">
                <h3>📑 Quick Navigation</h3>
                <ul style="columns: 2; column-gap: 30px;">
                    <li><a href="#help-config-overview">Configuration</a></li>
                    <li><a href="#help-database-overview">Database</a></li>
                    <li><a href="#help-labeled-examples-overview">Labeled Examples</a></li>
                    <li><a href="#help-search-overview">Search</a></li>
                    <li><a href="#help-explore-overview">Explore</a></li>
                    <li><a href="#help-train-overview">Train</a></li>
                    <li><a href="#help-refine-overview">Refine</a></li>
                    <li><a href="#help-inference-overview">Inference</a></li>
                </ul>
            </div>
            
            <!-- CONFIGURATION PAGE -->
            <div id="help-config-overview" style="margin-top: 40px; border-left: 4px solid #007bff; padding-left: 20px;">
                <h2>📋 Configuration Page</h2>
                <p>The Configuration page is your starting point. Set up your project settings, API connections, and folder paths here before proceeding to other tabs.</p>
                
                <div id="help-project-file" style="margin-top: 20px;">
                    <h4>📄 Project File Management</h4>
                    <ul>
                        <li><strong>New:</strong> Create a fresh configuration</li>
                        <li><strong>Open:</strong> Load an existing configuration file</li>
                        <li><strong>Save:</strong> Save current settings (auto-saves as you type)</li>
                        <li><strong>Save As:</strong> Save to a new location</li>
                    </ul>
                </div>
                
                <div id="help-annotator-id" style="margin-top: 25px;">
                    <h4>👤 Annotator ID</h4>
                    <p>Your unique identifier as an annotator. This helps track who created which annotations and labels in the system.</p>
                    <ul>
                        <li>Use a consistent identifier across all your projects</li>
                        <li>Avoid spaces or special characters</li>
                        <li>Examples: "john_smith", "researcher_01", "lab_technician_a"</li>
                    </ul>
                </div>
                
                <div id="help-api-domain" style="margin-top: 25px;">
                    <h4>🌐 API Domain</h4>
                    <p>The base URL for your Workbench API server.</p>
                    <ul>
                        <li>Default: https://api.ecosounds.org</li>
                        <li>For local installations, use your server's URL</li>
                        <li>Must include the protocol (http:// or https://)</li>
                        <li>Test the connection after entering</li>
                    </ul>
                </div>
                
                <div id="help-api-token" style="margin-top: 25px;">
                    <h4>🔑 API Token</h4>
                    <p>Your personal authentication token for accessing the Workbench API.</p>
                    <ul>
                        <li>Found in your user profile on the Workbench website</li>
                        <li>Keep this token secure and private</li>
                        <li>Required for downloading audio and uploading results</li>
                        <li>Token format: usually a long alphanumeric string</li>
                    </ul>
                </div>
                
                <div id="help-folder-config" style="margin-top: 25px;">
                    <h4>📁 Advanced Folder Configuration</h4>
                    <p>Customize where your project files are stored. Leave empty to use sensible defaults.</p>
                    <ul>
                        <li><strong>DB Path:</strong> Local database files (default: ./db)</li>
                        <li><strong>Labeled Examples:</strong> Training examples (default: ./labeled_examples)</li>
                        <li><strong>Models:</strong> Trained models (default: ./models)</li>
                        <li><strong>Predictions:</strong> Results (default: ./predictions)</li>
                    </ul>
                    <p><em>💡 Tip: Use the folder buttons to browse and select directories.</em></p>
                </div>
                
                <div id="help-engine" style="margin-top: 25px;">
                    <h4>🚀 Engine Initialization</h4>
                    <p>The engine sets up all core components. Click "Start Engine" after configuring your settings.</p>
                    <ul>
                        <li>Tests your API connection</li>
                        <li>Sets up the local database</li>
                        <li>Fetches available tags from the Workbench</li>
                        <li>Enables the other tabs in the application</li>
                    </ul>
                </div>
                
                <div id="help-target-labels" style="margin-top: 25px;">
                    <h4>🎯 Target Labels Configuration</h4>
                    <p>Select which species or sound types you want to detect and work with.</p>
                    <ul>
                        <li>Choose from tags available in your Workbench</li>
                        <li>Add custom labels if needed</li>
                        <li>These labels determine what the system will learn to recognize</li>
                        <li>You can modify this selection later</li>
                    </ul>
                </div>
            </div>
            
            <!-- DATABASE PAGE -->
            <div id="help-database-overview" style="margin-top: 40px; border-left: 4px solid #28a745; padding-left: 20px;">
                <h2>🗄️ Database Page</h2>
                <p>Manage your local audio database. Download audio files from the Workbench and organize them for analysis.</p>
                
                <div id="help-site-selection" style="margin-top: 20px;">
                    <h4>📍 Site Selection</h4>
                    <p>Choose which recording sites to include in your analysis.</p>
                    <ul>
                        <li>Browse available sites from your Workbench</li>
                        <li>Filter by location, date, or other criteria</li>
                        <li>Select sites relevant to your research question</li>
                    </ul>
                </div>
                
                <div id="help-audio-download" style="margin-top: 25px;">
                    <h4>⬇️ Audio Download</h4>
                    <p>Download audio files to your local machine for processing.</p>
                    <ul>
                        <li>Choose date ranges and time periods</li>
                        <li>Set quality and format preferences</li>
                        <li>Monitor download progress</li>
                        <li>Resume interrupted downloads</li>
                    </ul>
                </div>
                
                <div id="help-database-management" style="margin-top: 25px;">
                    <h4>🔧 Database Management</h4>
                    <p>Organize and maintain your local audio database.</p>
                    <ul>
                        <li>View downloaded files and metadata</li>
                        <li>Check file integrity</li>
                        <li>Remove corrupted or unwanted files</li>
                        <li>Monitor storage usage</li>
                    </ul>
                </div>
            </div>
            
            <!-- LABELED EXAMPLES PAGE -->
            <div id="help-labeled-examples-overview" style="margin-top: 40px; border-left: 4px solid #ffc107; padding-left: 20px;">
                <h2>🏷️ Labeled Examples Page</h2>
                <p>Create and manage training examples for your machine learning models.</p>
                
                <div id="help-example-creation" style="margin-top: 20px;">
                    <h4>➕ Creating Examples</h4>
                    <p>Generate positive and negative examples for each target species.</p>
                    <ul>
                        <li>Select audio segments containing your target species</li>
                        <li>Mark background/noise segments as negative examples</li>
                        <li>Aim for balanced datasets</li>
                        <li>Quality over quantity - focus on clear examples</li>
                    </ul>
                </div>
                
                <div id="help-example-review" style="margin-top: 25px;">
                    <h4>�️ Reviewing Examples</h4>
                    <p>Quality control your training data before model training.</p>
                    <ul>
                        <li>Listen to examples to verify labels</li>
                        <li>Remove mislabeled or ambiguous examples</li>
                        <li>Check for class balance</li>
                        <li>Ensure diverse acoustic conditions</li>
                    </ul>
                </div>
                
                <div id="help-example-export" style="margin-top: 25px;">
                    <h4>📤 Export Examples</h4>
                    <p>Prepare your labeled data for training.</p>
                    <ul>
                        <li>Export in various formats (CSV, JSON)</li>
                        <li>Include metadata and annotations</li>
                        <li>Split into training/validation sets</li>
                    </ul>
                </div>
            </div>
            
            <!-- SEARCH PAGE -->
            <div id="help-search-overview" style="margin-top: 40px; border-left: 4px solid #17a2b8; padding-left: 20px;">
                <h2>🔍 Search Page</h2>
                <p>Find specific audio patterns and events in your database using various search methods.</p>
                
                <div id="help-similarity-search" style="margin-top: 20px;">
                    <h4>🎯 Similarity Search</h4>
                    <p>Find audio segments similar to a reference example.</p>
                    <ul>
                        <li>Upload a reference audio file</li>
                        <li>Adjust similarity threshold</li>
                        <li>Review and validate results</li>
                        <li>Add findings to labeled examples</li>
                    </ul>
                </div>
                
                <div id="help-keyword-search" style="margin-top: 25px;">
                    <h4>🔤 Keyword Search</h4>
                    <p>Search using metadata and annotations.</p>
                    <ul>
                        <li>Search by species names</li>
                        <li>Filter by date, time, location</li>
                        <li>Use existing tags and labels</li>
                        <li>Combine multiple criteria</li>
                    </ul>
                </div>
                
                <div id="help-advanced-search" style="margin-top: 25px;">
                    <h4>⚙️ Advanced Search</h4>
                    <p>Complex queries using multiple parameters.</p>
                    <ul>
                        <li>Frequency range filtering</li>
                        <li>Duration constraints</li>
                        <li>Acoustic feature matching</li>
                        <li>Boolean operators (AND, OR, NOT)</li>
                    </ul>
                </div>
            </div>
            
            <!-- EXPLORE PAGE -->
            <div id="help-explore-overview" style="margin-top: 40px; border-left: 4px solid #6f42c1; padding-left: 20px;">
                <h2>🔍 Explore Page</h2>
                <p>Visualize your audio data in embedding space and discover patterns.</p>
                
                <div id="help-umap-visualization" style="margin-top: 20px;">
                    <h4>📊 UMAP Visualization</h4>
                    <p>Interactive 2D representation of your audio embeddings.</p>
                    <ul>
                        <li>Each point represents an audio segment</li>
                        <li>Similar sounds cluster together</li>
                        <li>Click points to hear audio samples</li>
                        <li>Zoom and pan to explore different regions</li>
                    </ul>
                </div>
                
                <div id="help-clustering" style="margin-top: 25px;">
                    <h4>🎯 Pattern Discovery</h4>
                    <p>Identify natural groupings in your audio data.</p>
                    <ul>
                        <li>Look for tight clusters of points</li>
                        <li>Investigate outliers and sparse regions</li>
                        <li>Color-code by known labels to validate clustering</li>
                        <li>Use findings to improve training data</li>
                    </ul>
                </div>
                
                <div id="help-embedding-analysis" style="margin-top: 25px;">
                    <h4>🧬 Embedding Analysis</h4>
                    <p>Understand how the AI represents different sounds.</p>
                    <ul>
                        <li>Compare embeddings of known species</li>
                        <li>Identify confusable species (close in space)</li>
                        <li>Find potential mislabels (outliers from their group)</li>
                        <li>Discover new acoustic patterns</li>
                    </ul>
                </div>
            </div>
            
            <!-- TRAIN PAGE -->
            <div id="help-train-overview" style="margin-top: 40px; border-left: 4px solid #dc3545; padding-left: 20px;">
                <h2>🎓 Train Page</h2>
                <p>Train machine learning models using your labeled examples.</p>
                
                <div id="help-model-selection" style="margin-top: 20px;">
                    <h4>🤖 Model Selection</h4>
                    <p>Choose the right model architecture for your data.</p>
                    <ul>
                        <li>Pre-trained models for quick results</li>
                        <li>Custom architectures for specialized needs</li>
                        <li>Consider data size and computational resources</li>
                        <li>Start simple, then experiment with complexity</li>
                    </ul>
                </div>
                
                <div id="help-training-params" style="margin-top: 25px;">
                    <h4>⚙️ Training Parameters</h4>
                    <p>Configure the training process for optimal results.</p>
                    <ul>
                        <li>Learning rate: Start with defaults, adjust if needed</li>
                        <li>Batch size: Limited by available memory</li>
                        <li>Epochs: Monitor validation metrics to avoid overfitting</li>
                        <li>Data augmentation: Improves generalization</li>
                    </ul>
                </div>
                
                <div id="help-training-monitoring" style="margin-top: 25px;">
                    <h4>📈 Training Monitoring</h4>
                    <p>Track training progress and model performance.</p>
                    <ul>
                        <li>Watch loss curves for convergence</li>
                        <li>Monitor validation accuracy</li>
                        <li>Check for overfitting (validation plateaus while training improves)</li>
                        <li>Use early stopping to prevent overfitting</li>
                    </ul>
                </div>
            </div>
            
            <!-- REFINE PAGE -->
            <div id="help-refine-overview" style="margin-top: 40px; border-left: 4px solid #fd7e14; padding-left: 20px;">
                <h2>✨ Refine Page</h2>
                <p>Improve your models through iterative refinement and active learning.</p>
                
                <div id="help-active-learning" style="margin-top: 20px;">
                    <h4>🎯 Active Learning</h4>
                    <p>Efficiently improve models by labeling the most informative examples.</p>
                    <ul>
                        <li>System suggests uncertain predictions for review</li>
                        <li>Focus labeling effort where it matters most</li>
                        <li>Iteratively retrain with new labels</li>
                        <li>Achieve better performance with less labeling</li>
                    </ul>
                </div>
                
                <div id="help-error-analysis" style="margin-top: 25px;">
                    <h4>🔍 Error Analysis</h4>
                    <p>Understand and fix model mistakes.</p>
                    <ul>
                        <li>Review false positives and false negatives</li>
                        <li>Identify systematic errors and biases</li>
                        <li>Add corrective examples to training data</li>
                        <li>Adjust model parameters if needed</li>
                    </ul>
                </div>
                
                <div id="help-model-tuning" style="margin-top: 25px;">
                    <h4>🔧 Model Tuning</h4>
                    <p>Fine-tune model parameters for better performance.</p>
                    <ul>
                        <li>Adjust classification thresholds</li>
                        <li>Optimize for precision vs recall trade-offs</li>
                        <li>Test different confidence levels</li>
                        <li>Validate changes with held-out data</li>
                    </ul>
                </div>
            </div>
            
            <!-- INFERENCE PAGE -->
            <div id="help-inference-overview" style="margin-top: 40px; border-left: 4px solid #20c997; padding-left: 20px;">
                <h2>🔮 Inference Page</h2>
                <p>Apply your trained models to new audio data for automated detection.</p>
                
                <div id="help-batch-inference" style="margin-top: 20px;">
                    <h4>📦 Batch Processing</h4>
                    <p>Process large amounts of audio data efficiently.</p>
                    <ul>
                        <li>Select audio files or folders to process</li>
                        <li>Choose appropriate models for each target species</li>
                        <li>Set confidence thresholds</li>
                        <li>Monitor processing progress</li>
                    </ul>
                </div>
                
                <div id="help-result-review" style="margin-top: 25px;">
                    <h4>📋 Result Review</h4>
                    <p>Examine and validate automated detections.</p>
                    <ul>
                        <li>Browse detections sorted by confidence</li>
                        <li>Listen to detected segments</li>
                        <li>Confirm or reject automated detections</li>
                        <li>Export validated results</li>
                    </ul>
                </div>
                
                <div id="help-real-time-inference" style="margin-top: 25px;">
                    <h4>⚡ Real-time Processing</h4>
                    <p>Process audio streams as they arrive.</p>
                    <ul>
                        <li>Connect to live audio sources</li>
                        <li>Set up automated alerts</li>
                        <li>Monitor detection rates</li>
                        <li>Log all detections for later review</li>
                    </ul>
                </div>
                
                <div id="help-export-results" style="margin-top: 25px;">
                    <h4>📤 Export Results</h4>
                    <p>Share your findings with collaborators and databases.</p>
                    <ul>
                        <li>Export to CSV, JSON, or XML formats</li>
                        <li>Include confidence scores and metadata</li>
                        <li>Upload to Workbench or other databases</li>
                        <li>Generate summary reports</li>
                    </ul>
                </div>
            </div>
            
            <!-- GENERAL TIPS -->
            <div id="help-general-tips" style="margin-top: 40px; border-left: 4px solid #6c757d; padding-left: 20px;">
                <h2>💡 General Tips & Best Practices</h2>
                
                <div id="help-workflow-tips" style="margin-top: 20px;">
                    <h4>🔄 Workflow Tips</h4>
                    <ul>
                        <li>Always start with the Configuration page</li>
                        <li>Save your configuration frequently</li>
                        <li>Test with small datasets before scaling up</li>
                        <li>Keep notes on parameter settings and results</li>
                        <li>Backup your models and training data</li>
                    </ul>
                </div>
                
                <div id="help-troubleshooting" style="margin-top: 25px;">
                    <h4>🔧 Troubleshooting</h4>
                    <ul>
                        <li>Check the Status tab for system health</li>
                        <li>Review the Logs tab for error messages</li>
                        <li>Verify your API token is correct and active</li>
                        <li>Ensure you have sufficient disk space</li>
                        <li>Restart the engine if components seem unresponsive</li>
                    </ul>
                </div>
                
                <div id="help-performance-tips" style="margin-top: 25px;">
                    <h4>⚡ Performance Tips</h4>
                    <ul>
                        <li>Use SSD storage for better I/O performance</li>
                        <li>Increase batch sizes if you have more RAM</li>
                        <li>Process audio in chunks for large files</li>
                        <li>Consider GPU acceleration for training</li>
                        <li>Close unused browser tabs to free memory</li>
                    </ul>
                </div>
            </div>
            
            <div style="margin-top: 40px; padding: 20px; background-color: #e9ecef; border-radius: 8px; text-align: center;">
                <h3>📚 Need More Help?</h3>
                <p>This help system covers the main features of the Recognizer Workshop. For additional support:</p>
                <ul style="text-align: left; display: inline-block;">
                    <li>Check the project documentation</li>
                    <li>Visit the community forums</li>
                    <li>Report bugs or request features on GitHub</li>
                    <li>Contact the development team</li>
                </ul>
            </div>
        </div>
        """
        return pn.pane.HTML(help_html, sizing_mode="stretch_width")

    def _create_scroll_handler(self):
        """Create the JavaScript scroll handler that will be set up once."""
        scroll_js = """
        <script>
        // Global function to scroll to help sections
        window.scrollToHelpSection = function(sectionId) {
            console.log('Scrolling to section:', sectionId);
            setTimeout(function() {
                const element = document.getElementById(sectionId);
                if (!element) {
                    console.log('Help section not found:', sectionId);
                    return;
                }
                
                // Simply scroll the element into view
                element.scrollIntoView({ behavior: 'smooth', block: 'start' });
                
                // Add highlight effect
                element.style.transition = 'background-color 0.5s ease, border 0.5s ease';
                element.style.backgroundColor = '#fff3cd';
                element.style.border = '2px solid #ffc107';
                element.style.borderRadius = '8px';
                element.style.padding = '10px';
                
                // Remove highlight after 3 seconds
                setTimeout(function() {
                    element.style.backgroundColor = '';
                    element.style.border = '';
                    element.style.borderRadius = '';
                    element.style.padding = '';
                }, 3000);
            }, 100);
        };
        
        console.log('Help scroll handler loaded');
        </script>
        """
        return pn.pane.HTML(scroll_js, width=0, height=0)

    def show_help_section(self, section_id):
        """Show the help panel and scroll to a specific section."""
        # First, make sure help is visible
        if not bool(self.visible):
            self.visible = True
        
        # The JavaScript will handle the scrolling automatically
        # No need for complex parameter watching or DOM manipulation

    def _create_help_panel(self):
        """Create the collapsible help panel."""
        if not hasattr(self, '_help_content'):
            self._help_content = self._create_help_content()
        
        # Create the scroll handler JavaScript
        self._scroll_handler = self._create_scroll_handler()
        
        # Create the panel structure
        toggle_button = pn.widgets.Button(
            name="❓ Help",
            button_type="primary",
            width=100,
            margin=(5, 5)
        )
        toggle_button.on_click(self._toggle_help)
        
        help_content = pn.Column(
            self._scroll_handler,  # Include the JavaScript handler
            toggle_button,
            pn.Spacer(height=10),
            pn.pane.HTML("<h3>Workshop Help</h3>", margin=(0, 10)),
            self._help_content,
            margin=(5, 10),
            scroll=True,
            height=600,
            visible=pn.bind(lambda x: bool(x), self.param.visible)
        )
        
        # Collapsed view
        collapsed_view = pn.Column(
            toggle_button,
            width=110,
            visible=pn.bind(lambda x: not bool(x), self.param.visible)
        )
        
        return pn.Column(
            help_content,
            collapsed_view,
            width=400,
            margin=(5, 5)
        )

    def _toggle_help(self, event=None):
        """Toggle the help panel visibility."""
        self.visible = not bool(self.visible)
    
    def get_panel(self):
        """Get the current help panel."""
        return self._create_help_panel()
    
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


