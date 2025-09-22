import panel as pn
import asyncio
import holoviews as hv
from holoviews import opts, streams
from bokeh.palettes import Category20
# The service now handles all the backend logic
from recognizer_workshop.app.services import umap_service
import colorcet as cc
from holoviews.operation import datashader
from holoviews.operation.datashader import dynspread
import datashader as ds
# HoloViews extension is now loaded in app.py to prevent conflicts

def build(vm):
    # Create help buttons
    explore_help_btn = vm.help_system.create_help_button("help-explore-overview", "Help: Explore")
    umap_help_btn = vm.help_system.create_help_button("help-umap-visualization", "Help: UMAP Visualization")

    # --- Widgets and Placeholders ---
    btn_generate = pn.widgets.Button(name="Generate/Update UMAP Cache", button_type="primary", icon="pixels", sizing_mode="stretch_width", margin=(5, 5))
    btn_refresh = pn.widgets.Button(name="Refresh Plot", icon="refresh")
    
    status_pane = pn.pane.Markdown("...", sizing_mode="stretch_width", margin=(5, 5))
    plot_pane = pn.Column(sizing_mode="stretch_both", min_height=600, min_width=800)
    selected_embedding_pane = vm.ui.get_region_box("explore.selected_embedding")
    
    # Create a placeholder for when no embedding is selected
    embedding_placeholder = pn.pane.Markdown(
        "### 🎯 Click on an item on the plot to display the spectrogram",
        margin=(10, 5),  # Reduced margins
        sizing_mode="stretch_width"
    )
    
    # Container that will show either the placeholder or the actual embedding
    embedding_display_container = pn.Column(
        embedding_placeholder, 
        selected_embedding_pane,
        sizing_mode="stretch_width",  # Changed from stretch_both to reduce height
        max_height=400  # Limit height to prevent excessive blank space
    )
    
    # Initially show placeholder and hide embedding pane
    embedding_placeholder.visible = True
    selected_embedding_pane.visible = False
    
    # NEW: Add controls for the UMAP parameters
    fit_subset_size = pn.widgets.IntInput(name="Fit Subset Size", value=100_000, start=1000, step=1000, sizing_mode="stretch_width", margin=(5, 2))
    transform_batch_size = pn.widgets.IntInput(name="Transform Batch Size", value=100_000, start=1000, step=1000, sizing_mode="stretch_width", margin=(5, 2))

    def _update_status():
        """Checks for the cache files and updates the UI accordingly."""
        if not vm.eng:
            status_pane.object = "⚠️ Engine not initialized. Please start it on the 'Config' tab."
            btn_generate.disabled = True
            plot_pane.clear()
            return

        btn_generate.disabled = False
        db_path = vm.eng.config.db_path
        model_path, coords_path = umap_service._get_umap_paths(db_path) # Use helper from service

        if model_path.exists() and coords_path.exists():
            status_pane.object = f"✅ UMAP cache and model found."
            btn_generate.name = "Generate UMAP"
            _create_and_display_plot()
        elif model_path.exists():
             status_pane.object = f"ℹ️ UMAP model found, but coordinates are missing or incomplete. Click 'Update' to generate them."
             btn_generate.name = "Generate/Update UMAP Coordinates"
        else:
            status_pane.object = f"ℹ️ UMAP model not found. Click 'Generate' to create it. This will be slow."
            btn_generate.name = "Generate UMAP Model & Cache"
    
    async def _generate_umap_cache(_):
        if not vm.eng: return vm.notify("Engine not started.", "error")
        
        btn_generate.disabled = True
        status_pane.object = "⏳ Working on UMAP cache... This can take a long time. Please be patient."
        
        try:
            # UPDATED: Call the new main service function with the new parameters
            await asyncio.to_thread(
                umap_service.generate_umap_cache,
                db=vm.eng.db,
                db_path=vm.eng.config.db_path,
                fit_subset_size=fit_subset_size.value,
                transform_batch_size=transform_batch_size.value
            )
            vm.notify("UMAP cache processing complete!", "success")
        except Exception as e:
            vm.notify(f"UMAP generation failed: {e}", "error")
        finally:
            btn_generate.disabled = False
            # Refresh the status and plot after generation is complete
            _update_status()









    def _create_and_display_plot():
        df_plot = umap_service.load_umap_data(vm.eng.db, vm.eng.config.db_path)
        if df_plot is None or df_plot.empty:
            plot_pane.objects = [pn.pane.Alert("Could not load UMAP data.", alert_type='warning')]
            return

        
        unique_labels = sorted([l for l in df_plot['composite_label'].unique() if l != 'unlabeled'])

        # map labels to colours, where unlabelled is grey
        # we have a palette for positive and another one for negative
        color_map = {'unlabeled': 'lightgray'}
        palette_pos = iter(Category20[10])
        palette_neg = iter(Category20[20][10:])
        for label in unique_labels:
            if '(Positive)' in label: color_map[label] = next(palette_pos)
            else: color_map[label] = next(palette_neg)

        
        
        # we will plot the labelled and unlabelled points separately
        # to make sure that the labelled points are always on top and visible,
        unlabeled_df = df_plot[df_plot['composite_label'] == 'unlabeled']
        labeled_df = df_plot[df_plot['composite_label'] != 'unlabeled']
        unlabeled_points = hv.Points(unlabeled_df, kdims=['x', 'y'], vdims=['embedding_id', 'label', 'composite_label'])
        labeled_points = hv.Points(labeled_df, kdims=['x', 'y'], vdims=['embedding_id', 'label', 'composite_label'])

        # Handle the case where there might be no unlabeled or labeled points
        if unlabeled_df.empty and labeled_df.empty:
            plot_pane.objects = [pn.pane.Alert("No data points to display.", alert_type='warning')]
            return
        
        plot_elements = []
        
        # Add background layer if there are unlabeled points
        if not unlabeled_df.empty:
            # Create points for unlabeled data with small, subtle styling
            # unlabeled_styled = unlabeled_points.opts(
            #     size=2, alpha=0.3, color='lightgray',
            #     tools=['pan', 'wheel_zoom', 'reset'],
            #     width=1000, height=700  # Reduced height from 800 to 700
            # )
            # plot_elements.append(unlabeled_styled)
            
            background = datashader.datashade(unlabeled_points, aggregator=ds.count(), cmap=cc.gray)

            spread_background = dynspread(background, threshold=0.7, max_px=5)
            
            plot_elements.append(spread_background)


        # Add interactive layer if there are labeled points (for visual distinction)
        # These need to be added AFTER the background so they appear on top
        if not labeled_df.empty:
            # These will be actual vector points that show on top of the background
            labeled_styled = labeled_points.opts(
                size=12, alpha=1.0,  # Larger and fully opaque
                color='composite_label', 
                cmap=color_map,
                legend_position='top_left',  # Overlay legend on plot
                legend_cols=1,
                tools=['hover'],
                line_color='white',  # White outline for better visibility
                line_width=1,
                width=1000, height=700  # Reduced height from 800 to 700
            )
            plot_elements.append(labeled_styled)
        
        # Create a transparent overlay of ALL points for click detection
        # This ensures clicks work everywhere
        all_points_overlay = hv.Points(df_plot, kdims=['x', 'y'], vdims=['embedding_id', 'label', 'composite_label'])
        
        # Combine all plot elements
        if len(plot_elements) == 1:
            plot = plot_elements[0]
        else:
            plot = plot_elements[0]
            for element in plot_elements[1:]:
                plot = plot * element
        
        # Create a transparent overlay of ALL points for click detection with proper styling
        all_points_overlay = all_points_overlay.opts(
            alpha=0,  # Make completely transparent
            size=1,   # Minimal size
            tools=['tap'],
            width=1000, 
            height=700  # Reduced height from 800 to 700
        )
        
        # Add the transparent click-detection layer on top
        plot = plot * all_points_overlay
        
        # Create a tap stream that works on the transparent overlay
        tap_stream = streams.Tap(source=all_points_overlay, x=0, y=0)
        
        @pn.depends(tap_stream.param.x, tap_stream.param.y, watch=True)
        def _on_tap(x, y):
            if x is None or y is None or x == 0 or y == 0:
                return
            
            # Find the nearest point in the FULL dataset (both labeled and unlabeled)
            # Calculate distances to all points
            distances = ((df_plot['x'] - x) ** 2 + (df_plot['y'] - y) ** 2) ** 0.5
            nearest_idx = distances.idxmin()
            
            # Only show embedding if the click is reasonably close to a point
            min_distance = distances.min()
            if min_distance < 5.0:  # Adjust this threshold as needed
                embedding_id = df_plot.loc[nearest_idx, 'embedding_id']
                vm.ui.show_selected_embedding(vm.eng, embedding_id)
                # Hide placeholder and show embedding when something is selected
                embedding_placeholder.visible = False
                selected_embedding_pane.visible = True
            else:
                # Show placeholder if click is not near any point
                embedding_placeholder.visible = True
                selected_embedding_pane.visible = False
        
        # Wrap the plot in a Panel pane with explicit sizing
        plot_panel = pn.pane.HoloViews(plot, sizing_mode='stretch_both', max_height=700, min_width=1000)
        plot_pane.objects = [plot_panel]






    @pn.depends(vm.obs.ensure("engine_state").param.version, watch=True)
    def _on_engine_change(_):
        _update_status()

    def _refresh_plot(_):
        _create_and_display_plot()

    btn_generate.on_click(_generate_umap_cache)
    btn_refresh.on_click(_refresh_plot)
    _update_status()

    return pn.Column(
        pn.Row("## Explore Embeddings with UMAP", pn.Spacer(), explore_help_btn),
        pn.Row(
            # Left column: Controls and embedding display (1/3 width)
            pn.Column(
                pn.Card(
                    fit_subset_size,  
                    transform_batch_size,  
                    pn.FlexBox(
                        pn.Column(btn_generate, sizing_mode="fixed", width=200),
                        status_pane,
                        flex_direction="row",
                        flex_wrap="wrap",
                        align_items="center",
                        gap="10px",
                        sizing_mode="stretch_width"
                    ),
                    sizing_mode="stretch_width",
                    margin=5,  
                    max_height=320  
                ),
                pn.Row(
                    pn.Column(
                        pn.Row("Selected Embedding", pn.Spacer(), umap_help_btn),
                        embedding_display_container
                    )
                ),
                sizing_mode="stretch_width",  
                max_width=500  
            ),
            # Right column: Plot (2/3 width)
            pn.Column(
                plot_pane, 
                sizing_mode="stretch_both"
            ),
            sizing_mode="stretch_both"
        ),
        sizing_mode="stretch_both"
    )