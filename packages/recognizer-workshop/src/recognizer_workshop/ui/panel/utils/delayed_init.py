import panel as pn

def run_after_render(callback, delay_ms=500):
    """
    Schedules a callback to run after a user connects and the UI renders.
    
    Args:
        callback (callable): The function to run.
        delay_ms (int): Delay in ms to allow the browser to paint first.
    """
    def _on_connect_hook():
        # We must schedule the callback on the specific user's document.
        # This ensures it runs for THAT user, not the server.
        #session_context.doc.add_periodic_callback(callback, period=delay_ms, count=1)
        pn.state.add_periodic_callback(callback, period=delay_ms, count=1)
        
    # Register this hook globally. 
    # Whenever a new user connects, Panel will run _on_connect_hook.
    #pn.state.on_session_created(_on_connect_hook)
    pn.state.onload(_on_connect_hook)