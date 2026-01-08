
from recognizer_workshop.app.agile_engine import agile_state
import numpy as np
import pytest

@pytest.mark.timeout(60) 
def test_embed_query_process_pool(): 
    """
    Tests the new multiprocess architecture.
    This should succeed in VS Code because the heavy TF lifting 
    happens in a child process, keeping the test runner clean.
    """

    agile = agile_state(config=None)
    
    # Dummy signal: Batch size 1, 5 seconds at 32kHz
    signal = np.zeros((1, 5 * 32000), dtype=np.float32)

    try:
        # This triggers the ProcessPoolExecutor
        embedding = agile.embed_query(signal)
        print(f"Embedding shape: {embedding.shape}")
        assert embedding is not None
        assert embedding.size > 0 
        
    finally:
        # Clean up the process pool
        agile.shutdown()

