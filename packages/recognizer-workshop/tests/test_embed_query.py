
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
    # 1. Initialize with None config (safe as long as we don't call init_db)
    agile = agile_state(config=None)
    
    # 2. Create dummy signal: Batch size 1, 5 seconds at 32kHz
    signal = np.zeros((1, 5 * 32000), dtype=np.float32)

    try:
        # 3. Call the wrapper method (blocks until worker returns)
        # This triggers the ProcessPoolExecutor
        embedding = agile.embed_query(signal)

        print(f"Embedding shape: {embedding.shape}")

        # 4. Assertions
        assert embedding is not None
        # Check that we got data back (shape depends on model, usually (1, 128) or similar)
        assert embedding.size > 0 
        
    finally:
        # 5. Important: Clean up the process pool
        agile.shutdown()



@pytest.mark.timeout(25)  # Requires pytest-timeout: pip install pytest-timeout
@pytest.mark.xfail(reason="Embedding hangs - known issue with tensorflow dependency of perch-hoplite")
def test_embed_query(): 

    agile = agile_state(config=None)
    model = agile.embedding_model
    signal = np.zeros((1,5 * 32000), dtype=np.float32)
    outputs= model.embed(signal)

    print(outputs)

    assert outputs.embeddings.shape[0] > 0

@pytest.mark.timeout(25)  # Requires pytest-timeout: pip install pytest-timeout
@pytest.mark.xfail(reason="Embedding hangs - known issue with tensorflow dependency of perch-hoplite")
def test_embed_direct():
    """
    Some versions of tensorflow don't seem to work on 
    certain platforms (e.g., M1 macs). This is the minimum test for 
    running the embedding model directly to test that it's working
    """

    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # os.environ["TF_METAL_DEVICE_NS"] = "0"
    import kagglehub
    import tensorflow as tf
    import numpy as np


    model_path = "google/bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier/8"
    cached_model_path = kagglehub.model_download(model_path)
    model = tf.saved_model.load(cached_model_path)
    
    assert model is not None
    signal = np.zeros((1,5 * 32000), dtype=np.float32)
    outputs = model.infer_tf(signal)

    print(outputs)

    assert outputs is not None

