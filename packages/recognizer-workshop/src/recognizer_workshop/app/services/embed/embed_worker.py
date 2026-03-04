"""
After updating tensorflow version, it seems that the embed step, specifically 
the model's infer_tf method, hangs when running in vscode testing or panel. 

THe theory is that there is some conflict between the event loop of panel and somthign similar in vscode testing environment
So, we are trying to run it in its own process, meaning we need a dedicated file for the worker process. 



"""

import os
import sys
import threading
import faulthandler
import time

faulthandler.enable()

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

# Keep it out of GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Global cache for the worker process
_WORKER_MODEL_CACHE = {}



def debug_timer(pid):
    """Prints a heartbeat so we know the process thread is alive."""
    for i in range(60):
        time.sleep(1)
        # Write directly to stderr to bypass buffering
        sys.stderr.write(f"[{pid}] Worker Heartbeat: {i}\n")
        sys.stderr.flush()


def run_embed_task(model_name: str, audio_window):
    """
    Runs in the separate process.
    Loads the model (cached) and returns the embedding as a numpy array.
    """



    pid = os.getpid()
    print(f"\n>>> [Worker PID: {pid}] STARTING TASK. <<<")
    

    import tensorflow as tf
    from perch_hoplite.zoo import model_configs


    # Start a background thread just to prove the process isn't totally dead
    t = threading.Thread(target=debug_timer, args=(pid,), daemon=True)
    t.start()

    global _WORKER_MODEL_CACHE
    
    if model_name not in _WORKER_MODEL_CACHE:
        print(f">>> [Worker] Loading model '{model_name}'...")
        # Use the exact same loader logic your app normally uses
        model = model_configs.load_model_by_name(model_name)
        _WORKER_MODEL_CACHE[model_name] = model
    
    model = _WORKER_MODEL_CACHE[model_name]
    
    # Run Inference
    print(">>> [Worker] Running embedding...")
    results = model.embed(audio_window)
    print(">>> [Worker] Embedding complete.")
    
    # Return pure numpy (detach from TF graph)
    return results.embeddings[0, 0]