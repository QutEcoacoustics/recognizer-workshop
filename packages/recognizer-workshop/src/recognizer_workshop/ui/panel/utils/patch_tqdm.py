# We want to show progress in our Panel app
# but it's looping through somewhere else with a tqdm progress bar.
# So, we monkey patch tqdm to update a panel progress bar instead.


import tqdm
from contextlib import contextmanager
import panel as pn
import threading

# Create a thread-local object to store data for each thread.
# Some processes run asynchronously in separate threads
thread_local_data = threading.local()

# Create a lock to protect the patching process.
patch_lock = threading.Lock()

class TqdmCallback(tqdm.tqdm):
    """A custom tqdm class that updates a Panel progress bar via thread-local storage."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 3. Each instance gets its own progress_bar from its thread's local storage.
        self.progress_bar = getattr(thread_local_data, 'progress_bar', None)

    def update(self, n=1):
        super().update(n)
        if self.progress_bar and self.total:
            self.progress_bar.value = int(self.n / self.total * 100)

@contextmanager
def patch_tqdm(progress_bar: pn.widgets.Progress):
    """A thread-safe context manager to temporarily replace tqdm.tqdm."""
    with patch_lock: # 4. Only one thread can apply/remove the patch at a time.
        original_tqdm = tqdm.tqdm
        # 5. Store the progress bar in the current thread's local storage.
        thread_local_data.progress_bar = progress_bar
        tqdm.tqdm = TqdmCallback
    try:
        yield
    finally:
        with patch_lock: # 6. Ensure cleanup is also synchronized.
            tqdm.tqdm = original_tqdm
            # Clean up the thread-local attribute
            if hasattr(thread_local_data, 'progress_bar'):
                del thread_local_data.progress_bar