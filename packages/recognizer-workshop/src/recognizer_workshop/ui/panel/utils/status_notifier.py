import asyncio
from collections import deque
import panel as pn

_CSS_INJECTED = False

def _ensure_css():
    global _CSS_INJECTED
    if _CSS_INJECTED:
        return
    pn.config.raw_css.append("""
    @keyframes rw-status-fadeout {
        from { opacity: 1; }
        to   { opacity: 0; }
    }
    .rw-status-fading { animation: rw-status-fadeout 1s ease-out forwards; }
    """)
    _CSS_INJECTED = True


class StatusNotifier:
    MIN_DISPLAY   = 1.0   # seconds each message is shown at minimum
    MAX_DISPLAY   = 5.0   # seconds total before bar clears
    FADE_DURATION = 1.0   # seconds for fade-out animation
    POLL_INTERVAL = 0.05  # seconds between queue checks

    def __init__(self):
        _ensure_css()
        self.widget = pn.pane.Markdown("", sizing_mode="stretch_width")
        self._queue: deque[tuple[str, str]] = deque()
        self._draining = False

    def notify(self, msg: str, icon: str) -> None:
        self._queue.append((icon, msg))
        if self._draining:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.widget.object = f"{icon} {msg}"
            return
        self._draining = True
        loop.create_task(self._drain())

    async def _drain(self) -> None:
        try:
            while True:
                # Show all queued messages, each held for MIN_DISPLAY seconds.
                while self._queue:
                    icon, msg = self._queue.popleft()
                    self.widget.css_classes = []
                    self.widget.object = f"{icon} {msg}"
                    await asyncio.sleep(self.MIN_DISPLAY)

                # Poll until it's time to start the fade, restarting if new items arrive.
                wait_before_fade = self.MAX_DISPLAY - self.MIN_DISPLAY - self.FADE_DURATION
                elapsed = 0.0
                interrupted = False
                while elapsed < wait_before_fade:
                    await asyncio.sleep(self.POLL_INTERVAL)
                    elapsed += self.POLL_INTERVAL
                    if self._queue:
                        interrupted = True
                        break
                if interrupted:
                    continue

                # Start fade-out, still interruptible by new messages.
                self.widget.css_classes = ["rw-status-fading"]
                fade_elapsed = 0.0
                interrupted = False
                while fade_elapsed < self.FADE_DURATION:
                    await asyncio.sleep(self.POLL_INTERVAL)
                    fade_elapsed += self.POLL_INTERVAL
                    if self._queue:
                        interrupted = True
                        break
                if interrupted:
                    self.widget.css_classes = []
                    continue

                # Fade complete — clear and exit drain.
                self.widget.object = ""
                self.widget.css_classes = []
                break
        finally:
            self._draining = False
