"""A lightweight guard for the single-process public demo, not a billing cap."""
from collections import deque
from threading import Lock
from time import monotonic


class DemoLimit:
    def __init__(self, daily_runs=20, active_runs=2, *, clock=monotonic):
        self.daily_runs = daily_runs
        self.active_runs = active_runs
        self._starts = deque()
        self._active = 0
        self._clock = clock
        self._lock = Lock()

    def enter(self):
        with self._lock:
            now = self._clock()
            while self._starts and self._starts[0] <= now - 86400:
                self._starts.popleft()
            if self._active >= self.active_runs:
                return "The demo is already running its maximum concurrent reviews. Try again in a moment."
            if len(self._starts) >= self.daily_runs:
                return "The demo has reached its rolling review limit. Try again later."
            self._starts.append(now)
            self._active += 1
        return ""

    def leave(self):
        with self._lock:
            self._active -= 1
