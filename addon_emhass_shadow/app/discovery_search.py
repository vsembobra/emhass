"""One explicit discovery job, independent of the planner and HTTP requests."""

from copy import deepcopy
from threading import Lock, Thread


class DiscoverySearch:
    def __init__(self, fetch):
        self.fetch = fetch
        self.lock = Lock()
        self.thread = None
        self.state = {"running": False, "result": None, "error": None}

    def status(self) -> dict:
        with self.lock:
            return deepcopy(self.state)

    def start(self) -> bool:
        with self.lock:
            if self.state["running"]:
                return False
            self.state = {"running": True, "result": None, "error": None}
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()
        return True

    def _run(self):
        try:
            result = self.fetch()
            with self.lock:
                self.state["result"] = result
        except Exception as error:
            with self.lock:
                self.state["error"] = str(error) or type(error).__name__
        finally:
            with self.lock:
                self.state["running"] = False
