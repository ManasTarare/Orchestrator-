"""Append-only routing decision log that never stores message text."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from .types import RoutingDecision


class JsonlDecisionLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, decision: RoutingDecision) -> None:
        line = json.dumps(decision.to_dict(), ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()
