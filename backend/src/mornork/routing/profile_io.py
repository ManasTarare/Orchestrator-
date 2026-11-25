"""Read expert profile metadata for command-line and web entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import ExpertProfile


def load_expert_profiles(path: str | Path) -> list[ExpertProfile]:
    data: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("experts")
    if not isinstance(data, list):
        raise ValueError("Expert profile file must contain a JSON list or an experts list")

    profiles = []
    for item in data:
        fields = dict(item)
        fields["routing_hints"] = tuple(fields.get("routing_hints", ()))
        profiles.append(ExpertProfile(**fields))
    return profiles
