"""Routing thresholds and low-risk scope rules."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class RoutingThresholds:
    """Thresholds calibrated on a labelled set; no production defaults are guessed."""

    low: float
    high: float
    domain_recognition: float
    ambiguity_gap: float
    low_by_domain: Mapping[str, float] = field(default_factory=dict)
    high_by_domain: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label, value in (
            ("low", self.low),
            ("high", self.high),
            ("domain_recognition", self.domain_recognition),
        ):
            if not isinstance(value, (int, float)) or not -1.0 <= value <= 1.0:
                raise ValueError(f"{label} must be a calibrated cosine score from -1 to 1")
        if self.high <= self.low:
            raise ValueError("high must be greater than low")
        if not isinstance(self.ambiguity_gap, (int, float)) or not 0.0 <= self.ambiguity_gap <= 2.0:
            raise ValueError("ambiguity_gap must be between 0 and 2")

        domains = set(self.low_by_domain) | set(self.high_by_domain)
        for domain in domains:
            low = self._lookup(self.low_by_domain, domain, self.low)
            high = self._lookup(self.high_by_domain, domain, self.high)
            if not -1.0 <= low <= 1.0 or not -1.0 <= high <= 1.0:
                raise ValueError(f"threshold for {domain} must be from -1 to 1")
            if high <= low:
                raise ValueError(f"high threshold for {domain} must exceed low")

    @staticmethod
    def _lookup(values: Mapping[str, float], domain: str, fallback: float) -> float:
        folded = domain.casefold()
        for key, value in values.items():
            if key.casefold() == folded:
                return value
        return fallback

    def low_for(self, domain: str) -> float:
        return self._lookup(self.low_by_domain, domain, self.low)

    def high_for(self, domain: str) -> float:
        return self._lookup(self.high_by_domain, domain, self.high)

    @classmethod
    def from_file(cls, path: str | Path) -> "RoutingThresholds":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if any(values.get(key) is None for key in ("low", "high", "domain_recognition", "ambiguity_gap")):
            raise ValueError("Fill threshold values from calibration before loading this config")
        for key in ("low_by_domain", "high_by_domain"):
            if any(value is None for value in values.get(key, {}).values()):
                raise ValueError("Fill per-domain thresholds from calibration before loading this config")
        return cls(**values)


class ScopeRules:
    """Keep greetings and simple conversational openers with Mornork."""

    _patterns = (
        (re.compile(r"^(hi|hello|hey|hiya)[.! ]*$", re.I), "greeting"),
        (re.compile(r"^(good morning|good afternoon|good evening)[.! ]*$", re.I), "greeting"),
        (re.compile(r"^(thanks|thank you|thx|much appreciated)[.! ]*$", re.I), "thanks"),
        (re.compile(r"^(what can you do|who are you|how can you help)[?.! ]*$", re.I), "capability_question"),
        (re.compile(r"^(how are you|how is it going)[?.! ]*$", re.I), "small_talk"),
    )

    def match(self, message: str) -> str | None:
        normalized = " ".join(message.strip().split())
        for pattern, reason in self._patterns:
            if pattern.fullmatch(normalized):
                return reason
        return None
