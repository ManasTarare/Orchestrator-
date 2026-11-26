"""Small, configurable high-confidence routing rules for common request intents."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Pattern

from .router import STANDARD_DOMAINS


@dataclass(frozen=True)
class IntentRule:
    rule_id: str
    domain: str
    priority: int
    patterns: tuple[Pattern[str], ...]


def load_intent_rules(path: str | Path) -> tuple[IntentRule, ...]:
    """Load routing intent rules; a missing optional file disables this layer."""
    rule_path = Path(path)
    if not rule_path.is_file():
        return ()
    data = json.loads(rule_path.read_text(encoding="utf-8"))
    rows = data.get("rules") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise ValueError("Routing intent rules must contain a rules list")

    domains = {name.casefold(): name for name in STANDARD_DOMAINS}
    rules: list[IntentRule] = []
    identifiers: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each routing intent rule must be an object")
        rule_id = str(row.get("id") or "").strip()
        domain = domains.get(str(row.get("domain") or "").strip().casefold())
        priority = row.get("priority")
        patterns = row.get("patterns")
        if not rule_id or rule_id in identifiers or domain is None:
            raise ValueError("Routing intent rules need unique IDs and supported domains")
        if not isinstance(priority, int) or not isinstance(patterns, list) or not patterns:
            raise ValueError(f"Routing intent rule {rule_id} needs an integer priority and patterns")
        try:
            compiled = tuple(re.compile(str(pattern), re.IGNORECASE) for pattern in patterns)
        except re.error as error:
            raise ValueError(f"Routing intent rule {rule_id} has an invalid pattern") from error
        identifiers.add(rule_id)
        rules.append(IntentRule(rule_id, domain, priority, compiled))
    return tuple(sorted(rules, key=lambda rule: (-rule.priority, rule.rule_id)))


def match_configured_intent(message: str, rules: tuple[IntentRule, ...]) -> IntentRule | None:
    """Return the highest-priority unambiguous domain match, if one exists."""
    normalized = " ".join(message.split())
    matches = [
        rule
        for rule in rules
        if any(pattern.search(normalized) for pattern in rule.patterns)
    ]
    if not matches:
        return None
    highest = max(rule.priority for rule in matches)
    winners = [rule for rule in matches if rule.priority == highest]
    if len({rule.domain.casefold() for rule in winners}) != 1:
        return None
    return winners[0]
