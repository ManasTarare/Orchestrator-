"""Shared, explicit input formats for answer-model and router training."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any


def read_records(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(dataset["file_path"])
    if path.suffix.casefold() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            return [dict(row) for row in csv.DictReader(source)]
    if path.suffix.casefold() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig") as source:
            for line_no, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL line {line_no} must contain an object")
                rows.append(row)
        return rows
    raise ValueError("This data path requires a CSV or JSONL file")


def normalized(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key).strip().casefold(): value for key, value in row.items()}


def is_approved(row: dict[str, Any]) -> bool:
    values = normalized(row)
    return "approved" not in values or str(values["approved"]).strip().casefold() in {
        "true", "yes", "1", "approved",
    }


def row_split(row: dict[str, Any]) -> str:
    value = str(normalized(row).get("split") or "").strip().casefold()
    aliases = {"valid": "validation", "val": "validation", "dev": "validation"}
    return aliases.get(value, value)


def route_example(row: dict[str, Any]) -> dict[str, str] | None:
    if not is_approved(row):
        return None
    values = normalized(row)

    def get(*keys: str) -> str:
        for key in keys:
            value = values.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    message = get("message", "query", "text", "input", "prompt", "question")
    route = get("route", "label", "expert_id", "target")
    if not message or not route:
        return None
    return {"message": message, "route": route}


def load_route_examples(dataset: dict[str, Any], split: str | None = None) -> list[dict[str, str]]:
    rows = read_records(dataset)
    has_split = any(str(key).strip().casefold() == "split" for key in dataset.get("columns", []))
    examples: list[dict[str, str]] = []
    for row in rows:
        if split is not None and has_split and row_split(row) != split:
            continue
        example = route_example(row)
        if example:
            examples.append(example)
    return examples


def qa_example(row: dict[str, Any]) -> dict[str, str] | None:
    """Parse one reviewed prompt/response row while retaining its evidence context."""
    if not is_approved(row):
        return None
    values = normalized(row)

    def get(*keys: str) -> str:
        for key in keys:
            value = values.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    prompt = get("prompt", "question", "instruction", "input")
    response = get("response", "answer", "output", "completion")
    if not prompt or not response:
        return None
    context_parts = [get("context", "source", "reference")]
    for label, keys in (
        ("Applicable period", ("tax_period", "tax_year", "financial_year", "effective_period")),
        ("Official source section", ("source_section", "section", "paragraph")),
        ("Official source excerpt", ("official_source_excerpt", "source_excerpt", "excerpt")),
        ("Source URL", ("source_url", "url", "citation_url")),
    ):
        value = get(*keys)
        if value:
            context_parts.append(f"{label}: {value}")
    return {"prompt": prompt, "expected": response,
            "context": "\n".join(part for part in context_parts if part)}


def load_qa_examples(dataset: dict[str, Any], split: str | None = None) -> list[dict[str, str]]:
    """Load approved QA examples from CSV/JSONL, optionally selecting one split."""
    rows = read_records(dataset)
    has_split = any(str(column).strip().casefold() == "split" for column in dataset.get("columns", []))
    examples: list[dict[str, str]] = []
    for row in rows:
        if split is not None and has_split:
            actual = row_split(row)
            if split == "evaluation":
                if actual not in {"validation", "test"}:
                    continue
            elif actual != split:
                continue
        example = qa_example(row)
        if example:
            examples.append(example)
    return examples


def deterministic_route_splits(
    examples: list[dict[str, str]], seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Create deterministic 80/10/10 partitions with at least one held-out row each."""
    if len(examples) < 3:
        raise ValueError("Orchestrator data needs at least 3 examples to create train, validation, and test splits")
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    train_end = max(1, int(len(indices) * 0.8))
    validation_count = max(1, int(len(indices) * 0.1))
    if train_end + validation_count >= len(indices):
        train_end = len(indices) - 2
        validation_count = 1
    train = [examples[i] for i in indices[:train_end]]
    validation = [examples[i] for i in indices[train_end:train_end + validation_count]]
    test = [examples[i] for i in indices[train_end + validation_count:]]
    return train, validation, test

