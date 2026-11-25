"""Fit routing threshold proposals from labeled JSONL examples."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .embeddings import SentenceTransformerEmbedder, TextEmbedder
from .policy import RoutingThresholds
from .profile_io import load_expert_profiles
from .router import MornorkRouter
from .types import ExpertProfile, RoutingDecision

EXPECTED_ACTIONS = {"answer", "offer", "handoff", "ask", "decline"}


@dataclass(frozen=True)
class LabeledEvidence:
    message: str
    expected_action: str
    expected_expert_id: str | None
    expected_expert_ids: frozenset[str]
    decision: RoutingDecision
    scope_matched: bool

    @property
    def top_score(self) -> float:
        return self.decision.expert_scores[0].score if self.decision.expert_scores else -1.0

    @property
    def domain_score(self) -> float:
        return self.decision.domain_scores[0].score if self.decision.domain_scores else -1.0

    @property
    def score_gap(self) -> float:
        scores = self.decision.expert_scores
        return scores[0].score - scores[1].score if len(scores) > 1 else 2.0


def read_labeled_examples(path: str | Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on line {line_number}: {error.msg}") from error
        if not isinstance(item, dict):
            raise ValueError(f"Calibration line {line_number} must be a JSON object")
        message = item.get("message")
        action = item.get("expected_action")
        if not isinstance(message, str) or not message.strip():
            raise ValueError(f"Calibration line {line_number} needs a non-empty message")
        if not isinstance(action, str) or action not in EXPECTED_ACTIONS:
            raise ValueError(
                f"Calibration line {line_number} expected_action must be one of "
                + ", ".join(sorted(EXPECTED_ACTIONS))
            )
        expected_id = item.get("expected_expert_id")
        if action in {"offer", "handoff"} and (
            not isinstance(expected_id, str) or not expected_id.strip()
        ):
            raise ValueError(f"Calibration line {line_number} needs expected_expert_id for {action}")
        expected_ids = item.get("expected_expert_ids", [])
        if action == "ask" and (
            not isinstance(expected_ids, list)
            or not all(isinstance(value, str) and value.strip() for value in expected_ids)
            or len(set(expected_ids)) < 2
        ):
            raise ValueError(f"Calibration line {line_number} needs at least two expected_expert_ids for ask")
        examples.append(item)
    if not examples:
        raise ValueError("Calibration file contains no labeled examples")
    return examples


def _fit_binary_cutoff(
    values: Sequence[float],
    positive: Sequence[bool],
    *,
    positive_when_greater: bool,
    minimum: float,
    maximum: float,
    exclusive_lower_bound: float | None = None,
) -> float:
    if len(values) != len(positive) or not values:
        raise ValueError("Not enough labeled examples to fit this threshold")
    if not any(positive) or all(positive):
        raise ValueError("Each threshold needs both positive and negative labeled examples")

    ordered = sorted(set(max(minimum, min(maximum, value)) for value in values))
    candidates = {minimum, maximum, *ordered}
    candidates.update((left + right) / 2 for left, right in zip(ordered, ordered[1:]))
    candidates = {
        candidate
        for candidate in candidates
        if exclusive_lower_bound is None or candidate > exclusive_lower_bound
    }
    if not candidates:
        raise ValueError("Could not find a threshold consistent with low < high; add clearer examples")

    def score(cutoff: float) -> tuple[float, float]:
        predictions = [
            value >= cutoff if positive_when_greater else value <= cutoff
            for value in values
        ]
        true_positive = sum(predicted and actual for predicted, actual in zip(predictions, positive))
        false_positive = sum(predicted and not actual for predicted, actual in zip(predictions, positive))
        false_negative = sum(not predicted and actual for predicted, actual in zip(predictions, positive))
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2 * true_positive / denominator if denominator else 0.0
        conservative_tie_break = cutoff if positive_when_greater else -cutoff
        return f1, conservative_tie_break

    return max(candidates, key=score)


def fit_thresholds(rows: Sequence[LabeledEvidence], profiles: Sequence[ExpertProfile]) -> RoutingThresholds:
    selectable = [row for row in rows if not row.scope_matched]
    low_rows = [row for row in selectable if row.expected_action in {"answer", "decline", "offer", "handoff", "ask"}]
    low = _fit_binary_cutoff(
        [row.top_score for row in low_rows],
        [row.expected_action in {"offer", "handoff", "ask"} for row in low_rows],
        positive_when_greater=True,
        minimum=-1.0,
        maximum=1.0,
    )

    high_rows = [row for row in selectable if row.expected_action in {"offer", "handoff", "ask"}]
    high = _fit_binary_cutoff(
        [row.top_score for row in high_rows],
        [row.expected_action in {"handoff", "ask"} for row in high_rows],
        positive_when_greater=True,
        minimum=-1.0,
        maximum=1.0,
        exclusive_lower_bound=low,
    )

    domain_rows = [row for row in selectable if row.expected_action in {"answer", "decline"}]
    domain_recognition = _fit_binary_cutoff(
        [row.domain_score for row in domain_rows],
        [row.expected_action == "decline" for row in domain_rows],
        positive_when_greater=True,
        minimum=-1.0,
        maximum=1.0,
    )

    ambiguity_rows = [row for row in selectable if row.expected_action in {"ask", "handoff"}]
    for row in ambiguity_rows:
        if row.expected_action == "ask" and len(row.decision.expert_scores) < 2:
            raise ValueError("Ask examples must have at least two ready experts to compare")
    ambiguity_gap = _fit_binary_cutoff(
        [row.score_gap for row in ambiguity_rows],
        [row.expected_action == "ask" for row in ambiguity_rows],
        positive_when_greater=False,
        minimum=0.0,
        maximum=2.0,
    )

    return RoutingThresholds(
        low=low,
        high=high,
        domain_recognition=domain_recognition,
        ambiguity_gap=ambiguity_gap,
    )


def _predict(row: LabeledEvidence, thresholds: RoutingThresholds, ready_domains: set[str]) -> tuple[str, str | None, frozenset[str]]:
    if row.scope_matched:
        return "answer", None, frozenset()

    scores = row.decision.expert_scores
    top = scores[0] if scores else None
    action = "answer"
    selected: str | None = None
    choices: frozenset[str] = frozenset()
    if top is not None and top.score >= thresholds.high_for(top.domain):
        second = scores[1] if len(scores) > 1 else None
        if (
            second is not None
            and second.score >= thresholds.low_for(second.domain)
            and top.score - second.score <= thresholds.ambiguity_gap
        ):
            action = "ask"
            choices = frozenset((top.target_id, second.target_id))
        else:
            action = "handoff"
            selected = top.target_id
    elif top is not None and top.score >= thresholds.low_for(top.domain):
        action = "offer"
        selected = top.target_id

    if action == "answer" and row.decision.domain_scores:
        best_domain = row.decision.domain_scores[0]
        if (
            best_domain.score >= thresholds.domain_recognition
            and best_domain.domain.casefold() not in ready_domains
        ):
            action = "decline"
    return action, selected, choices


def evaluate_thresholds(rows: Sequence[LabeledEvidence], thresholds: RoutingThresholds, profiles: Sequence[ExpertProfile]) -> dict[str, Any]:
    ready_domains = {
        profile.domain.casefold()
        for profile in profiles
        if profile.is_ready and not profile.is_default
    }
    per_action: dict[str, dict[str, int]] = {}
    correct = 0
    for row in rows:
        predicted, selected, choices = _predict(row, thresholds, ready_domains)
        hit = predicted == row.expected_action
        if hit and predicted in {"offer", "handoff"} and row.expected_expert_id:
            hit = selected == row.expected_expert_id
        if hit and predicted == "ask" and row.expected_expert_ids:
            hit = choices == row.expected_expert_ids
        counts = per_action.setdefault(row.expected_action, {"total": 0, "correct": 0})
        counts["total"] += 1
        if hit:
            counts["correct"] += 1
            correct += 1
    return {
        "example_count": len(rows),
        "correct_count": correct,
        "in_sample_accuracy": correct / len(rows) if rows else 0.0,
        "by_expected_action": per_action,
    }


def _collect_evidence(
    examples: Sequence[dict[str, Any]],
    profiles: Sequence[ExpertProfile],
    embedder: TextEmbedder,
) -> list[LabeledEvidence]:
    probe = MornorkRouter(
        embedder,
        RoutingThresholds(low=-1.0, high=1.0, domain_recognition=1.0, ambiguity_gap=0.0),
        expert_profiles=profiles,
    )
    evidence: list[LabeledEvidence] = []
    for index, example in enumerate(examples, 1):
        decision = probe.route(
            example["message"],
            workspace_id="calibration",
            message_id=f"example-{index}",
        )
        evidence.append(
            LabeledEvidence(
                message=example["message"],
                expected_action=example["expected_action"],
                expected_expert_id=example.get("expected_expert_id"),
                expected_expert_ids=frozenset(example.get("expected_expert_ids", ())),
                decision=decision,
                scope_matched=decision.reason in {"greeting", "thanks", "capability_question", "small_talk"},
            )
        )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit routing-threshold proposals from labeled examples."
    )
    parser.add_argument("--embedding-model-id", required=True)
    parser.add_argument("--experts-file", required=True)
    parser.add_argument("--examples-file", required=True)
    parser.add_argument("--validation-file", help="Optional separate labeled set for held-out evaluation")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    examples = read_labeled_examples(args.examples_file)
    profiles = load_expert_profiles(args.experts_file)
    profile_ids = {profile.expert_id for profile in profiles if profile.is_ready and not profile.is_default}
    validation_examples = read_labeled_examples(args.validation_file) if args.validation_file else []
    for dataset_name, dataset in (("Calibration", examples), ("Validation", validation_examples)):
        for index, example in enumerate(dataset, 1):
            targets = {example.get("expected_expert_id")}
            targets.update(example.get("expected_expert_ids", ()))
            missing = {target for target in targets if target and target not in profile_ids}
            if missing:
                raise ValueError(
                    f"{dataset_name} example {index} references expert IDs that are not Ready to Deploy: "
                    + ", ".join(sorted(missing))
                )
    embedder = SentenceTransformerEmbedder(args.embedding_model_id, device=args.device)
    rows = _collect_evidence(examples, profiles, embedder)
    thresholds = fit_thresholds(rows, profiles)
    output = {
        "low": thresholds.low,
        "high": thresholds.high,
        "domain_recognition": thresholds.domain_recognition,
        "ambiguity_gap": thresholds.ambiguity_gap,
        "low_by_domain": {},
        "high_by_domain": {},
    }
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    report = evaluate_thresholds(rows, thresholds, profiles)
    report["thresholds"] = output
    report["note"] = "In-sample fit only. Review on a separate labeled validation set before use."
    if validation_examples:
        validation_rows = _collect_evidence(validation_examples, profiles, embedder)
        report["validation"] = evaluate_thresholds(validation_rows, thresholds, profiles)
    report_path = output_path.with_name(output_path.stem + ".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Threshold proposal written to {output_path}")
    print(f"Calibration report written to {report_path}")
    print(f"In-sample action/expert accuracy: {report['in_sample_accuracy']:.1%} ({report['correct_count']}/{report['example_count']})")
    if "validation" in report:
        validation = report["validation"]
        print(
            "Held-out action/expert accuracy: "
            f"{validation['in_sample_accuracy']:.1%} "
            f"({validation['correct_count']}/{validation['example_count']})"
        )
    print("This fit is not production calibration; validate with held-out labeled examples.")


if __name__ == "__main__":
    main()
