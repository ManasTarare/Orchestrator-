from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mornork.routing.calibrate import (
    LabeledEvidence,
    _fit_binary_cutoff,
    evaluate_thresholds,
    fit_thresholds,
    read_labeled_examples,
)
from mornork.routing.types import (
    DomainScore,
    ExpertProfile,
    RouteAction,
    RoutingDecision,
    RoutingScore,
)


def evidence(
    action: str,
    top_score: float,
    domain: str,
    domain_score: float,
    *,
    expert_id: str = "tax",
    second_score: float | None = None,
    expected_id: str | None = None,
    expected_ids: frozenset[str] = frozenset(),
) -> LabeledEvidence:
    scores = [RoutingScore(expert_id, "Finance Tax", "Finance", top_score, 1)]
    if second_score is not None:
        scores.append(RoutingScore("invest", "Finance Investing", "Finance", second_score, 1))
    decision = RoutingDecision(
        workspace_id="cal",
        message_id=action,
        action=RouteAction.ANSWER_SELF,
        reason="below_threshold",
        expert_scores=tuple(scores),
        domain_scores=(DomainScore(domain, domain_score),),
    )
    return LabeledEvidence(
        message="synthetic example",
        expected_action=action,
        expected_expert_id=expected_id,
        expected_expert_ids=expected_ids,
        decision=decision,
        scope_matched=False,
    )


class CalibrationTests(unittest.TestCase):
    def test_cutoff_separates_labeled_scores(self) -> None:
        cutoff = _fit_binary_cutoff(
            [0.1, 0.2, 0.8, 0.9],
            [False, False, True, True],
            positive_when_greater=True,
            minimum=-1.0,
            maximum=1.0,
        )
        self.assertGreater(cutoff, 0.2)
        self.assertLessEqual(cutoff, 0.8)

    def test_fit_thresholds_reproduces_synthetic_policy_cases(self) -> None:
        profiles = [
            ExpertProfile("tax", "Finance Tax", "Finance", "Tax", "Ready to Deploy", 1),
            ExpertProfile("invest", "Finance Investing", "Finance", "Investing", "Ready to Deploy", 1),
        ]
        rows = [
            evidence("answer", 0.1, "Customer support", 0.1),
            evidence("offer", 0.5, "Finance", 0.4, expected_id="tax"),
            evidence("handoff", 0.9, "Finance", 0.5, expected_id="tax"),
            evidence("ask", 0.92, "Finance", 0.55, second_score=0.92, expected_ids=frozenset({"tax", "invest"})),
            evidence("decline", 0.1, "Legal", 0.8),
        ]
        thresholds = fit_thresholds(rows, profiles)
        report = evaluate_thresholds(rows, thresholds, profiles)
        self.assertEqual(report["correct_count"], 5)
        self.assertEqual(report["in_sample_accuracy"], 1.0)
        self.assertGreater(thresholds.high, thresholds.low)

    def test_read_labeled_examples_validates_action_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "examples.jsonl"
            path.write_text('{"message":"hello","expected_action":"not-an-action"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected_action"):
                read_labeled_examples(path)


if __name__ == "__main__":
    unittest.main()
