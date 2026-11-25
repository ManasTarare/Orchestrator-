from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

from mornork.routing import (
    ExpertProfile,
    JsonlDecisionLog,
    MornorkRouter,
    RouteAction,
    RoutingThresholds,
)


class KeywordEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        vectors: list[list[float]] = []
        for text in texts:
            value = text.casefold()
            finance = any(term in value for term in ("finance", "tax", "investment"))
            legal = any(term in value for term in ("legal", "contract", "agreement", "law"))
            health = any(term in value for term in ("healthcare", "medicine", "symptom"))
            hr = any(term in value for term in ("human resources", "employment", "workplace"))
            support = any(term in value for term in ("customer support", "troubleshooting"))
            vector = [
                float(finance),
                float(legal),
                float(health),
                float(hr),
                float(support),
            ]
            vectors.append(vector)
        return vectors


def profile(
    expert_id: str,
    name: str,
    domain: str,
    description: str,
    *,
    status: str = "Ready to Deploy",
    is_default: bool = False,
) -> ExpertProfile:
    return ExpertProfile(
        expert_id=expert_id,
        name=name,
        domain=domain,
        description=description,
        status=status,
        version=3,
        is_default=is_default,
    )


def thresholds() -> RoutingThresholds:
    # Test-only values exercise policy bands; production values require calibration.
    return RoutingThresholds(
        low=0.40,
        high=0.90,
        domain_recognition=0.80,
        ambiguity_gap=0.05,
    )


class RoutingTests(unittest.TestCase):
    def test_greeting_stays_with_mornork_without_embedding_message(self) -> None:
        embedder = KeywordEmbedder()
        router = MornorkRouter(embedder, thresholds())
        calls_before = embedder.calls
        decision = router.route("  Hello! ", workspace_id="w1", message_id="m1")
        self.assertEqual(decision.action, RouteAction.ANSWER_SELF)
        self.assertEqual(decision.reason, "greeting")
        self.assertEqual(embedder.calls, calls_before)

    def test_clear_ready_expert_match_hands_off(self) -> None:
        router = MornorkRouter(
            KeywordEmbedder(),
            thresholds(),
            expert_profiles=[
                profile("fin-1", "Finance", "Finance", "Tax and investment advice")
            ],
        )
        decision = router.route("Summarize the tax implications", workspace_id="w1", message_id="m2")
        self.assertEqual(decision.action, RouteAction.HANDOFF)
        self.assertEqual(decision.chosen_expert_ids, ("fin-1",))

    def test_close_experts_prompt_user_and_offer_both(self) -> None:
        router = MornorkRouter(
            KeywordEmbedder(),
            thresholds(),
            expert_profiles=[
                profile("fin-a", "Finance A", "Finance", "Tax advice"),
                profile("fin-b", "Finance B", "Finance", "Investment tax advice"),
            ],
        )
        decision = router.route("Tax question", workspace_id="w1", message_id="m3")
        self.assertEqual(decision.action, RouteAction.ASK_USER)
        self.assertEqual(decision.user_options, ("fin-a", "fin-b", "both"))

    def test_moderate_match_offers_expert(self) -> None:
        router = MornorkRouter(
            KeywordEmbedder(),
            thresholds(),
            expert_profiles=[
                profile("fin-1", "Finance", "Finance", "Tax advice")
            ],
        )
        decision = router.route(
            "A tax and contract matter",
            workspace_id="w1",
            message_id="m4",
        )
        self.assertEqual(decision.action, RouteAction.ANSWER_AND_OFFER)
        self.assertEqual(decision.offer_expert_id, "fin-1")

    def test_known_domain_without_ready_expert_declines(self) -> None:
        router = MornorkRouter(
            KeywordEmbedder(),
            thresholds(),
            expert_profiles=[
                profile(
                    "legal-old",
                    "Legal",
                    "Legal",
                    "Contracts",
                    status="Deactivated",
                )
            ],
        )
        decision = router.route("Review this contract", workspace_id="w1", message_id="m5")
        self.assertEqual(decision.action, RouteAction.DECLINE_CREATE_EXPERT)

    def test_default_expert_is_never_a_routing_target(self) -> None:
        router = MornorkRouter(
            KeywordEmbedder(),
            thresholds(),
            expert_profiles=[
                profile(
                    "mornork",
                    "Mornork",
                    "Finance",
                    "Tax and investment",
                    is_default=True,
                )
            ],
        )
        decision = router.route("Explain the tax rules", workspace_id="w1", message_id="m6")
        self.assertEqual(decision.action, RouteAction.DECLINE_CREATE_EXPERT)
        self.assertEqual(decision.expert_scores, ())

    def test_decision_log_records_no_message_body(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "routing.jsonl"
            logger = JsonlDecisionLog(path)
            router = MornorkRouter(
                KeywordEmbedder(),
                thresholds(),
                expert_profiles=[
                    profile("fin-1", "Finance", "Finance", "Tax advice")
                ],
                decision_writer=logger,
            )
            router.route(
                "private tax details should not be logged",
                workspace_id="w1",
                message_id="m7",
            )
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["action"], "handoff")
            self.assertNotIn("private tax details", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
