"""Typed routing profiles, scores, and decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class RouteAction(str, Enum):
    ANSWER_SELF = "answer"
    HANDOFF = "handoff"
    ASK_USER = "ask"
    ANSWER_AND_OFFER = "offer"
    DECLINE_CREATE_EXPERT = "decline"


@dataclass(frozen=True)
class ExpertProfile:
    expert_id: str
    name: str
    domain: str
    description: str
    status: str
    version: str | int | None = None
    routing_hints: tuple[str, ...] = ()
    is_default: bool = False
    inference_url: str | None = None
    workspace_id: str | None = None
    model_path: str | None = None
    backend_url: str | None = None
    system_prompt: str | None = None
    inference_mode: str = "checkpoint"

    @property
    def is_ready(self) -> bool:
        status = getattr(self.status, "value", self.status)
        return str(status).casefold().replace("_", " ") == "ready to deploy"


@dataclass(frozen=True)
class RoutingScore:
    target_id: str
    name: str
    domain: str
    score: float
    version: str | int | None = None
    inference_mode: str = "checkpoint"


@dataclass(frozen=True)
class DomainScore:
    domain: str
    score: float


@dataclass(frozen=True)
class RoutingDecision:
    workspace_id: str
    message_id: str
    action: RouteAction
    reason: str
    expert_scores: tuple[RoutingScore, ...] = ()
    domain_scores: tuple[DomainScore, ...] = ()
    chosen_expert_ids: tuple[str, ...] = ()
    candidate_expert_ids: tuple[str, ...] = ()
    user_options: tuple[str, ...] = ()
    offer_expert_id: str | None = None
    mornork_version: str = "0.1.0"
    user_choice: str | None = None
    decision_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    score_type: str = "cosine_similarity"
    orchestrator_model_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["action"] = self.action.value
        return data
