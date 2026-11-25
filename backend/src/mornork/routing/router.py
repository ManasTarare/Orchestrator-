"""Metadata-driven expert matching and user-visible routing decisions."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol

from .embeddings import TextEmbedder
from .policy import RoutingThresholds, ScopeRules
from .types import (
    DomainScore,
    ExpertProfile,
    RouteAction,
    RoutingDecision,
    RoutingScore,
)


STANDARD_DOMAINS: Mapping[str, str] = {
    "Legal": "Law, regulations, contracts, agreements, legal rights, litigation, and compliance.",
    "Finance": "Taxes, investments, banking, insurance, accounting, budgets, and financial planning.",
    "Healthcare": "Medicine, symptoms, diagnosis, treatment, wellness, and health care.",
    "HR": "Human resources, employment, employee relations, hiring, compensation, and workplace policies.",
    "Customer support": "Customer questions, product help, troubleshooting, service issues, and account support.",
}


class DecisionWriter(Protocol):
    def write(self, decision: RoutingDecision) -> None: ...


def _normalize(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    if not values:
        raise ValueError("Embedder returned an empty vector")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Embedder returned a non-finite value")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return tuple(0.0 for _ in values)
    return tuple(value / norm for value in values)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedder returned vectors with inconsistent dimensions")
    if not left:
        raise ValueError("Embedder returned an empty vector")
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(left, right))))


def _profile_text(profile: ExpertProfile) -> str:
    parts = [
        f"Expert: {profile.name}",
        f"Domain: {profile.domain}",
        profile.description.strip(),
    ]
    parts.extend(hint.strip() for hint in profile.routing_hints if hint.strip())
    return ". ".join(part for part in parts if part)


class MornorkRouter:
    """Routes messages using expert metadata and a caller-selected embedder."""

    def __init__(
        self,
        embedder: TextEmbedder,
        thresholds: RoutingThresholds,
        *,
        expert_profiles: Iterable[ExpertProfile] = (),
        domains: Mapping[str, str] = STANDARD_DOMAINS,
        scope_rules: ScopeRules | None = None,
        mornork_version: str = "0.1.0",
        decision_writer: DecisionWriter | None = None,
    ) -> None:
        if not domains:
            raise ValueError("At least one standard domain is required")
        self.embedder = embedder
        self.thresholds = thresholds
        self.domains = dict(domains)
        self.scope_rules = scope_rules or ScopeRules()
        self.mornork_version = mornork_version
        self.decision_writer = decision_writer
        domain_texts = [f"Domain: {name}. {description}" for name, description in self.domains.items()]
        domain_vectors = self._encode(self.embedder, domain_texts)
        self._domain_embeddings = dict(zip(self.domains, domain_vectors))
        self._profiles: tuple[ExpertProfile, ...] = ()
        self._profile_embeddings: tuple[tuple[float, ...], ...] = ()
        self.refresh_experts(expert_profiles)

    @staticmethod
    def _encode(embedder: TextEmbedder, texts: Sequence[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []
        vectors = embedder.encode(texts)
        if len(vectors) != len(texts):
            raise ValueError("Embedder must return one vector per input")
        return [_normalize(vector) for vector in vectors]

    def refresh_experts(self, expert_profiles: Iterable[ExpertProfile]) -> None:
        """Rebuild the profile index; call after expert status or metadata changes."""
        active = tuple(
            profile
            for profile in expert_profiles
            if profile.is_ready and not profile.is_default
        )
        vectors = self._encode(self.embedder, [_profile_text(profile) for profile in active])
        self._profiles = active
        self._profile_embeddings = tuple(vectors)

    def _finish(self, decision: RoutingDecision) -> RoutingDecision:
        if self.decision_writer is not None:
            self.decision_writer.write(decision)
        return decision

    def route(
        self,
        message: str,
        *,
        workspace_id: str,
        message_id: str,
    ) -> RoutingDecision:
        cleaned = message.strip()
        if not cleaned:
            return self._finish(
                RoutingDecision(
                    workspace_id=workspace_id,
                    message_id=message_id,
                    action=RouteAction.ANSWER_SELF,
                    reason="empty_message",
                    mornork_version=self.mornork_version,
                )
            )

        scope_reason = self.scope_rules.match(cleaned)
        if scope_reason:
            return self._finish(
                RoutingDecision(
                    workspace_id=workspace_id,
                    message_id=message_id,
                    action=RouteAction.ANSWER_SELF,
                    reason=scope_reason,
                    mornork_version=self.mornork_version,
                )
            )

        query = self._encode(self.embedder, [cleaned])[0]
        expert_scores = [
            RoutingScore(
                target_id=profile.expert_id,
                name=profile.name,
                domain=profile.domain,
                version=profile.version,
                score=_cosine(query, vector),
                inference_mode=profile.inference_mode,
            )
            for profile, vector in zip(self._profiles, self._profile_embeddings)
        ]
        expert_scores.sort(key=lambda item: (-item.score, item.target_id))

        domain_scores = [
            DomainScore(
                domain=domain,
                score=_cosine(query, vector),
            )
            for domain, vector in self._domain_embeddings.items()
        ]
        domain_scores.sort(key=lambda item: (-item.score, item.domain))

        best_domain = domain_scores[0] if domain_scores else None
        top = expert_scores[0] if expert_scores else None
        reason = "below_threshold"
        action = RouteAction.ANSWER_SELF
        chosen: tuple[str, ...] = ()
        candidates: tuple[str, ...] = ()
        options: tuple[str, ...] = ()
        offer: str | None = None

        if top is not None:
            high = self.thresholds.high_for(top.domain)
            low = self.thresholds.low_for(top.domain)
            if top.score >= high:
                second = expert_scores[1] if len(expert_scores) > 1 else None
                second_low = (
                    self.thresholds.low_for(second.domain)
                    if second is not None
                    else math.inf
                )
                if (
                    second is not None
                    and second.score >= second_low
                    and top.score - second.score <= self.thresholds.ambiguity_gap
                ):
                    action = RouteAction.ASK_USER
                    reason = "multiple_experts_fit"
                    candidates = (top.target_id, second.target_id)
                    options = candidates + ("both",)
                else:
                    action = RouteAction.HANDOFF
                    reason = "clear_expert_match"
                    chosen = (top.target_id,)
            elif top.score >= low:
                action = RouteAction.ANSWER_AND_OFFER
                reason = "moderate_expert_match"
                offer = top.target_id

        if (
            action == RouteAction.ANSWER_SELF
            and best_domain is not None
            and best_domain.score >= self.thresholds.domain_recognition
        ):
            ready_for_domain = any(
                profile.domain.casefold() == best_domain.domain.casefold()
                for profile in self._profiles
            )
            if not ready_for_domain:
                action = RouteAction.DECLINE_CREATE_EXPERT
                reason = "recognized_domain_without_ready_expert"

        return self._finish(
            RoutingDecision(
                workspace_id=workspace_id,
                message_id=message_id,
                action=action,
                reason=reason,
                expert_scores=tuple(expert_scores),
                domain_scores=tuple(domain_scores),
                chosen_expert_ids=chosen,
                candidate_expert_ids=candidates,
                user_options=options,
                offer_expert_id=offer,
                mornork_version=self.mornork_version,
            )
        )
