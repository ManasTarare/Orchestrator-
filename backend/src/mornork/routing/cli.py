"""One-shot CLI for testing a calibrated metadata-router configuration."""

from __future__ import annotations

import argparse
from .audit import JsonlDecisionLog
from .embeddings import SentenceTransformerEmbedder
from .policy import RoutingThresholds
from .profile_io import load_expert_profiles
from .router import MornorkRouter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score one message and print Mornork's routing decision."
    )
    parser.add_argument("--embedding-model-id", required=True)
    parser.add_argument("--experts-file", required=True)
    parser.add_argument("--thresholds-file", required=True)
    parser.add_argument("--workspace-id", required=True)
    parser.add_argument("--message-id", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mornork-version", default="0.1.0")
    parser.add_argument("--audit-log")
    args = parser.parse_args()

    thresholds = RoutingThresholds.from_file(args.thresholds_file)
    expert_profiles = load_expert_profiles(args.experts_file)
    writer = JsonlDecisionLog(args.audit_log) if args.audit_log else None
    embedder = SentenceTransformerEmbedder(
        args.embedding_model_id,
        device=args.device,
    )
    router = MornorkRouter(
        embedder,
        thresholds,
        expert_profiles=expert_profiles,
        mornork_version=args.mornork_version,
        decision_writer=writer,
    )
    decision = router.route(
        args.message,
        workspace_id=args.workspace_id,
        message_id=args.message_id,
    )
    print(json.dumps(decision.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
