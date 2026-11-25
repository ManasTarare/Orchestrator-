"""FastAPI orchestrator and local Experts module API."""

from __future__ import annotations

import argparse
import asyncio
import csv
from collections import Counter
import json
import os
import re
import shutil
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import torch
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request as FastAPIRequest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool

from .experts.registry import ExpertRegistry
from .modeling.generation import LocalChatModel
from .modeling.routing import LocalRouteModel
from .paths import project_root
from .routing.audit import JsonlDecisionLog
from .routing.embeddings import SentenceTransformerEmbedder
from .routing.policy import RoutingThresholds
from .routing.router import MornorkRouter, STANDARD_DOMAINS
from .routing.types import ExpertProfile, RouteAction, RoutingDecision, RoutingScore
from .training.manager import TrainingManager
from .training.artifacts import ModelArtifactStore
from .training.data import deterministic_route_splits, is_approved, load_route_examples, read_records, row_split
from .training.metrics import calculate_answer_metrics
from .training.store import TrainingStore
from .workflows import WorkflowStore

PROJECT_ROOT = project_root()
MAX_MESSAGE_CHARS = 10000
APP_VERSION = "0.1.0"


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def project_path(value: str, default: str) -> Path:
    path = Path(os.environ.get(value, default))
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _has_saved_inrun_evaluation(evaluation: Any) -> bool:
    """Check saved held-out results only; never launch a new evaluation."""
    if not isinstance(evaluation, dict):
        return False
    task = evaluation.get("task_metrics")
    if isinstance(task, dict):
        split = str(task.get("split") or "").casefold()
        try:
            count = int(task.get("examples_evaluated") or 0)
        except (TypeError, ValueError):
            count = 0
        if split in {"test", "validation"} and count > 0:
            return True
    split = str(evaluation.get("test_split") or "").casefold()
    try:
        count = int(evaluation.get("test_examples_evaluated") or 0)
    except (TypeError, ValueError):
        count = 0
    return split == "test" and count > 0

@dataclass(frozen=True)
class Settings:
    embedding_model_id: str
    experts_file: Path
    thresholds_file: Path
    audit_log: Path
    expert_audit_log: Path
    checkpoint: Path
    expert_model_root: Path
    embedder_device: str
    generation_device: str
    generation_dtype: str
    routing_mode: str
    expert_device: str
    expert_dtype: str
    max_new_tokens: int
    expert_max_new_tokens: int
    expert_timeout_seconds: float
    expert_inference_url: str
    expert_api_key: str | None
    expert_backend_api_key: str | None
    training_database: Path
    training_dataset_root: Path
    training_artifact_root: Path
    training_log_root: Path
    max_dataset_upload_mb: int
    host: str
    port: int
    no_browser: bool

    @classmethod
    def from_environment(cls) -> "Settings":
        load_dotenv()
        return cls(
            embedding_model_id=os.environ.get("MORNORK_EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2"),
            experts_file=project_path("MORNORK_EXPERTS_FILE", "config/experts.json"),
            thresholds_file=project_path("MORNORK_THRESHOLDS_FILE", "config/routing.thresholds.json"),
            audit_log=project_path("MORNORK_AUDIT_LOG", "logs/routing.jsonl"),
            expert_audit_log=project_path("MORNORK_EXPERT_AUDIT_LOG", "logs/experts.audit.jsonl"),
            checkpoint=project_path("MORNORK_CHECKPOINT", "artifacts/mornork-0.6b"),
            expert_model_root=project_path("MORNORK_EXPERT_MODEL_ROOT", "artifacts/experts"),
            embedder_device=os.environ.get("MORNORK_EMBEDDER_DEVICE", "cpu"),
            generation_device=os.environ.get("MORNORK_GENERATION_DEVICE", "auto"),
            generation_dtype=os.environ.get("MORNORK_GENERATION_DTYPE", "bfloat16"),
            routing_mode=os.environ.get("MORNORK_ROUTING_MODE", "production").strip().casefold(),
            expert_device=os.environ.get("MORNORK_EXPERT_DEVICE", "cpu"),
            expert_dtype=os.environ.get("MORNORK_EXPERT_DTYPE", "bfloat16"),
            max_new_tokens=int(os.environ.get("MORNORK_MAX_NEW_TOKENS", "256")),
            expert_max_new_tokens=int(os.environ.get("MORNORK_EXPERT_MAX_NEW_TOKENS", "256")),
            expert_timeout_seconds=float(os.environ.get("MORNORK_EXPERT_TIMEOUT_SECONDS", "45")),
            expert_inference_url=os.environ.get("MORNORK_EXPERT_INFERENCE_URL", "http://127.0.0.1:8765/api/experts/infer").strip(),
            expert_api_key=os.environ.get("MORNORK_EXPERT_API_KEY", "").strip() or None,
            expert_backend_api_key=os.environ.get("MORNORK_EXPERT_BACKEND_API_KEY", "").strip() or None,
            training_database=project_path("MORNORK_DATABASE", "data/mornork.sqlite3"),
            training_dataset_root=project_path("MORNORK_DATASET_ROOT", "data/datasets"),
            training_artifact_root=project_path("MORNORK_TRAINED_MODEL_ROOT", "artifacts"),
            training_log_root=project_path("MORNORK_TRAINING_LOG_ROOT", "logs/training"),
            max_dataset_upload_mb=max(1, int(os.environ.get("MORNORK_MAX_DATASET_UPLOAD_MB", "256"))),
            host=os.environ.get("MORNORK_HOST", "127.0.0.1"),
            port=int(os.environ.get("MORNORK_PORT", "8765")),
            no_browser=os.environ.get("MORNORK_NO_BROWSER", "false").strip().casefold() in {"1", "true", "yes", "on"},
        )


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workspace_id: str = Field(default="local", min_length=1, max_length=200)
    message_id: str | None = Field(default=None, max_length=200)
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    user_choice: str | None = Field(default=None, max_length=200)
    workflow_id: str | None = Field(default=None, max_length=100)

    @field_validator("message")
    @classmethod
    def message_must_have_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Enter a message to route")
        return value.strip()


class ExpertCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expert_id: str = Field(min_length=2, max_length=120, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    name: str = Field(min_length=1, max_length=120)
    domain: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=5, max_length=3000)
    workspace_id: str = Field(default="local", min_length=1, max_length=200)
    status: str = Field(default="Not Configured")
    version: str | int = 1
    routing_hints: list[str] = Field(default_factory=list, max_length=100)
    model_path: str | None = Field(default=None, max_length=500)
    backend_url: str | None = Field(default=None, max_length=1000)
    inference_url: str | None = Field(default=None, max_length=1000)
    system_prompt: str | None = Field(default=None, max_length=12000)
    inference_mode: Literal["checkpoint", "mornork_role"] = "checkpoint"
    is_default: bool = False


class ExpertUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = Field(default=None, min_length=1, max_length=120)
    domain: str | None = Field(default=None, min_length=1, max_length=80)
    description: str | None = Field(default=None, min_length=5, max_length=3000)
    status: str | None = None
    version: str | int | None = None
    routing_hints: list[str] | None = Field(default=None, max_length=100)
    model_path: str | None = Field(default=None, max_length=500)
    backend_url: str | None = Field(default=None, max_length=1000)
    inference_url: str | None = Field(default=None, max_length=1000)
    system_prompt: str | None = Field(default=None, max_length=12000)
    inference_mode: Literal["checkpoint", "mornork_role"] | None = None


class StatusUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workspace_id: str = Field(min_length=1, max_length=200)
    status: str


class ExpertInferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workspace_id: str = Field(min_length=1, max_length=200)
    message_id: str = Field(min_length=1, max_length=200)
    expert_id: str = Field(min_length=1, max_length=120)
    expert_version: str | int | None = None
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)

    @field_validator("message")
    @classmethod
    def infer_message_must_have_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message cannot be blank")
        return value.strip()


class TrainingJobParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    epochs: int = Field(default=1, ge=1, le=50)
    learning_rate: float = Field(default=0.0003, gt=0, le=0.1)
    batch_size: int = Field(default=1, ge=1, le=16)
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=128)
    sequence_length: int = Field(default=128, ge=32, le=2048)
    device: Literal["auto", "cuda", "cpu"] = "auto"
    gradient_checkpointing: bool = True
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    model_size_preset: Literal["tiny", "small", "medium", "custom"] = "tiny"
    vocab_size: int = Field(default=8000, ge=256, le=64000)
    hidden_size: int = Field(default=256, ge=64, le=1024)
    layers: int = Field(default=4, ge=1, le=24)
    attention_heads: int = Field(default=4, ge=1, le=32)


class TrainingJobCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: Literal["expert", "orchestrator"]
    model_name: str = Field(min_length=3, max_length=60)
    domain: str = Field(default="Routing", min_length=2, max_length=80)
    description: str = Field(default="", max_length=2000)
    approach: Literal["scratch", "fine_tune"]
    architecture: Literal["transformer"] = "transformer"
    base_model: str | None = Field(default=None, max_length=500)
    dataset_id: str = Field(min_length=3, max_length=100)
    eval_dataset_id: str | None = Field(default=None, max_length=100)
    routing_hints: list[str] = Field(default_factory=list, max_length=100)
    config: TrainingJobParameters = Field(default_factory=TrainingJobParameters)


class ModelEvaluationReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reviewer: str = Field(min_length=2, max_length=120)
    reviewer_role: str = Field(min_length=2, max_length=160)
    qualified_reviewer_confirmed: bool
    ratings: list[bool] = Field(min_length=1, max_length=50)


class WorkflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    routing_mode: Literal["semantic", "rule", "broadcast", "hybrid"] = "semantic"
    expert_ids: list[str] = Field(default_factory=list, max_length=20)
    top_k: int = Field(default=3, ge=1, le=10)
    execution_mode: Literal["sequential", "parallel"] = "sequential"
    aggregator: Literal["first", "concatenate", "mornork"] = "mornork"
    escalation_enabled: bool = True
    minimum_rule_score: float = Field(default=0.25, ge=0, le=1)
    response_format: Literal["markdown", "plain"] = "markdown"


class WorkflowCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workspace_id: str = Field(default="w1", min_length=1, max_length=200)
    name: str = Field(min_length=3, max_length=120)
    description: str = Field(default="", max_length=2000)
    config: WorkflowConfig = Field(default_factory=WorkflowConfig)


class WorkflowUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=3, max_length=120)
    description: str = Field(default="", max_length=2000)
    config: WorkflowConfig


class WorkflowTestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    workspace_id: str = Field(default="w1", min_length=1, max_length=200)

    @field_validator("message")
    @classmethod
    def workflow_message_must_have_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Enter a message to test")
        return value.strip()


class ServiceState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        try:
            self.thresholds = RoutingThresholds.from_file(settings.thresholds_file)
        except ValueError as error:
            if str(error).startswith("Fill threshold values from calibration"):
                self.thresholds = None
            else:
                raise
        self.decision_log = JsonlDecisionLog(settings.audit_log)
        self.training_store = TrainingStore(
            settings.training_database,
            settings.training_dataset_root,
            settings.training_artifact_root,
        )
        self.workflow_store = WorkflowStore(settings.training_database)
        self.chat_model = LocalChatModel(
            settings.checkpoint,
            device=settings.generation_device,
            dtype=settings.generation_dtype,
            max_new_tokens=settings.max_new_tokens,
        )
        self.artifact_store = ModelArtifactStore()
        self.artifact_sync_lock = threading.Lock()
        self.artifact_sync_pending = False
        self.artifact_sync_last_started = 0.0
        self.artifact_sync_error: str | None = None
        self._orchestrator_lock = threading.RLock()
        self._orchestrator_model: LocalChatModel | LocalRouteModel | None = None
        self._orchestrator_model_id: str | None = None
        self.registry = ExpertRegistry(
            settings.experts_file,
            settings.expert_model_root,
            device=settings.expert_device,
            dtype=settings.expert_dtype,
            max_new_tokens=settings.expert_max_new_tokens,
            backend_api_key=settings.expert_backend_api_key,
            audit_path=settings.expert_audit_log,
            shared_model=self.chat_model,
        )
        self.training_manager = TrainingManager(
            self.training_store,
            PROJECT_ROOT,
            settings.training_log_root,
            before_gpu_job=self._offload_models_for_training,
            after_gpu_job=self._restore_models_after_training,
        )
        self.embedder = (
            SentenceTransformerEmbedder(settings.embedding_model_id, device=settings.embedder_device)
            if self.thresholds is not None
            else None
        )
        self.training_manager.start()
        self.routers: dict[str, MornorkRouter] = {}
        self._router_lock = threading.RLock()

    def _offload_models_for_training(self) -> bool:
        changed = self.chat_model.offload_for_training()
        with self._orchestrator_lock:
            if self._orchestrator_model is not None:
                changed = self._orchestrator_model.offload_for_training() or changed
        return changed

    def _restore_models_after_training(self) -> None:
        self.chat_model.restore_after_training()
        with self._orchestrator_lock:
            if self._orchestrator_model is not None:
                self._orchestrator_model.restore_after_training()

    def set_active_orchestrator(self, model_version_id: str | None) -> None:
        with self._orchestrator_lock:
            if model_version_id is None or model_version_id != self._orchestrator_model_id:
                self._orchestrator_model = None
                self._orchestrator_model_id = None

    def _get_active_orchestrator(self) -> tuple[dict[str, Any], LocalChatModel | LocalRouteModel] | None:
        version = self.training_store.get_active_model("orchestrator")
        if version is None:
            self.set_active_orchestrator(None)
            return None
        with self._orchestrator_lock:
            if self._orchestrator_model_id != version["id"]:
                checkpoint = self.artifact_store.materialize(version["artifact_path"], version.get("artifact_uri"))
                manifest_path = checkpoint / "mornork-model.json"
                metadata = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
                if metadata.get("model_task") == "route_classifier":
                    self._orchestrator_model = LocalRouteModel(
                        checkpoint, device=self.settings.generation_device, dtype=self.settings.generation_dtype)
                else:
                    self._orchestrator_model = LocalChatModel(
                        checkpoint, device=self.settings.generation_device, dtype=self.settings.generation_dtype,
                        max_new_tokens=self.settings.max_new_tokens,
                    )
                self._orchestrator_model_id = version["id"]
            return version, self._orchestrator_model

    def _route_with_active_orchestrator(self, workspace_id: str, message_id: str,
                                        message: str) -> RoutingDecision | None:
        active = self._get_active_orchestrator()
        if active is None:
            return None
        version, model = active
        profiles = list(self.routable_profiles_by_id(workspace_id).values())
        allowed = {profile.expert_id: profile for profile in profiles}
        if isinstance(model, LocalRouteModel):
            label, confidence = model.predict(message)
            if label.casefold() in {"mornork", "answer", "answer_self", "default"}:
                decision = RoutingDecision(
                    workspace_id=workspace_id, message_id=message_id, action=RouteAction.ANSWER_SELF,
                    reason="trained_orchestrator_mornork_class", mornork_version=APP_VERSION,
                    score_type="model_confidence", orchestrator_model_version=version["version"],
                )
                self.decision_log.write(decision)
                return decision
            expert = allowed.get(label)
            if expert is None:
                expert = next((item for item in profiles if item.domain.casefold() == label.casefold()
                               or item.name.casefold() == label.casefold()), None)
            if expert is None:
                return None
            score = RoutingScore(target_id=expert.expert_id, name=expert.name, domain=expert.domain,
                                 version=expert.version, score=confidence, inference_mode=expert.inference_mode)
            if confidence < 0.70:
                return None
            if confidence >= 0.85:
                decision = RoutingDecision(
                    workspace_id=workspace_id, message_id=message_id, action=RouteAction.HANDOFF,
                    reason="trained_orchestrator_clear_match", expert_scores=(score,),
                    chosen_expert_ids=(expert.expert_id,), mornork_version=APP_VERSION,
                    score_type="model_confidence", orchestrator_model_version=version["version"],
                )
            else:
                decision = RoutingDecision(
                    workspace_id=workspace_id, message_id=message_id, action=RouteAction.ANSWER_AND_OFFER,
                    reason="trained_orchestrator_moderate_match", expert_scores=(score,),
                    offer_expert_id=expert.expert_id, mornork_version=APP_VERSION,
                    score_type="model_confidence", orchestrator_model_version=version["version"],
                )
            self.decision_log.write(decision)
            return decision
        expert_context = [
            {"expert_id": profile.expert_id, "name": profile.name, "domain": profile.domain,
             "description": profile.description, "routing_hints": list(profile.routing_hints)}
            for profile in profiles
        ]
        system_prompt = (
            "You are Mornork's request router. Select only from the listed Ready to Deploy experts. "
            "Return exactly one JSON object with keys expert_ids (array), confidence (number from 0 to 1), and reason (short text). "
            "Use an empty expert_ids array when no listed expert fits. Never invent IDs.\n"
            f"Ready experts: {json.dumps(expert_context, ensure_ascii=False)}"
        )
        try:
            raw = model.generate(message, system_prompt=system_prompt)
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                return None
            decision_data = json.loads(match.group(0))
            raw_ids = decision_data.get("expert_ids", [])
            if not isinstance(raw_ids, list) or not all(isinstance(item, str) for item in raw_ids):
                return None
            confidence = float(decision_data.get("confidence", 0.0))
            if not 0 <= confidence <= 1:
                return None
            chosen = tuple(dict.fromkeys(expert_id for expert_id in raw_ids if expert_id in allowed))[:3]
            # A malformed or out-of-registry answer falls back to the deterministic router.
            if raw_ids and not chosen:
                return None
            score_rows = tuple(
                RoutingScore(target_id=expert_id, name=allowed[expert_id].name,
                             domain=allowed[expert_id].domain, version=allowed[expert_id].version,
                             score=confidence, inference_mode=allowed[expert_id].inference_mode)
                for expert_id in chosen
            )
            if not chosen:
                decision = RoutingDecision(
                    workspace_id=workspace_id, message_id=message_id, action=RouteAction.ANSWER_SELF,
                    reason="trained_orchestrator_no_match", expert_scores=score_rows,
                    mornork_version=APP_VERSION, score_type="model_confidence",
                    orchestrator_model_version=version["version"],
                )
            elif confidence < 0.70:
                return None
            elif len(chosen) > 1:
                decision = RoutingDecision(
                    workspace_id=workspace_id, message_id=message_id, action=RouteAction.ASK_USER,
                    reason="trained_orchestrator_multiple_matches", expert_scores=score_rows,
                    candidate_expert_ids=chosen, user_options=chosen + ("both",),
                    mornork_version=APP_VERSION, score_type="model_confidence",
                    orchestrator_model_version=version["version"],
                )
            elif confidence >= 0.85:
                decision = RoutingDecision(
                    workspace_id=workspace_id, message_id=message_id, action=RouteAction.HANDOFF,
                    reason="trained_orchestrator_clear_match", expert_scores=score_rows,
                    chosen_expert_ids=chosen, mornork_version=APP_VERSION,
                    score_type="model_confidence", orchestrator_model_version=version["version"],
                )
            else:
                decision = RoutingDecision(
                    workspace_id=workspace_id, message_id=message_id, action=RouteAction.ANSWER_AND_OFFER,
                    reason="trained_orchestrator_moderate_match", expert_scores=score_rows,
                    offer_expert_id=chosen[0], mornork_version=APP_VERSION,
                    score_type="model_confidence", orchestrator_model_version=version["version"],
                )
            self.decision_log.write(decision)
            return decision
        except Exception as error:
            print(f"Trained orchestrator routing failed ({type(error).__name__}); message content omitted.")
            return None

    def route(self, workspace_id: str, message_id: str, message: str) -> RoutingDecision:
        trained_decision = self._route_with_active_orchestrator(workspace_id, message_id, message)
        if trained_decision is not None:
            return trained_decision
        if self.thresholds is None or self.embedder is None:
            return RoutingDecision(
                workspace_id=workspace_id,
                message_id=message_id,
                action=RouteAction.ANSWER_SELF,
                reason="routing_not_calibrated",
                mornork_version=APP_VERSION,
            )
        profiles = list(self.routable_profiles_by_id(workspace_id).values())
        profiles = [
            profile if profile.inference_url else replace(profile, inference_url=self.settings.expert_inference_url)
            for profile in profiles
        ]
        with self._router_lock:
            router = self.routers.get(workspace_id)
            if router is None:
                router = MornorkRouter(
                    self.embedder,
                    self.thresholds,
                    expert_profiles=profiles,
                    decision_writer=self.decision_log,
                    mornork_version=APP_VERSION,
                )
                self.routers[workspace_id] = router
            else:
                router.refresh_experts(profiles)
            return router.route(message, workspace_id=workspace_id, message_id=message_id)

    def routable_profiles_by_id(self, workspace_id: str) -> dict[str, ExpertProfile]:
        active_s3_ids = {
            str(row.get("expert_id"))
            for row in self.registry.list_profiles(workspace_id)
            if row.get("status") == "Ready to Deploy"
            and row.get("trained_model_version_id")
            and str(row.get("artifact_uri") or "").startswith("s3://")
        }
        profiles = [profile for profile in self.registry.list_routable_profiles(workspace_id)
                    if profile.expert_id in active_s3_ids]
        return {
            profile.expert_id: profile if profile.inference_url else replace(profile, inference_url=self.settings.expert_inference_url)
            for profile in profiles
        }

    def execute_workflow(self, config: dict[str, Any], workspace_id: str,
                         message_id: str, message: str) -> dict[str, Any]:
        """Run a saved workflow config and return a private trace plus generated answers."""
        started = time.perf_counter()
        trace: list[dict[str, Any]] = []
        mode = config.get("routing_mode", "semantic")
        execution = config.get("execution_mode", "sequential")
        aggregator = config.get("aggregator", "mornork")
        top_k = int(config.get("top_k", 3))
        profiles = self.routable_profiles_by_id(workspace_id)
        configured_ids = config.get("expert_ids", [])
        allowed_ids = [expert_id for expert_id in configured_ids if expert_id in profiles]
        selected_ids: list[str] = []
        scores: dict[str, float] = {}
        score_type = "rule_match"
        route_reason = "workflow_no_match"

        def rule_scores() -> dict[str, float]:
            message_tokens = set(re.findall(r"[a-z0-9]+", message.casefold()))
            values: dict[str, float] = {}
            for expert_id in allowed_ids:
                profile = profiles[expert_id]
                hints = list(profile.routing_hints) + [profile.domain, profile.name]
                best = 0.0
                for hint in hints:
                    hint_tokens = set(re.findall(r"[a-z0-9]+", hint.casefold()))
                    if not hint_tokens:
                        continue
                    overlap = len(message_tokens & hint_tokens) / len(hint_tokens)
                    if hint.casefold() in message.casefold():
                        overlap = 1.0
                    best = max(best, overlap)
                values[expert_id] = best
            return values

        if mode == "broadcast":
            selected_ids = allowed_ids[:top_k]
            scores = {expert_id: 1.0 for expert_id in selected_ids}
            score_type = "broadcast_selection"
            route_reason = "workflow_broadcast"
        elif mode in {"rule", "hybrid"}:
            rule_values = rule_scores()
            minimum = float(config.get("minimum_rule_score", 0.25))
            selected_ids = [expert_id for expert_id in sorted(rule_values, key=rule_values.get, reverse=True)
                            if rule_values[expert_id] >= minimum][:top_k]
            scores = {expert_id: rule_values[expert_id] for expert_id in selected_ids}
            route_reason = "workflow_rule_match" if selected_ids else "workflow_rule_no_match"
        if mode in {"semantic", "hybrid"} and (mode == "semantic" or not selected_ids):
            decision = self.route(workspace_id, message_id, message)
            score_type = decision.score_type
            selected_ids = [expert_id for expert_id in decision.chosen_expert_ids if expert_id in allowed_ids]
            if not selected_ids and decision.offer_expert_id in allowed_ids:
                selected_ids = [decision.offer_expert_id]
            if not selected_ids and decision.action == RouteAction.ASK_USER:
                selected_ids = [expert_id for expert_id in decision.candidate_expert_ids if expert_id in allowed_ids][:top_k]
            scores = {item.target_id: item.score for item in decision.expert_scores if item.target_id in selected_ids}
            route_reason = "workflow_semantic_" + decision.reason
        selected_ids = list(dict.fromkeys(selected_ids))[:top_k]
        trace.append({"node": "orchestrator", "status": "completed", "mode": mode,
                      "selected_expert_ids": selected_ids, "reason": route_reason})
        decision_action = RouteAction.HANDOFF if selected_ids else RouteAction.ANSWER_SELF
        expert_scores = tuple(
            RoutingScore(target_id=expert_id, name=profiles[expert_id].name, domain=profiles[expert_id].domain,
                         version=profiles[expert_id].version, score=float(scores.get(expert_id, 0.0)),
                         inference_mode=profiles[expert_id].inference_mode)
            for expert_id in selected_ids
        )
        decision = RoutingDecision(
            workspace_id=workspace_id, message_id=message_id, action=decision_action,
            reason=route_reason, expert_scores=expert_scores, chosen_expert_ids=tuple(selected_ids),
            mornork_version=APP_VERSION,
            score_type=score_type,
        )
        self.decision_log.write(decision)
        answers: list[dict[str, Any]] = []
        parallel_safe = all(profiles[expert_id].inference_mode != "mornork_role" for expert_id in selected_ids)
        if selected_ids:
            trace.append({"node": "experts", "status": "running",
                          "execution_mode": "parallel" if execution == "parallel" and parallel_safe else "sequential",
                          "expert_ids": selected_ids})

            def call(expert_id: str) -> dict[str, Any]:
                profile = profiles[expert_id]
                try:
                    answer = self.registry.infer({
                        "workspace_id": workspace_id, "message_id": message_id, "message": message,
                        "expert_id": expert_id, "expert_version": profile.version,
                    })
                    return {"expert_id": expert_id, "name": profile.name, "domain": profile.domain,
                            "version": profile.version, "prototype": profile.inference_mode == "mornork_role",
                            "text": answer}
                except Exception as error:
                    return {"expert_id": expert_id, "name": profile.name, "error": type(error).__name__}

            if execution == "parallel" and parallel_safe and len(selected_ids) > 1:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=min(len(selected_ids), 4)) as pool:
                    answers = list(pool.map(call, selected_ids))
            else:
                answers = [call(expert_id) for expert_id in selected_ids]
            successes = [item for item in answers if "text" in item]
            trace.append({"node": "experts", "status": "completed" if successes else "failed",
                          "succeeded": len(successes), "failed": len(answers) - len(successes)})
        else:
            successes = []

        failed_count = len(answers) - len(successes)
        if successes and failed_count and config.get("escalation_enabled", True):
            partial = "\n\n".join(f"{item['name']} ({item['domain']}):\n{item['text']}" for item in successes)
            final_answer = self.chat_model.generate(
                "User request:\n" + message + "\n\nResponses received before another expert failed:\n" + partial,
                system_prompt="Complete the response using the available expert responses. State any remaining uncertainty. Do not invent facts or citations.",
            )
            final_by = "Mornork escalation"
            trace.append({"node": "escalation", "status": "completed", "target": "Mornork",
                          "reason": "partial_expert_failure"})
        elif successes:
            if aggregator == "first" or len(successes) == 1:
                final_answer = successes[0]["text"]
                final_by = successes[0]["name"]
            elif aggregator == "concatenate":
                final_answer = "\n\n".join(f"### {item['name']}\n\n{item['text']}" for item in successes)
                final_by = "Workflow aggregator"
            else:
                context = "\n\n".join(f"{item['name']} ({item['domain']}):\n{item['text']}" for item in successes)
                final_answer = self.chat_model.generate(
                    "User request:\n" + message + "\n\nExpert responses:\n" + context,
                    system_prompt="Synthesize the expert responses into one useful answer. Preserve uncertainty and disagreements. Do not invent facts or citations.",
                )
                final_by = "Mornork aggregator"
            trace.append({"node": "aggregator", "status": "completed", "mode": aggregator,
                          "source_count": len(successes)})
        elif config.get("escalation_enabled", True):
            final_answer = self.chat_model.generate(message)
            final_by = "Mornork" if not selected_ids else "Mornork fallback"
            trace.append({"node": "escalation", "status": "completed", "target": "Mornork"})
        else:
            final_answer = (
                "No selected expert completed this request, and Mornork escalation is disabled."
                if selected_ids else "No configured Ready to Deploy expert matched this request, and Mornork escalation is disabled."
            )
            final_by = "Workflow"
            trace.append({"node": "escalation", "status": "skipped", "target": "disabled"})
        if config.get("response_format") == "plain":
            final_answer = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", final_answer)
            final_answer = re.sub(r"(\*\*|__|`|\*|_)", "", final_answer)
        trace.append({"node": "response", "status": "completed", "source": final_by})
        trace.append({"node": "total", "status": "completed", "elapsed_ms": round((time.perf_counter() - started) * 1000)})
        return {"decision": decision, "trace": trace, "answers": answers,
                "final_answer": final_answer, "final_by": final_by,
                "expert_ids": selected_ids}


def _valid_training_row(row: dict[str, Any]) -> bool:
    values = {str(key).strip().casefold(): value for key, value in row.items()}
    if "approved" in values and str(values["approved"]).strip().casefold() not in {"true", "yes", "1", "approved"}:
        return False
    if any(str(values.get(key) or "").strip() for key in ("text", "content", "document", "article")):
        return True
    prompt = any(str(values.get(key) or "").strip() for key in ("prompt", "question", "instruction", "input"))
    response = any(str(values.get(key) or "").strip() for key in ("response", "answer", "output", "completion"))
    if prompt and response:
        return True
    message = any(str(values.get(key) or "").strip() for key in ("message", "query", "text", "input", "prompt", "question"))
    route = any(str(values.get(key) or "").strip() for key in ("route", "label", "expert_id", "target"))
    return message and route


def _split_row_counts(dataset: dict[str, Any]) -> dict[str, int]:
    if not any(str(column).strip().casefold() == "split" for column in dataset.get("columns", [])):
        return {}
    counts: Counter[str] = Counter()
    for row in read_records(dataset):
        if is_approved(row) and _valid_training_row(row):
            split = row_split(row)
            if split in {"train", "validation", "test"}:
                counts[split] += 1
    return dict(counts)


def _inspect_training_dataset(path: Path) -> tuple[int, list[str]]:
    """Validate supported local training files without loading their rows into memory."""
    suffix = path.suffix.casefold()
    if suffix == ".txt":
        with path.open("r", encoding="utf-8-sig") as source:
            count = sum(1 for line in source if line.strip())
        if count < 2:
            raise ValueError("Text data needs at least two non-empty lines")
        return count, ["text"]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            columns = [str(column or "").strip() for column in (reader.fieldnames or [])]
            normalized = {column.casefold() for column in columns}
            has_text = bool(normalized & {"text", "content", "document", "article"})
            has_prompt_response = bool(normalized & {"prompt", "question", "instruction", "input"}) and bool(
                normalized & {"response", "answer", "output", "completion"}
            )
            has_route_schema = bool(normalized & {"message", "query", "text", "input", "prompt", "question"}) and bool(
                normalized & {"route", "label", "expert_id", "target"}
            )
            if not has_text and not has_prompt_response and not has_route_schema:
                raise ValueError("CSV needs text/content, prompt/response, or orchestrator message/route columns")
            count = sum(1 for row in reader if _valid_training_row(row))
        if count < 2:
            raise ValueError("CSV needs at least two non-empty training rows")
        return count, columns
    if suffix == ".jsonl":
        count = 0
        columns: set[str] = set()
        with path.open("r", encoding="utf-8-sig") as source:
            for line_no, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on line {line_no}") from error
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL line {line_no} must contain an object")
                columns.update(str(key) for key in row)
                if _valid_training_row(row):
                    count += 1
            if count < 2:
                raise ValueError("JSONL needs at least two rows with text, prompt/response, or message/route content")
        return count, sorted(columns)
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise RuntimeError("PDF support requires the training extra: python -m pip install -e .[training]") from error
        reader = PdfReader(str(path))
        count = sum(1 for page in reader.pages if (page.extract_text() or "").strip())
        if count < 1:
            raise ValueError("No selectable text was found in the PDF. OCR scanned documents before uploading")
        return count, ["page_text"]
    raise ValueError("Supported dataset formats are .txt, .csv, .jsonl, and text-based .pdf")


def _preview_training_dataset(dataset: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    path = Path(dataset["file_path"])
    rows: list[dict[str, Any]] = []
    if path.suffix.casefold() == ".txt":
        rows = [{"text": line.strip()} for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()][:limit]
    elif path.suffix.casefold() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))[:limit]
    elif path.suffix.casefold() == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as source:
            for line in source:
                if line.strip() and len(rows) < limit:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        rows.append(value)
    elif path.suffix.casefold() == ".pdf":
        from pypdf import PdfReader
        rows = [{"page": number, "text": (page.extract_text() or "").strip()}
                for number, page in enumerate(PdfReader(str(path)).pages, start=1)
                if (page.extract_text() or "").strip()][:limit]

    preview: list[dict[str, Any]] = []
    for row in rows:
        preview.append({str(key): (str(value)[:1200] + "…" if len(str(value)) > 1200 else value)
                        for key, value in row.items()})
    return preview


def _public_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    return {key: dataset[key] for key in ("id", "name", "domain", "file_name", "file_format", "row_count", "byte_size", "columns", "created_at")}


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in job.items() if key not in {"worker_pid", "output_dir", "file_path"}}


def _qa_example(row: dict[str, Any]) -> dict[str, str] | None:
    values = {str(key).strip().casefold(): value for key, value in row.items()}
    if "approved" in values and str(values["approved"]).strip().casefold() not in {"true", "yes", "1", "approved"}:
        return None

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
    context = "\n".join(part for part in context_parts if part)
    return {"prompt": prompt, "expected": response, "context": context}


def _load_qa_examples(dataset: dict[str, Any], split: str | None = None) -> list[dict[str, str]]:
    path = Path(dataset["file_path"])
    has_split = any(str(column).strip().casefold() == "split" for column in dataset.get("columns", []))
    examples: list[dict[str, str]] = []

    def include(row: dict[str, Any]) -> bool:
        if split is None or not has_split:
            return True
        values = {str(key).strip().casefold(): value for key, value in row.items()}
        value = str(values.get("split") or "").strip().casefold()
        if split == "evaluation":
            return value in {"validation", "valid", "val", "test"}
        if split == "validation":
            return value in {"validation", "valid", "val"}
        return value == split

    if path.suffix.casefold() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                item = _qa_example(row) if include(row) else None
                if item:
                    examples.append(item)
    elif path.suffix.casefold() == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as source:
            for line in source:
                if line.strip():
                    row = json.loads(line)
                    if isinstance(row, dict):
                        item = _qa_example(row) if include(row) else None
                        if item:
                            examples.append(item)
    return examples


def _metric_tokens(text: str) -> list[str]:
    normalized = text.casefold().replace(",", "")
    return re.findall(r"[a-z]+|[+-]?\d+(?:\.\d+)?", normalized)


def _last_numeric_value(text: str) -> float | None:
    matches = re.findall(r"(?<![A-Za-z0-9])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", text)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def _calculate_answer_metrics(cases: list[dict[str, str]], split: str,
                              requested_cases: int, available_cases: int) -> dict[str, Any]:
    return calculate_answer_metrics(cases, split, requested_cases, available_cases)


def _calculate_route_metrics(cases: list[dict[str, str]], labels: list[str], split: str,
                             requested_cases: int, available_cases: int) -> dict[str, Any]:
    label_index = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for item in cases:
        matrix[label_index[item["expected"]]][label_index[item["actual"]]] += 1
    precisions, recalls, f1s = [], [], []
    for index in range(len(labels)):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(len(labels)) if row != index)
        fn = sum(matrix[index][column] for column in range(len(labels)) if column != index)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    accuracy = sum(matrix[index][index] for index in range(len(labels))) / max(1, len(cases))
    mean = lambda values: sum(values) / max(1, len(values))
    return {
        "metric_version": 2, "task_type": "route_classification", "split": split,
        "examples_evaluated": len(cases), "available_examples": available_cases,
        "requested_cases": requested_cases, "accuracy": accuracy,
        "macro_precision": mean(precisions), "macro_recall": mean(recalls), "macro_f1": mean(f1s),
        "confusion_matrix": {"labels": labels, "rows": matrix, "row_meaning": "actual", "column_meaning": "predicted"},
        "confusion_matrix_note": "Rows are actual routes; columns are predicted routes. Label order is listed above the matrix.",
    }


def _sse(event_name: str, payload: dict[str, Any]) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _require_expert_key(state: ServiceState, authorization: str | None) -> None:
    expected = state.settings.expert_api_key
    if expected and authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="A valid expert API key is required")


def _sync_completed_artifacts_to_s3(state: ServiceState) -> None:
    """Persist saved in-run evaluations and sync completed checkpoints to S3."""
    try:
        artifacts = state.artifact_store.list_model_artifacts()
        existing = {str(item.get("id")): item for item in artifacts}
        versions = state.training_store.list_model_versions()
    except Exception:
        state.artifact_sync_error = "Could not check S3 for completed model artifacts."
        return

    stored_ids = {str(version.get("id") or "") for version in versions}
    for version in versions:
        model_id = str(version.get("id") or "")
        if not model_id:
            continue
        artifact = existing.get(model_id)
        local_evaluation = version.get("evaluation") or {}
        manifest = dict((artifact or {}).get("manifest") or {})
        evaluation = local_evaluation if _has_saved_inrun_evaluation(local_evaluation) else (
            manifest.get("evaluation") or {}
        )
        if not _has_saved_inrun_evaluation(evaluation):
            continue

        job_id = str(version.get("job_id") or "")
        job = state.training_store.get_job(job_id) if job_id else None
        if job and job.get("status") not in {"built_evaluated", "built_evaluation_required"}:
            continue

        model_dir = Path(str(version.get("artifact_path") or ""))
        manifest_path = model_dir / "mornork-model.json"
        if manifest_path.is_file():
            try:
                local_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(local_manifest, dict):
                    manifest = {**local_manifest, **manifest}
            except (OSError, json.JSONDecodeError):
                pass

        manifest.update({
            "id": model_id,
            "target": version["target"],
            "model_name": version["model_name"],
            "version": version["version"],
            "evaluation": evaluation,
            "readiness": "ready_to_deploy",
        })
        try:
            if artifact:
                artifact_uri = str(artifact["artifact_uri"])
                state.artifact_store.update_manifest(artifact_uri, manifest)
            else:
                if not job or not model_dir.is_dir() or not manifest_path.is_file():
                    state.artifact_sync_error = (
                        f"{version.get('model_name') or model_id} has saved evaluation results, "
                        "but its local checkpoint is unavailable for S3 upload."
                    )
                    continue
                artifact_uri = state.artifact_store.upload_directory(
                    model_dir, str(version["target"]), model_id, manifest=manifest
                )
                if not artifact_uri:
                    continue
            state.training_store.update_artifact_uri(model_id, artifact_uri)
            state.training_store.mark_model_ready_for_deployment(model_id)
            if manifest_path.is_file():
                manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            existing[model_id] = {
                "id": model_id, "artifact_uri": artifact_uri, "manifest": manifest,
            }
        except Exception:
            state.artifact_sync_error = (
                f"{version.get('model_name') or model_id} could not be synced to S3. "
                "Check bucket write permissions and backend logs."
            )

    # Normalize saved S3 manifests too, including artifacts that outlive the local run database.
    for artifact in artifacts:
        if str(artifact.get("id") or "") in stored_ids:
            continue
        manifest = artifact.get("manifest") or {}
        if not isinstance(manifest, dict) or manifest.get("readiness") == "ready_to_deploy":
            continue
        evaluation = manifest.get("evaluation") or {}
        if not _has_saved_inrun_evaluation(evaluation):
            continue
        manifest["readiness"] = "ready_to_deploy"
        try:
            state.artifact_store.update_manifest(str(artifact["artifact_uri"]), manifest)
        except Exception:
            state.artifact_sync_error = (
                f"{manifest.get('model_name') or artifact.get('id')} has saved evaluation results, "
                "but its S3 readiness record could not be updated. Check bucket write permissions."
            )

def _start_completed_artifact_sync(state: ServiceState) -> bool:
    if not state.artifact_store.upload_enabled or state.artifact_store.storage != "s3":
        return False
    if state.artifact_sync_pending:
        return True
    now = time.monotonic()
    if state.artifact_sync_last_started and now - state.artifact_sync_last_started < 60:
        return False
    if not state.artifact_sync_lock.acquire(blocking=False):
        return True
    state.artifact_sync_pending = True
    state.artifact_sync_last_started = now
    state.artifact_sync_error = None

    def run_sync() -> None:
        try:
            _sync_completed_artifacts_to_s3(state)
        finally:
            state.artifact_sync_pending = False
            state.artifact_sync_lock.release()

    try:
        threading.Thread(target=run_sync, daemon=True, name="mornork-s3-artifact-sync").start()
    except Exception:
        state.artifact_sync_pending = False
        state.artifact_sync_lock.release()
        state.artifact_sync_error = "Could not start the automatic S3 artifact sync."
    return True


def _start_s3_catalog_warmup(state: ServiceState) -> None:
    """Begin S3 discovery as soon as the FastAPI service starts."""
    if state.artifact_store.storage != "s3" or not state.artifact_store.bucket:
        return
    if state.artifact_store.upload_enabled:
        _start_completed_artifact_sync(state)
        return

    def warm_catalog() -> None:
        try:
            state.artifact_store.list_model_artifacts()
        except Exception:
            state.artifact_sync_error = "Could not read the configured S3 model catalog during backend startup."

    threading.Thread(target=warm_catalog, daemon=True, name="mornork-s3-catalog-warmup").start()


def create_app(settings: Settings | None = None) -> FastAPI:
    load_dotenv()
    resolved_settings = settings or Settings.from_environment()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = ServiceState(resolved_settings)
        _start_s3_catalog_warmup(app.state.runtime)
        try:
            yield
        finally:
            runtime = app.state.runtime
            if runtime is not None:
                runtime.training_manager.stop()
            app.state.runtime = None

    app = FastAPI(title="Mornork and Experts", version=APP_VERSION, lifespan=lifespan)
    frontend_origins = [
        origin.strip().rstrip("/")
        for origin in os.environ.get(
            "MORNORK_FRONTEND_ORIGINS",
            "http://127.0.0.1:5500,http://localhost:5500",
        ).split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=frontend_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/status")
    def status(request: FastAPIRequest, workspace_id: str = Query(default="w1", max_length=200)) -> dict[str, Any]:
        state: ServiceState = request.app.state.runtime
        profiles = state.registry.list_profiles(workspace_id)
        routable = list(state.routable_profiles_by_id(workspace_id).values())
        cuda_available = torch.cuda.is_available()
        cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else None
        cuda_total_memory_mb = (
            round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)) if cuda_available else None
        )
        training_jobs = state.training_store.list_jobs(limit=1000)
        return {
            "api_version": APP_VERSION,
            "capabilities": {"task_metrics": True, "route_classifier_training": True,
                             "artifact_storage": state.artifact_store.storage},
            "model_id": state.settings.embedding_model_id,
            "device": state.settings.embedder_device,
            "routing_enabled": state.thresholds is not None,
            "routing_mode": state.settings.routing_mode,
            "expert_handoff_enabled": bool(routable),
            "ready_experts": [
                {"expert_id": profile.expert_id, "name": profile.name, "domain": profile.domain, "version": profile.version}
                for profile in routable
            ],
            "expert_profiles": len(profiles),
            "expert_service": "ready",
            "prototype_experts": sum(profile.inference_mode == "mornork_role" for profile in routable),
            "audit_logging": True,
            "audit_log": str(state.settings.audit_log),
            "generation_enabled": True,
            "checkpoint": str(state.settings.checkpoint),
            "generation_device": state.chat_model.device,
            "generation_dtype": state.chat_model.dtype,
            "cuda_available": cuda_available,
            "cuda_device_name": cuda_device_name,
            "cuda_total_memory_mb": cuda_total_memory_mb,
            "training_jobs_active": sum(job["status"] in {"queued", "running"} for job in training_jobs),
            "training_device_mode": "single local worker; CUDA jobs are serialized",
        }

    @app.get("/api/route")
    async def route_preview(
        request: FastAPIRequest,
        message: str = Query(min_length=1, max_length=MAX_MESSAGE_CHARS),
        workspace_id: str = Query(default="w1", max_length=200),
        message_id: str = Query(default="", max_length=200),
    ) -> dict[str, Any]:
        state: ServiceState = request.app.state.runtime
        decision = await asyncio.to_thread(state.route, workspace_id, message_id or uuid4().hex, message)
        return decision.to_dict()

    @app.get("/api/training/datasets")
    def list_training_datasets(request: FastAPIRequest) -> dict[str, Any]:
        state: ServiceState = request.app.state.runtime
        return {"datasets": [_public_dataset(item) for item in state.training_store.list_datasets()]}

    @app.get("/api/training/datasets/{dataset_id}/preview")
    def preview_training_dataset(dataset_id: str, request: FastAPIRequest,
                                 limit: int = Query(default=5, ge=1, le=10)) -> dict[str, Any]:
        state: ServiceState = request.app.state.runtime
        dataset = state.training_store.get_dataset(dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return {"dataset": _public_dataset(dataset), "rows": _preview_training_dataset(dataset, limit)}

    @app.post("/api/training/datasets", status_code=201)
    async def upload_training_dataset(
        request: FastAPIRequest,
        filename: str = Query(min_length=1, max_length=255),
        name: str = Query(default="", max_length=120),
        domain: str = Query(default="General", min_length=2, max_length=80),
    ) -> dict[str, Any]:
        state: ServiceState = request.app.state.runtime
        safe_name = Path(filename.replace("\\", "/")).name
        suffix = Path(safe_name).suffix.casefold()
        if suffix not in {".txt", ".csv", ".jsonl", ".pdf"}:
            raise HTTPException(status_code=415, detail="Upload .txt, .csv, .jsonl, or text-based .pdf data")
        maximum_bytes = state.settings.max_dataset_upload_mb * 1024 * 1024
        declared_size = request.headers.get("content-length")
        if declared_size and declared_size.isdigit() and int(declared_size) > maximum_bytes:
            raise HTTPException(status_code=413, detail=f"Dataset exceeds the {state.settings.max_dataset_upload_mb} MB upload limit")
        destination = state.settings.training_dataset_root / f"{uuid4().hex}{suffix}"
        size = 0
        try:
            with destination.open("wb") as target:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > maximum_bytes:
                        raise HTTPException(status_code=413, detail=f"Dataset exceeds the {state.settings.max_dataset_upload_mb} MB upload limit")
                    target.write(chunk)
            if size == 0:
                raise HTTPException(status_code=400, detail="The uploaded dataset is empty")
            row_count, columns = await asyncio.to_thread(_inspect_training_dataset, destination)
            dataset = await asyncio.to_thread(
                state.training_store.add_dataset,
                name=name.strip() or Path(safe_name).stem,
                domain=domain.strip(),
                file_name=safe_name,
                file_path=destination,
                file_format=suffix.lstrip("."),
                row_count=row_count,
                byte_size=size,
                columns=columns,
            )
            return {"dataset": _public_dataset(dataset), "validation": {"status": "valid", "rows": row_count, "columns": columns}}
        except HTTPException:
            destination.unlink(missing_ok=True)
            raise
        except RuntimeError as error:
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=503, detail=str(error)) from error
        except (ValueError, UnicodeDecodeError, csv.Error) as error:
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception:
            destination.unlink(missing_ok=True)
            raise

    @app.post("/api/training/jobs", status_code=202)
    def create_training_job(payload: TrainingJobCreate, request: FastAPIRequest) -> dict[str, Any]:
        state: ServiceState = request.app.state.runtime
        if payload.approach == "fine_tune" and not (payload.base_model or "").strip():
            raise HTTPException(status_code=422, detail="Choose a base model for fine-tuning")
        if payload.approach == "scratch" and payload.base_model:
            raise HTTPException(status_code=422, detail="A from-scratch model cannot use pretrained base weights")
        primary = state.training_store.get_dataset(payload.dataset_id)
        if primary is None:
            raise HTTPException(status_code=404, detail="Training dataset not found")
        evaluation = state.training_store.get_dataset(payload.eval_dataset_id) if payload.eval_dataset_id else None
        if payload.eval_dataset_id and evaluation is None:
            raise HTTPException(status_code=404, detail="Evaluation dataset not found")
        if payload.eval_dataset_id == payload.dataset_id:
            raise HTTPException(status_code=422, detail="Training and evaluation datasets must be different files")
        if payload.config.device == "cuda" and not torch.cuda.is_available():
            raise HTTPException(status_code=422, detail="CUDA was selected, but this PyTorch environment cannot see a CUDA GPU")
        config = payload.config.model_dump()
        if payload.approach == "scratch" and config["model_size_preset"] != "custom":
            presets = {
                "tiny": {"hidden_size": 128, "layers": 2, "attention_heads": 4, "vocab_size": 4096, "sequence_length": 128},
                "small": {"hidden_size": 256, "layers": 4, "attention_heads": 4, "vocab_size": 8000, "sequence_length": 128},
                "medium": {"hidden_size": 384, "layers": 6, "attention_heads": 6, "vocab_size": 16000, "sequence_length": 256},
            }
            config.update(presets[config["model_size_preset"]])
        if config["hidden_size"] % config["attention_heads"]:
            raise HTTPException(status_code=422, detail="Hidden size must be divisible by the number of attention heads")
        if payload.approach == "scratch" and payload.config.device != "cpu" and torch.cuda.is_available():
            memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            hidden, layers, vocab = config["hidden_size"], config["layers"], config["vocab_size"]
            estimated_params = vocab * hidden + layers * 12 * hidden * hidden
            estimated_bytes = estimated_params * 16 + config["batch_size"] * config["sequence_length"] * hidden * layers * 32
            if estimated_bytes > memory_mb * 1024 * 1024 * 0.75:
                raise HTTPException(status_code=422, detail=(
                    f"The selected scratch model is too large for approximately {int(memory_mb)} MB VRAM. "
                    "Choose a smaller model-size preset or CPU."
                ))
        if torch.cuda.is_available() and payload.config.device != "cpu":
            memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            if memory_mb <= 8192 and (config["batch_size"] > 2 or config["sequence_length"] > 512):
                raise HTTPException(status_code=422, detail=(
                    f"This run requests batch size {config['batch_size']} and sequence length {config['sequence_length']} "
                    f"on a {int(memory_mb)} MB GPU. Use batch size at most 2 and sequence length at most 512, or select CPU."
                ))
        if payload.target == "expert":
            suffix = Path(primary["file_path"]).suffix.casefold()
            columns = {str(column).strip().casefold() for column in primary.get("columns", [])}
            has_qa_columns = bool(columns & {"prompt", "question", "instruction", "input"}) and bool(
                columns & {"response", "answer", "output", "completion"})
            has_corpus_column = bool(columns & {"text", "content", "document", "article", "page_text"})
            if suffix in {".csv", ".jsonl"} and not has_qa_columns and not has_corpus_column:
                raise HTTPException(status_code=422, detail="Expert answer training expects corpus text or prompt/response examples; message/route data is for the orchestrator")
        if payload.target == "expert" and payload.approach == "fine_tune":
            qa_examples = _load_qa_examples(primary, "train" if any(str(c).casefold() == "split" for c in primary.get("columns", [])) else None)
            if len(qa_examples) < 2:
                raise HTTPException(status_code=422, detail=(
                    "Expert fine-tuning needs at least two approved prompt/response examples. "
                    "Use columns such as prompt, response, optional context, and split."
                ))
        has_explicit_training_split = any(str(column).strip().casefold() == "split" for column in primary.get("columns", []))
        if payload.target == "expert" and has_explicit_training_split:
            primary_splits = _split_row_counts(primary)
            evaluation_splits = _split_row_counts(evaluation) if evaluation else {}
            if not primary_splits.get("train"):
                raise HTTPException(status_code=422, detail="Expert split data needs non-empty split=train rows")
            if not primary_splits.get("validation") and not evaluation_splits.get("validation"):
                raise HTTPException(status_code=422, detail="Expert split data needs non-empty split=validation rows in the training or separate evaluation file")
            if evaluation is None and primary_splits.get("test", 0) < 5:
                raise HTTPException(status_code=422, detail="Keep at least 5 final examples in split=test, or upload a separate test dataset")
        if payload.target == "orchestrator":
            explicit = any(str(column).strip().casefold() == "split" for column in primary.get("columns", []))
            try:
                if explicit:
                    train_rows = load_route_examples(primary, "train")
                    validation_rows = load_route_examples(primary, "validation")
                    test_rows = load_route_examples(primary, "test")
                    if evaluation is not None:
                        if any(str(c).strip().casefold() == "split" for c in evaluation.get("columns", [])):
                            validation_rows = validation_rows or load_route_examples(evaluation, "validation")
                            test_rows = test_rows or load_route_examples(evaluation, "test")
                        elif not test_rows:
                            test_rows = load_route_examples(evaluation)
                else:
                    train_rows, validation_rows, test_rows = deterministic_route_splits(
                        load_route_examples(primary), payload.config.seed)
                    if evaluation is not None:
                        test_rows = (load_route_examples(evaluation, "test")
                                     if any(str(c).strip().casefold() == "split" for c in evaluation.get("columns", []))
                                     else load_route_examples(evaluation))
                labels = {row["route"] for row in train_rows}
                if len(train_rows) < 2 or len(labels) < 2:
                    raise ValueError("Training data needs at least two examples and two distinct route labels")
                if not validation_rows or not test_rows:
                    raise ValueError("Provide disjoint train, validation, and test rows, or a separate test dataset")
                unseen = {row["route"] for row in validation_rows + test_rows} - labels
                if unseen:
                    raise ValueError(f"Held-out routes are missing from training: {', '.join(sorted(unseen))}")
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
        job_data = {
            "target": payload.target,
            "model_name": payload.model_name.strip(),
            "domain": payload.domain.strip(),
            "description": payload.description.strip(),
            "approach": payload.approach,
            "architecture": payload.architecture,
            "base_model": payload.base_model.strip() if payload.base_model else None,
            "dataset_id": payload.dataset_id,
            "eval_dataset_id": payload.eval_dataset_id,
            "config": {**config, "routing_hints": payload.routing_hints},
        }
        try:
            job = state.training_store.create_job(job_data)
            state.training_manager.enqueue(job["id"])
        except Exception as error:
            raise HTTPException(status_code=503, detail=f"Could not queue training: {type(error).__name__}") from error
        return {"job": _public_job(job), "training_dataset": _public_dataset(primary),
                "evaluation_dataset": _public_dataset(evaluation) if evaluation else None}

    @app.get("/api/training/jobs")
    def list_training_jobs(request: FastAPIRequest, target: Literal["expert", "orchestrator"] | None = None) -> dict[str, Any]:
        state: ServiceState = request.app.state.runtime
        jobs = state.training_store.list_jobs()
        if target:
            jobs = [job for job in jobs if job["target"] == target]
        return {"jobs": [_public_job(job) for job in jobs]}

    @app.get("/api/training/jobs/{job_id}")
    def get_training_job(job_id: str, request: FastAPIRequest) -> dict[str, Any]:
        state: ServiceState = request.app.state.runtime
        job = state.training_store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Training run not found")
        return {"job": _public_job(job)}

    @app.post("/api/training/jobs/{job_id}/keep")
    def keep_training_job(job_id: str, request: FastAPIRequest) -> dict[str, Any]:
        state: ServiceState = request.app.state.runtime
        try:
            job = state.training_store.keep_job(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Training run not found") from error
        return {"job": _public_job(job)}

    @app.delete("/api/training/jobs/{job_id}")
    def delete_training_job(job_id: str, request: FastAPIRequest) -> dict[str, Any]:
        state: ServiceState = request.app.state.runtime
        job = state.training_store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Training run not found")
        if job["status"] not in {"built_evaluation_required", "built_evaluated", "failed", "cancelled"}:
            raise HTTPException(status_code=409, detail="Cancel and wait for the run to stop before deleting it")

        versions = [version for version in state.training_store.list_model_versions()
                    if version["job_id"] == job_id]
        active_orchestrator = state.training_store.get_active_model("orchestrator")
        for version in versions:
            if active_orchestrator and active_orchestrator["id"] == version["id"]:
                raise HTTPException(status_code=409, detail="This run provides the active orchestrator model; deactivate it before deleting")
            artifact_path = Path(version["artifact_path"]).resolve()
            for profile in state.registry.list_profiles("w1"):
                model_path = profile.get("model_path")
                if not model_path:
                    continue
                try:
                    registered_path = state.registry._resolve_model_path(str(model_path))
                except ValueError:
                    continue
                if registered_path == artifact_path:
                    raise HTTPException(status_code=409, detail="This checkpoint is used by an expert profile; remove or replace that profile checkpoint first")

        try:
            for version in versions:
                state.artifact_store.delete_artifact(version.get("artifact_uri"))
            state.training_store.delete_job(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Training run not found") from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=503, detail=f"Could not remove the run's configured S3 model artifact: {type(error).__name__}") from error
        except OSError as error:
            raise HTTPException(status_code=500, detail=f"Run record deleted, but some artifact files could not be removed: {error}") from error

        # Expert checkpoints may have been copied into the registry's model directory at activation.
        # Remove only an unreferenced, direct child for this deleted run's model version.
        for version in versions:
            artifact_path = Path(version["artifact_path"]).resolve()
            if artifact_path.parent == state.registry.model_root.resolve() and artifact_path.exists():
                shutil.rmtree(artifact_path)
        return {"deleted": True, "job_id": job_id}

    @app.get("/api/training/jobs/{job_id}/events")
    async def stream_training_events(job_id: str, request: FastAPIRequest,
                                     after_id: int = Query(default=0, ge=0)) -> StreamingResponse:
        state: ServiceState = request.app.state.runtime
        if state.training_store.get_job(job_id) is None:
            raise HTTPException(status_code=404, detail="Training run not found")

        async def event_stream():
            cursor = after_id
            terminal = {"built_evaluation_required", "built_evaluated", "failed", "cancelled"}
            while not await request.is_disconnected():
                events = await asyncio.to_thread(state.training_store.get_events, job_id, cursor)
                for event in events:
                    cursor = event["id"]
                    yield f"id: {cursor}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                job = await asyncio.to_thread(state.training_store.get_job, job_id)
                if job is None or (job["status"] in terminal and not events):
                    break
                if job["status"] in terminal and events:
                    continue
                await asyncio.sleep(0.8)

        return StreamingResponse(event_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/training/jobs/{job_id}/cancel")
    def cancel_training_job(job_id: str, request: FastAPIRequest) -> dict[str, Any]:
        state: ServiceState = request.app.state.runtime
        if state.training_store.get_job(job_id) is None:
            raise HTTPException(status_code=404, detail="Training run not found")
        try:
            cancelled = state.training_manager.cancel(job_id)
        except (OSError, RuntimeError) as error:
            raise HTTPException(status_code=500, detail=f"Could not stop the training worker: {type(error).__name__}") from error
        if not cancelled:
            raise HTTPException(status_code=409, detail="This run can no longer be cancelled")
        return {"accepted": True, "status": "cancelled", "job_id": job_id}

    @app.post("/api/training/jobs/{job_id}/resume", status_code=202)
    def resume_training_job(job_id: str, request: FastAPIRequest) -> dict[str, Any]:
        state: ServiceState = request.app.state.runtime
        try:
            job = state.training_store.resume_job(job_id)
            state.training_manager.enqueue(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Training run not found") from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"job": _public_job(job), "accepted": True}

    @app.get("/api/models")
    def list_model_versions(request: FastAPIRequest,
                            target: Literal["expert", "orchestrator"] | None = None) -> dict[str, Any]:
        state: ServiceState = request.app.state.runtime
        return {"models": state.training_store.list_model_versions(target)}

    @app.get("/api/models/catalog")
    def list_s3_model_catalog(request: FastAPIRequest,
                              workspace_id: str = Query(default="w1", max_length=200)) -> dict[str, Any]:
        state: ServiceState = request.app.state.runtime
        try:
            artifacts = state.artifact_store.list_model_artifacts()
        except Exception as error:
            raise HTTPException(
                status_code=503,
                detail=f"Could not read the configured S3 model catalog: {str(error)[:300]}",
            ) from error

        sync_started = _start_completed_artifact_sync(state)
        stored = {model["id"]: model for model in state.training_store.list_model_versions()}
        profiles = state.registry.list_profiles(workspace_id)
        active_orchestrator = state.training_store.get_active_model("orchestrator")
        models: list[dict[str, Any]] = []
        for artifact in artifacts:
            manifest = artifact.get("manifest") or {}
            if not isinstance(manifest, dict):
                continue
            version_id = str(artifact["id"])
            local = stored.get(version_id) or {}
            target = str(artifact["target"])
            profile = next((row for row in profiles
                            if str(row.get("trained_model_version_id") or "") == version_id
                            or Path(str(row.get("model_path") or "")).name == version_id), None)
            active = (
                bool(active_orchestrator and active_orchestrator.get("id") == version_id)
                if target == "orchestrator"
                else bool(profile and profile.get("status") == "Ready to Deploy")
            )
            local_evaluation = local.get("evaluation") or {}
            manifest_evaluation = manifest.get("evaluation") or {}
            evaluation = local_evaluation if _has_saved_inrun_evaluation(local_evaluation) else (
                manifest_evaluation if _has_saved_inrun_evaluation(manifest_evaluation)
                else local_evaluation or manifest_evaluation
            )
            if not isinstance(evaluation, dict):
                evaluation = {}
            ready_from_saved_evaluation = _has_saved_inrun_evaluation(evaluation)
            readiness = (
                "ready_to_deploy" if ready_from_saved_evaluation
                else local.get("readiness") or manifest.get("readiness") or "evaluation_required"
            )
            models.append({
                "id": version_id,
                "target": target,
                "model_name": local.get("model_name") or manifest.get("model_name") or version_id,
                "version": local.get("version") or manifest.get("version") or "1",
                "domain": local.get("domain") or manifest.get("domain") or ("Routing" if target == "orchestrator" else "Expert"),
                "approach": local.get("approach") or manifest.get("approach"),
                "architecture": local.get("architecture") or manifest.get("architecture"),
                "readiness": readiness,
                "artifact_uri": artifact.get("artifact_uri"),
                "last_modified": artifact.get("last_modified"),
                "evaluation": evaluation,
                "review_available": state.training_store.get_evaluation_for_model_version(version_id) is not None,
                "active": active,
                "expert_id": profile.get("expert_id") if profile else None,
                "can_activate": bool(active or readiness == "ready_to_deploy"),
            })
        return {"models": models, "storage": state.artifact_store.storage,
                "bucket": state.artifact_store.bucket, "prefix": state.artifact_store.prefix,
                "sync_pending": bool(sync_started or state.artifact_sync_pending),
                "sync_error": state.artifact_sync_error}

    @app.get("/api/models/local")
    def list_local_model_builds(request: FastAPIRequest) -> dict[str, Any]:
        """List completed local builds that have not been found in the S3 catalog."""
        state: ServiceState = request.app.state.runtime
        try:
            s3_ids = {str(item.get("id")) for item in state.artifact_store.list_model_artifacts()}
            catalog_error = None
        except Exception as error:
            s3_ids = set()
            catalog_error = str(error)[:300]

        models = []
        for item in state.training_store.list_model_versions():
            if str(item["id"]) in s3_ids:
                continue
            artifact_path = Path(str(item.get("artifact_path") or ""))
            evaluation = item.get("evaluation") or {}
            readiness = "ready_to_deploy" if _has_saved_inrun_evaluation(evaluation) else (
                item.get("readiness") or "evaluation_required"
            )
            models.append({
                "id": item["id"],
                "target": item["target"],
                "model_name": item["model_name"],
                "version": item["version"],
                "domain": item.get("domain"),
                "approach": item.get("approach"),
                "architecture": item.get("architecture"),
                "readiness": readiness,
                "evaluation": evaluation,
                "review_available": state.training_store.get_evaluation_for_model_version(str(item["id"])) is not None,
                "local_available": artifact_path.is_dir(),
                "can_upload": artifact_path.is_dir() and state.artifact_store.storage == "s3" and bool(state.artifact_store.bucket),
            })
        return {
            "models": models,
            "storage": state.artifact_store.storage,
            "bucket": state.artifact_store.bucket,
            "prefix": state.artifact_store.prefix,
            "catalog_error": catalog_error,
        }

    @app.post("/api/models/{model_version_id}/upload-to-s3")
    async def upload_model_version_to_s3(model_version_id: str, request: FastAPIRequest) -> dict[str, Any]:
        state: ServiceState = request.app.state.runtime
        version = state.training_store.get_model_version(model_version_id)
        if version is None:
            raise HTTPException(status_code=404, detail="Model version not found")
        if state.artifact_store.storage != "s3" or not state.artifact_store.bucket:
            raise HTTPException(status_code=409, detail="Configure S3 artifact storage in .env before uploading a model")
        try:
            existing = next((item for item in state.artifact_store.list_model_artifacts()
                             if str(item.get("id")) == model_version_id), None)
        except Exception as error:
            raise HTTPException(status_code=503, detail=f"Could not check the S3 catalog: {str(error)[:300]}") from error
        if existing:
            state.training_store.update_artifact_uri(model_version_id, str(existing["artifact_uri"]))
            return {"model_version_id": model_version_id, "artifact_uri": existing["artifact_uri"], "already_uploaded": True}

        model_path = Path(str(version.get("artifact_path") or ""))
        manifest_path = model_path / "mornork-model.json"
        if not model_path.is_dir() or not manifest_path.is_file():
            raise HTTPException(status_code=404, detail="The local model artifact or its manifest is missing")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.update({
                "id": version["id"],
                "version": version["version"],
                "model_name": version["model_name"],
                "target": version["target"],
                "domain": version.get("domain"),
                "approach": version.get("approach"),
                "architecture": version.get("architecture"),
                "evaluation": version.get("evaluation") or {},
                "readiness": version.get("readiness") or "evaluation_required",
            })
            artifact_uri = await asyncio.to_thread(
                state.artifact_store.upload_directory,
                model_path,
                str(version["target"]),
                model_version_id,
                manifest=manifest,
                force=True,
            )
            if not artifact_uri:
                raise RuntimeError("S3 upload did not return an artifact URI")
            state.training_store.update_artifact_uri(model_version_id, artifact_uri)
        except Exception as error:
            raise HTTPException(status_code=502, detail=f"Could not upload model artifacts to S3: {str(error)[:300]}") from error
        return {"model_version_id": model_version_id, "artifact_uri": artifact_uri, "already_uploaded": False}

    @app.get("/api/workflows")
    def list_workflows(request: FastAPIRequest, workspace_id: str = Query(default="w1", max_length=200),
                       published_only: bool = False) -> dict[str, Any]:
        state: ServiceState = request.app.state.runtime
        workflows = state.workflow_store.list_workflows(workspace_id)
        if published_only:
            workflows = [item for item in workflows if item["status"] == "published" and item["active_version_id"]]
        return {"workflows": workflows}

    @app.post("/api/workflows", status_code=201)
    def create_workflow(payload: WorkflowCreate, request: FastAPIRequest) -> dict[str, Any]:
        state: ServiceState = request.app.state.runtime
        return state.workflow_store.create_workflow(
            workspace_id=payload.workspace_id, name=payload.name, description=payload.description,
            config=payload.config.model_dump(),
        )

    @app.put("/api/workflows/{workflow_id}")
    def update_workflow(workflow_id: str, payload: WorkflowUpdate,
                        request: FastAPIRequest) -> dict[str, Any]:
        state: ServiceState = request.app.state.runtime
        try:
            return state.workflow_store.save_draft(
                workflow_id, name=payload.name, description=payload.description,
                config=payload.config.model_dump(),
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Workflow not found") from error

    @app.get("/api/workflows/runs")
    def list_all_workflow_runs(request: FastAPIRequest,
                               workspace_id: str = Query(default="w1", max_length=200)) -> dict[str, Any]:
        state: ServiceState = request.app.state.runtime
        return {"runs": state.workflow_store.list_all_runs(workspace_id)}

    @app.get("/api/workflows/{workflow_id}/runs")
    def list_workflow_runs(workflow_id: str, request: FastAPIRequest) -> dict[str, Any]:
        state: ServiceState = request.app.state.runtime
        if state.workflow_store.get_workflow(workflow_id) is None:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return {"runs": state.workflow_store.list_runs(workflow_id)}

    @app.post("/api/workflows/{workflow_id}/test", status_code=201)
    async def test_workflow(workflow_id: str, payload: WorkflowTestRequest,
                            request: FastAPIRequest) -> dict[str, Any]:
        state: ServiceState = request.app.state.runtime
        workflow = state.workflow_store.get_workflow(workflow_id)
        if workflow is None:
            raise HTTPException(status_code=404, detail="Workflow not found")
        if workflow["workspace_id"] != payload.workspace_id:
            raise HTTPException(status_code=422, detail="Test workspace must match the workflow workspace")
        draft = workflow.get("draft")
        if draft is None:
            raise HTTPException(status_code=409, detail="Create or save a draft version before testing")
        message_id = f"workflow-test-{uuid4().hex[:10]}"
        started = time.perf_counter()
        try:
            result = await asyncio.to_thread(
                state.execute_workflow, draft["config"], payload.workspace_id, message_id, payload.message,
            )
            status = "completed"
            trace = result["trace"]
        except Exception as error:
            status = "failed"
            result = {"decision": {"action": "error", "reason": type(error).__name__},
                      "trace": [{"node": "workflow", "status": "failed", "error": type(error).__name__}],
                      "answers": [], "final_answer": "Workflow execution failed. Check the local service log.",
                      "final_by": "Workflow"}
            trace = result["trace"]
        elapsed = round((time.perf_counter() - started) * 1000)
        run = state.workflow_store.record_run(
            workflow_id=workflow_id, version_id=draft["id"], message=payload.message,
            status=status, route_action=result["decision"].get("action", "error"),
            expert_ids=result.get("expert_ids", []), trace=trace, elapsed_ms=elapsed,
        )
        return {"run": run, "decision": result["decision"].to_dict() if hasattr(result["decision"], "to_dict") else result["decision"],
                "answers": result["answers"], "final_answer": result["final_answer"],
                "final_by": result["final_by"], "trace": trace}

    @app.post("/api/workflows/{workflow_id}/publish")
    def publish_workflow(workflow_id: str, request: FastAPIRequest) -> dict[str, Any]:
        state: ServiceState = request.app.state.runtime
        workflow = state.workflow_store.get_workflow(workflow_id)
        if workflow is None:
            raise HTTPException(status_code=404, detail="Workflow not found")
        draft = workflow.get("draft")
        if draft is None:
            raise HTTPException(status_code=409, detail="Save a draft before publishing")
        config = draft["config"]
        ready = state.routable_profiles_by_id(workflow["workspace_id"])
        selected = config.get("expert_ids") or []
        missing = [expert_id for expert_id in selected if expert_id not in ready]
        if missing:
            raise HTTPException(status_code=409, detail="Every configured expert must be Ready to Deploy before publishing: " + ", ".join(missing))
        try:
            published = state.workflow_store.publish(workflow_id)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"workflow": published, "published": True}

    @app.post("/api/workflows/{workflow_id}/deactivate")
    def deactivate_workflow(workflow_id: str, request: FastAPIRequest) -> dict[str, Any]:
        state: ServiceState = request.app.state.runtime
        try:
            return {"workflow": state.workflow_store.deactivate(workflow_id), "deactivated": True}
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Workflow not found") from error

    @app.get("/api/models/{model_version_id}/evaluation")
    def get_saved_model_evaluation(model_version_id: str, request: FastAPIRequest) -> dict[str, Any]:
        """Read the test outputs saved by the training run; never start model inference here."""
        state: ServiceState = request.app.state.runtime
        version = state.training_store.get_model_version(model_version_id)
        if version is None:
            raise HTTPException(status_code=404, detail="Model version not found")
        evaluation = state.training_store.get_evaluation_for_model_version(model_version_id)
        if evaluation is None:
            raise HTTPException(status_code=404, detail="This version has no saved in-run test examples. New training runs save the evaluation packet automatically.")
        return {"evaluation": evaluation, "threshold": 0.85,
                "review_required": evaluation.get("status") == "awaiting_review"}

    @app.post("/api/models/{model_version_id}/evaluation-preview")
    def reuse_saved_evaluation(model_version_id: str, request: FastAPIRequest) -> dict[str, Any]:
        """Compatibility alias: return saved training outputs without regenerating them."""
        state: ServiceState = request.app.state.runtime
        if state.training_store.get_model_version(model_version_id) is None:
            raise HTTPException(status_code=404, detail="Model version not found")
        evaluation = state.training_store.get_evaluation_for_model_version(model_version_id)
        if evaluation is None:
            raise HTTPException(status_code=409, detail="This version has no saved in-run test examples. New training runs capture them automatically.")
        return {"evaluation": evaluation, "threshold": 0.85,
                "review_required": evaluation.get("status") == "awaiting_review"}

    @app.post("/api/models/{model_version_id}/task-metrics")
    def reuse_saved_task_metrics(model_version_id: str, request: FastAPIRequest) -> dict[str, Any]:
        """Compatibility alias: never run a second model inference pass."""
        state: ServiceState = request.app.state.runtime
        version = state.training_store.get_model_version(model_version_id)
        if version is None:
            raise HTTPException(status_code=404, detail="Model version not found")
        metrics = (version.get("evaluation") or {}).get("task_metrics")
        if not metrics:
            raise HTTPException(status_code=409, detail="This version has no saved in-run task metrics. New training runs record evaluation automatically.")
        return {"model_version_id": model_version_id, "metrics": metrics,
                "already_evaluated": True}

    @app.post("/api/models/evaluations/{evaluation_id}/review")
    def review_model_evaluation(evaluation_id: str, payload: ModelEvaluationReview,
                                request: FastAPIRequest) -> dict[str, Any]:
        if not payload.qualified_reviewer_confirmed:
            raise HTTPException(status_code=422, detail="A qualified reviewer must confirm the evaluation before submission")
        state: ServiceState = request.app.state.runtime
        try:
            result = state.training_store.submit_evaluation_review(
                evaluation_id,
                reviewer=payload.reviewer,
                reviewer_role=payload.reviewer_role,
                ratings=payload.ratings,
                threshold=0.85,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Evaluation run not found") from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        version = state.training_store.get_model_version(result["model_version_id"])
        if version and version.get("artifact_uri"):
            manifest_path = Path(str(version.get("artifact_path") or "")) / "mornork-model.json"
            if manifest_path.is_file():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest["evaluation"] = version.get("evaluation") or {}
                    manifest["readiness"] = result["readiness"]
                    state.artifact_store.update_manifest(str(version["artifact_uri"]), manifest)
                    result["artifact_manifest_updated"] = True
                except Exception as error:
                    result["artifact_manifest_updated"] = False
                    result["artifact_manifest_warning"] = str(error)[:300]
        return result

    @app.post("/api/models/{model_version_id}/activate")
    def activate_model_version(model_version_id: str, request: FastAPIRequest,
                               workspace_id: str = Query(default="w1", max_length=200)) -> dict[str, Any]:
        state: ServiceState = request.app.state.runtime
        current = state.training_store.get_model_version(model_version_id)
        if current is None:
            raise HTTPException(status_code=404, detail="Model version not found")
        try:
            s3_artifact = next((item for item in state.artifact_store.list_model_artifacts()
                                if item.get("id") == model_version_id), None)
        except Exception as error:
            raise HTTPException(status_code=503, detail=f"Could not verify the model in S3 ({type(error).__name__})") from error
        if s3_artifact is None:
            raise HTTPException(status_code=409, detail="Only model artifacts currently present in S3 can be activated from this page")
        if current.get("artifact_uri") != s3_artifact.get("artifact_uri"):
            state.training_store.update_artifact_uri(model_version_id, str(s3_artifact["artifact_uri"]))
        saved_evaluation = current.get("evaluation") or (
            (s3_artifact.get("manifest") or {}).get("evaluation") or {}
        )
        if current.get("readiness") != "ready_to_deploy":
            if not _has_saved_inrun_evaluation(saved_evaluation):
                raise HTTPException(
                    status_code=409,
                    detail="This model has no saved held-out results from its training run and cannot be activated",
                )
            state.training_store.mark_model_ready_for_deployment(model_version_id)
        try:
            version = state.training_store.activate_model_version(model_version_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Model version not found") from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        manifest_warning = None
        try:
            manifest = dict(s3_artifact.get("manifest") or {})
            manifest.update({
                "id": version["id"],
                "target": version["target"],
                "model_name": version["model_name"],
                "version": version["version"],
                "evaluation": version.get("evaluation") or saved_evaluation,
                "readiness": "ready_to_deploy",
            })
            state.artifact_store.update_manifest(str(s3_artifact["artifact_uri"]), manifest)
        except Exception:
            manifest_warning = "Activation succeeded, but the S3 manifest could not be updated."
        if version["target"] == "orchestrator":
            state.set_active_orchestrator(None)
            return {"model": version, "activated": True, "artifact_manifest_warning": manifest_warning}

        job = state.training_store.get_job(version["job_id"])
        if job is None:
            raise HTTPException(status_code=404, detail="Source training run not found")
        try:
            model_path = state.artifact_store.materialize(version["artifact_path"], version.get("artifact_uri"))
        except Exception as error:
            raise HTTPException(status_code=503, detail=f"Could not load this model artifact: {type(error).__name__}") from error
        try:
            model_path.relative_to(state.registry.model_root)
        except ValueError:
            state.registry.model_root.mkdir(parents=True, exist_ok=True)
            destination = state.registry.model_root / version["id"]
            if destination.exists():
                raise HTTPException(status_code=409, detail="A checkpoint directory for this model version already exists")
            shutil.copytree(model_path, destination)
            model_path = destination
            state.training_store.update_artifact_path(version["id"], model_path)
            version = state.training_store.get_model_version(version["id"]) or version
        domain_profile = next((item for item in state.registry.list_profiles(workspace_id)
                               if not item.get("is_default")
                               and str(item.get("domain", "")).casefold() == str(job["domain"]).casefold()), None)
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{job['domain']}-{job['model_name']}").strip("-.").lower()
        expert_id = str(domain_profile["expert_id"]) if domain_profile else f"local-{slug}"[:118]
        profile = {
            "expert_id": expert_id,
            "name": job["model_name"],
            "domain": job["domain"],
            "description": job["description"] or f"Locally trained {job['domain']} expert model.",
            "workspace_id": workspace_id,
            "status": "Ready to Deploy",
            "version": version["version"],
            "routing_hints": job["config"].get("routing_hints", []),
            "model_path": str(model_path),
            "trained_model_version_id": version["id"],
            "artifact_uri": version.get("artifact_uri"),
            "inference_mode": "checkpoint",
            "is_default": False,
        }
        try:
            current = state.registry.get_profile(expert_id, workspace_id)
            if current:
                profile = state.registry.update_profile(expert_id, workspace_id, {key: value for key, value in profile.items() if key not in {"expert_id", "workspace_id", "is_default"}})
            else:
                profile = state.registry.create_profile(profile, workspace_id)
        except (PermissionError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {"model": version, "activated": True, "expert_profile": profile,
                "artifact_manifest_warning": manifest_warning}

    @app.post("/api/models/{model_version_id}/deactivate")
    def deactivate_model_version(model_version_id: str, request: FastAPIRequest,
                                 workspace_id: str = Query(default="w1", max_length=200)) -> dict[str, Any]:
        state: ServiceState = request.app.state.runtime
        version = state.training_store.get_model_version(model_version_id)
        if version is None:
            raise HTTPException(status_code=404, detail="Model version not found")
        if version["target"] == "orchestrator":
            state.training_store.deactivate_model_version(model_version_id)
            state.set_active_orchestrator(None)
            return {"model_version_id": model_version_id, "activated": False}
        profile = next((row for row in state.registry.list_profiles(workspace_id)
                        if str(row.get("trained_model_version_id") or "") == model_version_id), None)
        if profile is None:
            raise HTTPException(status_code=409, detail="This S3 expert is not active in this workspace")
        try:
            updated = state.registry.update_profile(profile["expert_id"], workspace_id, {"status": "Deactivated"})
        except (KeyError, PermissionError, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"model_version_id": model_version_id, "activated": False, "expert_profile": updated}

    @app.post("/mornork/chat")
    async def mornork_chat(payload: ChatRequest, request: FastAPIRequest) -> StreamingResponse:
        state: ServiceState = request.app.state.runtime
        message_id = payload.message_id or f"chat-{uuid4().hex[:12]}"
        if payload.workflow_id:
            workflow = state.workflow_store.get_workflow(payload.workflow_id)
            if workflow is None or workflow["workspace_id"] != payload.workspace_id:
                raise HTTPException(status_code=404, detail="Published workflow not found in this workspace")
            if workflow["status"] != "published" or not workflow["active_version_id"]:
                raise HTTPException(status_code=409, detail="This workflow has no active published version")
            version = state.workflow_store.get_version(workflow["active_version_id"])
            if version is None:
                raise HTTPException(status_code=409, detail="The published workflow version is unavailable")
            if payload.user_choice:
                raise HTTPException(status_code=400, detail="Published workflows choose experts according to their saved routing configuration")
            workflow_result = await asyncio.to_thread(
                state.execute_workflow, version["config"], payload.workspace_id, message_id, payload.message,
            )
            state.workflow_store.record_run(
                workflow_id=workflow["id"], version_id=version["id"], message=payload.message,
                status="completed", route_action=workflow_result["decision"].action.value,
                expert_ids=workflow_result["expert_ids"], trace=workflow_result["trace"],
                elapsed_ms=next((item["elapsed_ms"] for item in reversed(workflow_result["trace"])
                                 if item.get("node") == "total"), 0),
            )

            async def workflow_stream():
                yield _sse("routing", {"decision": workflow_result["decision"].to_dict()})
                for answer in workflow_result["answers"]:
                    if answer.get("text"):
                        yield _sse("answer", {key: answer[key] for key in ("name", "domain", "expert_id", "version", "prototype", "text") if key in answer} | {"by": answer["name"]})
                    else:
                        yield _sse("expert_unavailable", {"expert_id": answer["expert_id"], "name": answer["name"]})
                yield _sse("workflow_trace", {"workflow": workflow["name"], "version": version["version"],
                                               "trace": workflow_result["trace"]})
                yield _sse("workflow_response", {"by": workflow_result["final_by"],
                                                  "version": version["version"],
                                                  "text": workflow_result["final_answer"]})

            return StreamingResponse(workflow_stream(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        decision = await asyncio.to_thread(state.route, payload.workspace_id, message_id, payload.message)
        chosen_ids: list[str] = []
        if payload.user_choice:
            if decision.action == RouteAction.ASK_USER:
                if payload.user_choice == "both":
                    chosen_ids = list(decision.candidate_expert_ids)
                elif payload.user_choice in decision.candidate_expert_ids:
                    chosen_ids = [payload.user_choice]
                else:
                    raise HTTPException(status_code=400, detail="Choose one of the experts offered for this message")
            elif decision.action == RouteAction.ANSWER_AND_OFFER and payload.user_choice == decision.offer_expert_id:
                chosen_ids = [payload.user_choice]
            else:
                raise HTTPException(status_code=400, detail="There is no pending expert offer for this message")
            available = state.routable_profiles_by_id(payload.workspace_id)
            if any(expert_id not in available for expert_id in chosen_ids):
                raise HTTPException(status_code=409, detail="An expert is no longer Ready to Deploy")

        async def stream():
            try:
                yield _sse("routing", {"decision": decision.to_dict()})
                if payload.user_choice:
                    async for event in _handoff_events(state, payload, message_id, chosen_ids):
                        yield event
                    return
                if decision.action == RouteAction.ANSWER_SELF:
                    answer = await asyncio.to_thread(state.chat_model.generate, payload.message)
                    yield _sse("answer", {"by": "Mornork", "version": APP_VERSION, "text": answer})
                elif decision.action == RouteAction.ANSWER_AND_OFFER:
                    answer = await asyncio.to_thread(state.chat_model.generate, payload.message)
                    yield _sse("answer", {"by": "Mornork", "version": APP_VERSION, "text": answer})
                    expert = state.routable_profiles_by_id(payload.workspace_id).get(decision.offer_expert_id or "")
                    if expert:
                        yield _sse("offer", {"expert_id": expert.expert_id, "name": expert.name, "domain": expert.domain, "version": expert.version})
                elif decision.action == RouteAction.HANDOFF:
                    async for event in _handoff_events(state, payload, message_id, list(decision.chosen_expert_ids)):
                        yield event
                elif decision.action == RouteAction.ASK_USER:
                    routable = state.routable_profiles_by_id(payload.workspace_id)
                    choices = [
                        {"expert_id": expert_id, "name": routable[expert_id].name}
                        for expert_id in decision.candidate_expert_ids
                        if expert_id in routable
                    ]
                    yield _sse("ask_user", {"options": choices, "allow_both": "both" in decision.user_options})
                elif decision.action == RouteAction.DECLINE_CREATE_EXPERT:
                    domain = decision.domain_scores[0].domain if decision.domain_scores else "this domain"
                    yield _sse("decline", {
                        "domain": domain,
                        "text": f"I can't answer {domain} questions reliably because no Ready to Deploy expert is available. You can add one in the Experts module.",
                        "create_expert_url": f"/models?workspace_id={payload.workspace_id}&domain={domain}",
                    })
            except Exception as error:
                print(f"Mornork chat failed ({type(error).__name__}); message content omitted.")
                yield _sse("error", {"message": "Mornork could not complete this request. Check the service log."})

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/experts")
    def list_experts(request: FastAPIRequest, workspace_id: str = Query(default="w1", max_length=200)) -> dict[str, Any]:
        state: ServiceState = request.app.state.runtime
        return {"experts": state.registry.list_profiles(workspace_id), "domains": list(STANDARD_DOMAINS)}

    @app.get("/api/experts/{expert_id}")
    def get_expert(expert_id: str, request: FastAPIRequest, workspace_id: str = Query(default="w1", max_length=200)) -> dict[str, Any]:
        state: ServiceState = request.app.state.runtime
        profile = state.registry.get_profile(expert_id, workspace_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Expert not found in this workspace")
        return profile

    @app.post("/api/experts", status_code=201)
    def create_expert(payload: ExpertCreate, request: FastAPIRequest) -> dict[str, Any]:
        state: ServiceState = request.app.state.runtime
        try:
            return state.registry.create_profile(payload.model_dump(), payload.workspace_id)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.patch("/api/experts/{expert_id}")
    def update_expert(expert_id: str, payload: ExpertUpdate, request: FastAPIRequest, workspace_id: str = Query(default="w1", max_length=200)) -> dict[str, Any]:
        state: ServiceState = request.app.state.runtime
        changes = payload.model_dump(exclude_unset=True)
        try:
            return state.registry.update_profile(expert_id, workspace_id, changes)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Expert not found in this workspace") from error
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.patch("/api/experts/{expert_id}/status")
    def update_expert_status(expert_id: str, payload: StatusUpdate, request: FastAPIRequest) -> dict[str, Any]:
        state: ServiceState = request.app.state.runtime
        try:
            return state.registry.update_profile(expert_id, payload.workspace_id, {"status": payload.status})
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Expert not found in this workspace") from error
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/api/experts/infer")
    async def infer_expert(
        payload: ExpertInferenceRequest,
        request: FastAPIRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        state: ServiceState = request.app.state.runtime
        _require_expert_key(state, authorization)
        try:
            answer = await run_in_threadpool(state.registry.infer, payload.model_dump())
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Expert not found in this workspace") from error
        except PermissionError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except (URLError, HTTPError, TimeoutError, OSError, ValueError) as error:
            raise HTTPException(status_code=502, detail=f"Expert inference failed: {type(error).__name__}") from error
        return {"expert_id": payload.expert_id, "expert_version": payload.expert_version, "answer": answer}

    return app


async def _handoff_events(
    state: ServiceState,
    payload: ChatRequest,
    message_id: str,
    expert_ids: list[str],
):
    profiles = state.routable_profiles_by_id(payload.workspace_id)
    selected = [profiles[expert_id] for expert_id in expert_ids if expert_id in profiles]
    missing = [expert_id for expert_id in expert_ids if expert_id not in profiles]
    for expert in selected:
        yield _sse("handoff", {"expert_id": expert.expert_id, "name": expert.name, "domain": expert.domain, "version": expert.version, "prototype": expert.inference_mode == "mornork_role"})

    request_base = {
        "workspace_id": payload.workspace_id,
        "message_id": message_id,
        "message": payload.message,
    }

    def call(expert: ExpertProfile) -> dict[str, Any]:
        endpoint = (expert.inference_url or state.settings.expert_inference_url).strip()
        body = json.dumps({
            **request_base,
            "expert_id": expert.expert_id,
            "expert_version": expert.version,
        }, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if state.settings.expert_api_key:
            headers["Authorization"] = f"Bearer {state.settings.expert_api_key}"
        request = Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=state.settings.expert_timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
            answer = result.get("answer") if isinstance(result, dict) else None
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Expert API returned no answer")
            return {"profile": expert, "answer": answer.strip()}
        except Exception as error:
            return {"profile": expert, "error": type(error).__name__}

    results = await asyncio.gather(*(asyncio.to_thread(call, expert) for expert in selected))
    successes = [item for item in results if "answer" in item]
    failures = [item for item in results if "answer" not in item]
    for expert_id in missing:
        yield _sse("expert_unavailable", {"expert_id": expert_id, "name": expert_id})
    if failures:
        for item in failures:
            expert = item["profile"]
            yield _sse("expert_unavailable", {"expert_id": expert.expert_id, "name": expert.name, "domain": expert.domain})
    for item in successes:
        expert = item["profile"]
        yield _sse("answer", {
            "by": expert.name,
            "domain": expert.domain,
            "expert_id": expert.expert_id,
            "version": expert.version,
            "prototype": expert.inference_mode == "mornork_role",
            "text": item["answer"],
        })
    if failures or missing:
        fallback = await asyncio.to_thread(state.chat_model.generate, payload.message)
        yield _sse("answer", {"by": "Mornork fallback", "version": APP_VERSION, "text": fallback})

    try:
        state.decision_log.write(RoutingDecision(
            workspace_id=payload.workspace_id,
            message_id=message_id,
            action=RouteAction.HANDOFF,
            reason="expert_handoff_partial" if (failures or missing) and successes else ("expert_handoff_unavailable" if failures or missing else "expert_handoff_completed"),
            chosen_expert_ids=tuple(expert_ids),
            user_choice=payload.user_choice or (selected[0].expert_id if len(selected) == 1 else ",".join(expert.expert_id for expert in selected)),
            mornork_version=APP_VERSION,
        ))
    except Exception as error:
        print(f"Handoff audit write failed ({type(error).__name__}); message content omitted.")


app = create_app()


def main() -> None:
    settings = Settings.from_environment()
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()




