"""Workspace-scoped expert profiles and lazy inference backends."""

from __future__ import annotations

import gc
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import torch

from ..knowledge import (
    LocalKnowledgeBase,
    build_grounding_prompt,
    has_wrong_jurisdiction_tax_terms,
    source_line,
)
from ..modeling.generation import IncompleteGenerationError, LocalChatModel
from ..routing.router import STANDARD_DOMAINS
from ..routing.types import ExpertProfile

VALID_STATUSES = {"Not Configured", "Needs improvement", "Ready to Deploy", "Deactivated", "Deleted"}


class ExpertRegistry:
    """Read and update expert metadata, and execute configured model backends."""

    def __init__(
        self,
        profile_path: str | Path,
        model_root: str | Path,
        *,
        device: str = "cpu",
        dtype: str = "bfloat16",
        max_new_tokens: int = 256,
        backend_api_key: str | None = None,
        audit_path: str | Path | None = None,
        shared_model: LocalChatModel | None = None,
        knowledge_root: str | Path | None = None,
    ) -> None:
        self.profile_path = Path(profile_path).resolve()
        self.model_root = Path(model_root).resolve()
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.backend_api_key = backend_api_key
        self.audit_path = Path(audit_path).resolve() if audit_path else None
        self.shared_model = shared_model
        root = knowledge_root or (self.profile_path.parent.parent / "data" / "knowledge")
        self.knowledge = LocalKnowledgeBase(root)
        self._file_lock = threading.RLock()
        self._inference_lock = threading.Lock()
        self._loaded_path: Path | None = None
        self._loaded_model: LocalChatModel | None = None
        self._loaded_model_lock = threading.Lock()
        self._audit_lock = threading.Lock()

    def _read_rows(self) -> list[dict[str, Any]]:
        with self._file_lock:
            data = json.loads(self.profile_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("experts")
        if not isinstance(data, list):
            raise ValueError("Experts profile file must contain a JSON list or an experts list")
        rows = []
        for row in data:
            if not isinstance(row, dict):
                raise ValueError("Each expert profile must be an object")
            rows.append(dict(row))
        return rows

    @staticmethod
    def _belongs_to_workspace(row: dict[str, Any], workspace_id: str) -> bool:
        return row.get("workspace_id") in {None, "", workspace_id}

    def list_profiles(self, workspace_id: str) -> list[dict[str, Any]]:
        return [row for row in self._read_rows() if self._belongs_to_workspace(row, workspace_id)]

    def list_routable_profiles(self, workspace_id: str) -> list[ExpertProfile]:
        profiles = []
        for row in self.list_profiles(workspace_id):
            profile = self._profile(row)
            if profile.is_ready and not profile.is_default and self.has_backend(row):
                profiles.append(profile)
        return profiles

    def get_profile(self, expert_id: str, workspace_id: str) -> dict[str, Any] | None:
        return next(
            (row for row in self.list_profiles(workspace_id) if row.get("expert_id") == expert_id),
            None,
        )

    def create_profile(self, row: dict[str, Any], workspace_id: str) -> dict[str, Any]:
        candidate = dict(row)
        candidate["workspace_id"] = workspace_id
        candidate.setdefault("status", "Not Configured")
        candidate.setdefault("version", 1)
        candidate.setdefault("routing_hints", [])
        candidate.setdefault("is_default", False)
        if candidate["is_default"]:
            raise ValueError("The Mornork default record is managed by the service")
        self._validate_profile(candidate)
        with self._file_lock:
            rows = self._read_rows()
            if any(item.get("expert_id") == candidate["expert_id"] and self._belongs_to_workspace(item, workspace_id) for item in rows):
                raise ValueError("An expert with this ID already exists in the workspace")
            rows.append(candidate)
            self._write_rows(rows)
            self._write_audit("expert_profile_created", candidate, old_status=None)
        return candidate

    def update_profile(self, expert_id: str, workspace_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "name", "domain", "description", "status", "version", "routing_hints",
            "inference_url", "model_path", "backend_url", "system_prompt", "is_default",
            "trained_model_version_id", "artifact_uri",
        }
        allowed.add("inference_mode")
        if set(changes) - allowed:
            raise ValueError("Profile update contains unsupported fields")
        with self._file_lock:
            rows = self._read_rows()
            index = next((i for i, row in enumerate(rows) if row.get("expert_id") == expert_id and row.get("workspace_id") == workspace_id), None)
            if index is None:
                raise KeyError(expert_id)
            previous = dict(rows[index])
            if previous.get("is_default"):
                raise PermissionError("The Mornork default record is managed by the service")
            updated = dict(previous)
            updated.update(changes)
            updated["workspace_id"] = workspace_id
            self._validate_profile(updated)
            rows[index] = updated
            self._write_rows(rows)
            self._write_audit(
                "expert_status_changed" if updated.get("status") != previous.get("status") else "expert_profile_updated",
                updated,
                old_status=previous.get("status"),
            )
        return updated

    def has_backend(self, row: dict[str, Any]) -> bool:
        if row.get("inference_mode") == "mornork_role":
            return self.shared_model is not None and bool(str(row.get("system_prompt") or "").strip())
        if isinstance(row.get("backend_url"), str) and row["backend_url"].strip():
            return True
        if not isinstance(row.get("model_path"), str) or not row["model_path"].strip():
            return False
        try:
            path = self._resolve_model_path(row["model_path"])
        except ValueError:
            return False
        return path.is_dir()

    def infer(self, request_data: dict[str, Any]) -> str:
        workspace_id = str(request_data["workspace_id"])
        expert_id = str(request_data["expert_id"])
        message = str(request_data["message"])
        row = self.get_profile(expert_id, workspace_id)
        if row is None:
            raise KeyError(expert_id)
        profile = self._profile(row)
        if not profile.is_ready or profile.is_default:
            raise PermissionError("Expert is not Ready to Deploy")
        if request_data.get("expert_version") is not None and str(request_data["expert_version"]) != str(profile.version):
            raise PermissionError("Expert version changed; route the message again")
        if not self.has_backend(row):
            raise FileNotFoundError("Expert backend is not configured")

        documents = self.knowledge.retrieve(profile.domain, message)
        if row.get("inference_mode") == "mornork_role":
            if self.shared_model is None:
                raise FileNotFoundError("The shared Mornork model is not loaded")
            system_prompt = build_grounding_prompt(
                row.get("system_prompt"),
                profile.domain,
                message,
                documents,
                compact=bool(
                    self.shared_model.context_limit
                    and self.shared_model.context_limit <= 256
                ),
            )
            answer = self._generate_with_reference_fallback(
                self.shared_model, message, system_prompt, documents
            )
        else:
            backend_url = str(row.get("backend_url") or "").strip()
            if backend_url:
                answer = self._call_backend(backend_url, request_data)
            else:
                model_path = self._resolve_model_path(str(row["model_path"]))
                with self._inference_lock:
                    model = self._load_model(model_path)
                    system_prompt = build_grounding_prompt(
                        row.get("system_prompt"),
                        profile.domain,
                        message,
                        documents,
                        compact=bool(model.context_limit and model.context_limit <= 256),
                    )
                    answer = self._generate_with_reference_fallback(
                        model, message, system_prompt, documents
                    )

        if has_wrong_jurisdiction_tax_terms(profile.domain, message, answer):
            withheld = (
                "I withheld this response because it included U.S.-specific tax references "
                "for an India-specific question. I can’t verify those claims as Indian tax "
                "guidance. Please use the official Income Tax Department references or "
                "consult a qualified Indian tax professional."
            )
            sources = source_line(documents)
            return f"{withheld}\n\n{sources}" if sources else withheld

        sources = source_line(documents)
        if sources and not any(url in answer for document in documents for url in document.source_urls):
            answer = f"{answer.rstrip()}\n\n{sources}"
        return answer

    @staticmethod
    def _generate_with_reference_fallback(
        model: LocalChatModel,
        message: str,
        system_prompt: str,
        documents: list[Any],
    ) -> str:
        try:
            return model.generate(message, system_prompt=system_prompt)
        except IncompleteGenerationError:
            for document in documents:
                if document.fallback_answer:
                    return (
                        "The local model output was incomplete. Based on the retrieved reference: "
                        + document.fallback_answer
                    )
            raise

    def _load_model(self, model_path: Path) -> LocalChatModel:
        with self._loaded_model_lock:
            if self._loaded_path == model_path and self._loaded_model is not None:
                return self._loaded_model
            self._loaded_model = None
            self._loaded_path = None
            gc.collect()
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = LocalChatModel(
                model_path,
                device=self.device,
                dtype=self.dtype,
                max_new_tokens=self.max_new_tokens,
            )
            self._loaded_path = model_path
            self._loaded_model = model
            return model

    def _call_backend(self, url: str, request_data: dict[str, Any]) -> str:
        payload = json.dumps(request_data, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.backend_api_key:
            headers["Authorization"] = f"Bearer {self.backend_api_key}"
        request = Request(url, data=payload, headers=headers, method="POST")
        with urlopen(request, timeout=45) as response:
            result = json.loads(response.read().decode("utf-8"))
        answer = result.get("answer") if isinstance(result, dict) else None
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Expert backend must return a non-empty answer")
        return answer.strip()

    def _resolve_model_path(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = (self.model_root / path).resolve()
        else:
            path = path.resolve()
        try:
            path.relative_to(self.model_root)
        except ValueError as error:
            raise ValueError("Expert checkpoints must be stored under the configured expert model directory") from error
        return path

    @staticmethod
    def _profile(row: dict[str, Any]) -> ExpertProfile:
        values = {key: value for key, value in row.items() if key in ExpertProfile.__dataclass_fields__}
        values["routing_hints"] = tuple(values.get("routing_hints") or ())
        return ExpertProfile(**values)

    def _validate_profile(self, row: dict[str, Any]) -> None:
        required = ("expert_id", "name", "domain", "description", "status")
        if any(not isinstance(row.get(key), str) or not row[key].strip() for key in required):
            raise ValueError("Expert ID, name, domain, description, and status are required")
        is_default_record = row.get("is_default") is True
        if row["domain"].casefold() not in {domain.casefold() for domain in STANDARD_DOMAINS} and not (is_default_record and row["domain"].casefold() == "general"):
            raise ValueError("Domain must be one of the standard Mornork domains")
        if row["status"] not in VALID_STATUSES:
            raise ValueError("Status must be one of the supported Experts module statuses")
        if row.get("inference_mode", "checkpoint") not in {"checkpoint", "mornork_role"}:
            raise ValueError("inference_mode must be checkpoint or mornork_role")
        hints = row.get("routing_hints", [])
        if not isinstance(hints, (list, tuple)) or not all(isinstance(hint, str) for hint in hints):
            raise ValueError("routing_hints must be a list of strings")
        if row["status"] == "Ready to Deploy" and not is_default_record and not self.has_backend(row):
            raise ValueError("An expert needs a valid checkpoint or backend URL before Ready to Deploy")
        if row.get("model_path"):
            self._resolve_model_path(str(row["model_path"]))

    def _write_rows(self, rows: list[dict[str, Any]]) -> None:
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.profile_path.with_suffix(self.profile_path.suffix + ".tmp")
        temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.profile_path)

    def _write_audit(self, event: str, profile: dict[str, Any], *, old_status: str | None) -> None:
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "event": event,
            "workspace_id": profile.get("workspace_id"),
            "expert_id": profile.get("expert_id"),
            "old_status": old_status,
            "new_status": profile.get("status"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._audit_lock, self.audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
