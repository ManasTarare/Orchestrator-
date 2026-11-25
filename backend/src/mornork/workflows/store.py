"""SQLite persistence for workflow drafts, immutable published versions, and run metadata."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class WorkflowStore:
    """Stores workflow metadata beside training metadata in the local SQLite database."""

    def __init__(self, database: str | Path) -> None:
        self.database = Path(database).resolve()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._lock, self._connect() as db:
            db.execute("PRAGMA journal_mode = WAL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS workflows (
                    id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL, name TEXT NOT NULL,
                    description TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'draft',
                    active_version_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS workflows_workspace_idx ON workflows(workspace_id, updated_at DESC);
                CREATE TABLE IF NOT EXISTS workflow_versions (
                    id TEXT PRIMARY KEY, workflow_id TEXT NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
                    version INTEGER NOT NULL, config_json TEXT NOT NULL, status TEXT NOT NULL,
                    created_at TEXT NOT NULL, published_at TEXT,
                    UNIQUE(workflow_id, version)
                );
                CREATE INDEX IF NOT EXISTS workflow_versions_idx ON workflow_versions(workflow_id, version DESC);
                CREATE TABLE IF NOT EXISTS workflow_runs (
                    id TEXT PRIMARY KEY, workflow_id TEXT NOT NULL REFERENCES workflows(id),
                    version_id TEXT NOT NULL REFERENCES workflow_versions(id), status TEXT NOT NULL,
                    message_hash TEXT NOT NULL, route_action TEXT NOT NULL,
                    expert_ids_json TEXT NOT NULL, trace_json TEXT NOT NULL,
                    elapsed_ms INTEGER NOT NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS workflow_runs_idx ON workflow_runs(workflow_id, created_at DESC);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA busy_timeout = 30000")
        return db

    @staticmethod
    def _version(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        item = dict(row)
        item["config"] = json.loads(item.pop("config_json"))
        return item

    def _workflow(self, db: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        versions = db.execute(
            "SELECT * FROM workflow_versions WHERE workflow_id=? ORDER BY version DESC", (item["id"],)
        ).fetchall()
        item["versions"] = [self._version(version) for version in versions]
        item["draft"] = next((version for version in item["versions"] if version["status"] == "draft"), None)
        return item

    def list_workflows(self, workspace_id: str = "w1") -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM workflows WHERE workspace_id=? ORDER BY updated_at DESC", (workspace_id,)).fetchall()
            return [self._workflow(db, row) for row in rows]

    def get_workflow(self, workflow_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM workflows WHERE id=?", (workflow_id,)).fetchone()
            return self._workflow(db, row) if row else None

    def get_version(self, version_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM workflow_versions WHERE id=?", (version_id,)).fetchone()
        return self._version(row) if row else None

    def create_workflow(self, *, workspace_id: str, name: str, description: str,
                        config: dict[str, Any]) -> dict[str, Any]:
        workflow_id = f"wf-{uuid4().hex[:12]}"
        version_id = f"wfv-{uuid4().hex[:12]}"
        now = _now()
        with self._connect() as db:
            db.execute(
                "INSERT INTO workflows(id,workspace_id,name,description,status,created_at,updated_at) VALUES (?,?,?,?,?,?,?)",
                (workflow_id, workspace_id, name.strip(), description.strip(), "draft", now, now),
            )
            db.execute(
                "INSERT INTO workflow_versions(id,workflow_id,version,config_json,status,created_at) VALUES (?,?,?,?,?,?)",
                (version_id, workflow_id, 1, json.dumps(config, ensure_ascii=False), "draft", now),
            )
        return self.get_workflow(workflow_id)  # type: ignore[return-value]

    def save_draft(self, workflow_id: str, *, name: str, description: str,
                   config: dict[str, Any]) -> dict[str, Any]:
        now = _now()
        with self._connect() as db:
            workflow = db.execute("SELECT * FROM workflows WHERE id=?", (workflow_id,)).fetchone()
            if workflow is None:
                raise KeyError(workflow_id)
            draft = db.execute(
                "SELECT * FROM workflow_versions WHERE workflow_id=? AND status='draft' ORDER BY version DESC LIMIT 1",
                (workflow_id,),
            ).fetchone()
            if draft is None:
                latest = db.execute("SELECT COALESCE(MAX(version),0) AS n FROM workflow_versions WHERE workflow_id=?", (workflow_id,)).fetchone()
                db.execute(
                    "INSERT INTO workflow_versions(id,workflow_id,version,config_json,status,created_at) VALUES (?,?,?,?,?,?)",
                    (f"wfv-{uuid4().hex[:12]}", workflow_id, int(latest["n"]) + 1,
                     json.dumps(config, ensure_ascii=False), "draft", now),
                )
            else:
                db.execute("UPDATE workflow_versions SET config_json=? WHERE id=?",
                           (json.dumps(config, ensure_ascii=False), draft["id"]))
                # Any edit makes previous execution evidence stale for this draft.
                db.execute("DELETE FROM workflow_runs WHERE version_id=?", (draft["id"],))
            db.execute(
                "UPDATE workflows SET name=?,description=?,status=CASE WHEN active_version_id IS NULL THEN 'draft' ELSE 'published' END,updated_at=? WHERE id=?",
                       (name.strip(), description.strip(), now, workflow_id))
        return self.get_workflow(workflow_id)  # type: ignore[return-value]

    def publish(self, workflow_id: str) -> dict[str, Any]:
        now = _now()
        with self._connect() as db:
            workflow = db.execute("SELECT * FROM workflows WHERE id=?", (workflow_id,)).fetchone()
            if workflow is None:
                raise KeyError(workflow_id)
            draft = db.execute(
                "SELECT * FROM workflow_versions WHERE workflow_id=? AND status='draft' ORDER BY version DESC LIMIT 1",
                (workflow_id,),
            ).fetchone()
            if draft is None:
                raise ValueError("Save a workflow draft before publishing")
            test = db.execute(
                "SELECT status FROM workflow_runs WHERE workflow_id=? AND version_id=? ORDER BY created_at DESC LIMIT 1",
                (workflow_id, draft["id"]),
            ).fetchone()
            if test is None or test["status"] != "completed":
                raise ValueError("Run a successful workflow test on this draft before publishing")
            db.execute("UPDATE workflow_versions SET status='published',published_at=? WHERE id=?", (now, draft["id"]))
            db.execute("UPDATE workflows SET status='published',active_version_id=?,updated_at=? WHERE id=?",
                       (draft["id"], now, workflow_id))
        return self.get_workflow(workflow_id)  # type: ignore[return-value]

    def deactivate(self, workflow_id: str) -> dict[str, Any]:
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE workflows SET status='inactive',active_version_id=NULL,updated_at=? WHERE id=?",
                (_now(), workflow_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(workflow_id)
        return self.get_workflow(workflow_id)  # type: ignore[return-value]

    def record_run(self, *, workflow_id: str, version_id: str, message: str,
                   status: str, route_action: str, expert_ids: list[str],
                   trace: list[dict[str, Any]], elapsed_ms: int) -> dict[str, Any]:
        run = {
            "id": f"wrun-{uuid4().hex[:12]}", "workflow_id": workflow_id, "version_id": version_id,
            "status": status, "message_hash": hashlib.sha256(message.encode("utf-8")).hexdigest(),
            "route_action": route_action, "expert_ids": expert_ids, "trace": trace,
            "elapsed_ms": elapsed_ms, "created_at": _now(),
        }
        with self._connect() as db:
            db.execute(
                "INSERT INTO workflow_runs VALUES (?,?,?,?,?,?,?,?,?,?)",
                (run["id"], workflow_id, version_id, status, run["message_hash"], route_action,
                 json.dumps(expert_ids), json.dumps(trace, ensure_ascii=False), elapsed_ms, run["created_at"]),
            )
        return run

    def list_runs(self, workflow_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM workflow_runs WHERE workflow_id=? ORDER BY created_at DESC LIMIT ?",
                              (workflow_id, limit)).fetchall()
        return [{**dict(row), "expert_ids": json.loads(row["expert_ids_json"]),
                 "trace": json.loads(row["trace_json"])} for row in rows]

    def list_all_runs(self, workspace_id: str = "w1", limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT r.*,w.name AS workflow_name FROM workflow_runs r "
                "JOIN workflows w ON w.id=r.workflow_id WHERE w.workspace_id=? "
                "ORDER BY r.created_at DESC LIMIT ?", (workspace_id, limit),
            ).fetchall()
        return [{**dict(row), "expert_ids": json.loads(row["expert_ids_json"]),
                 "trace": json.loads(row["trace_json"])} for row in rows]
