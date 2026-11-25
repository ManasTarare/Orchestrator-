"""SQLite-backed local storage for datasets, training jobs, events, and versions."""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrainingStore:
    """Keep metadata in SQLite and uploaded datasets/checkpoints in local folders."""

    def __init__(self, database: str | Path, dataset_root: str | Path, artifact_root: str | Path) -> None:
        self.database = Path(database).resolve()
        self.dataset_root = Path(dataset_root).resolve()
        self.artifact_root = Path(artifact_root).resolve()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._schema_lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._schema_lock, self._connect() as db:
            db.execute("PRAGMA journal_mode = WAL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, domain TEXT NOT NULL,
                    file_name TEXT NOT NULL, file_path TEXT NOT NULL, file_format TEXT NOT NULL,
                    row_count INTEGER NOT NULL, byte_size INTEGER NOT NULL, schema_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS training_jobs (
                    id TEXT PRIMARY KEY,
                    target TEXT NOT NULL CHECK (target IN ('expert', 'orchestrator')),
                    model_name TEXT NOT NULL, domain TEXT NOT NULL, description TEXT NOT NULL,
                    approach TEXT NOT NULL CHECK (approach IN ('scratch', 'fine_tune')),
                    architecture TEXT NOT NULL, base_model TEXT,
                    dataset_id TEXT NOT NULL REFERENCES datasets(id),
                    eval_dataset_id TEXT REFERENCES datasets(id), config_json TEXT NOT NULL,
                    status TEXT NOT NULL, stage TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0,
                    cancel_requested INTEGER NOT NULL DEFAULT 0, worker_pid INTEGER,
                    output_dir TEXT NOT NULL, error_message TEXT,
                    created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT
                );
                CREATE INDEX IF NOT EXISTS training_jobs_created_idx ON training_jobs(created_at DESC);
                CREATE INDEX IF NOT EXISTS training_jobs_status_idx ON training_jobs(status);
                CREATE TABLE IF NOT EXISTS training_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL REFERENCES training_jobs(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL, level TEXT NOT NULL, stage TEXT NOT NULL,
                    progress REAL, message TEXT NOT NULL, metrics_json TEXT
                );
                CREATE INDEX IF NOT EXISTS training_events_job_idx ON training_events(job_id, id);
                CREATE TABLE IF NOT EXISTS model_versions (
                    id TEXT PRIMARY KEY, job_id TEXT NOT NULL UNIQUE REFERENCES training_jobs(id),
                    target TEXT NOT NULL, model_name TEXT NOT NULL, version TEXT NOT NULL,
                    approach TEXT NOT NULL, architecture TEXT NOT NULL, artifact_path TEXT NOT NULL,
                    artifact_uri TEXT, evaluation_json TEXT, readiness TEXT NOT NULL DEFAULT 'evaluation_required',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_evaluations (
                    id TEXT PRIMARY KEY, model_version_id TEXT NOT NULL REFERENCES model_versions(id),
                    cases_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'awaiting_review',
                    reviewer TEXT, reviewer_role TEXT, score REAL, pass_count INTEGER, case_count INTEGER,
                    created_at TEXT NOT NULL, reviewed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS active_models (
                    target TEXT PRIMARY KEY CHECK (target IN ('expert', 'orchestrator')),
                    model_version_id TEXT NOT NULL REFERENCES model_versions(id),
                    activated_at TEXT NOT NULL
                );
                """
            )
            job_columns = {row["name"] for row in db.execute("PRAGMA table_info(training_jobs)")}
            if "kept" not in job_columns:
                db.execute("ALTER TABLE training_jobs ADD COLUMN kept INTEGER NOT NULL DEFAULT 0")
            version_columns = {row["name"] for row in db.execute("PRAGMA table_info(model_versions)")}
            if "artifact_uri" not in version_columns:
                db.execute("ALTER TABLE model_versions ADD COLUMN artifact_uri TEXT")

    def add_dataset(self, *, name: str, domain: str, file_name: str, file_path: str | Path,
                    file_format: str, row_count: int, byte_size: int, columns: list[str]) -> dict[str, Any]:
        dataset_id = f"ds-{uuid4().hex[:12]}"
        created_at = _now()
        resolved_path = str(Path(file_path).resolve())
        with self._connect() as db:
            db.execute(
                "INSERT INTO datasets VALUES (?,?,?,?,?,?,?,?,?,?)",
                (dataset_id, name.strip(), domain.strip(), file_name, resolved_path, file_format,
                 row_count, byte_size, json.dumps(columns), created_at),
            )
        return {"id": dataset_id, "name": name.strip(), "domain": domain.strip(), "file_name": file_name,
                "file_path": resolved_path, "file_format": file_format, "row_count": row_count,
                "byte_size": byte_size, "columns": columns, "created_at": created_at}

    @staticmethod
    def _dataset_dict(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["columns"] = json.loads(value.pop("schema_json"))
        return value

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
        return self._dataset_dict(row) if row else None

    def list_datasets(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
        return [self._dataset_dict(row) for row in rows]

    def create_job(self, data: dict[str, Any]) -> dict[str, Any]:
        job_id, created_at = f"run-{uuid4().hex[:12]}", _now()
        artifact_group = "experts" if data["target"] == "expert" else "orchestrators"
        output_dir = self.artifact_root / artifact_group / job_id
        with self._connect() as db:
            db.execute(
                """INSERT INTO training_jobs
                (id,target,model_name,domain,description,approach,architecture,base_model,dataset_id,
                 eval_dataset_id,config_json,status,stage,progress,output_dir,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,'queued','queued',0,?,?)""",
                (job_id, data["target"], data["model_name"], data["domain"], data.get("description", ""),
                 data["approach"], data["architecture"], data.get("base_model"), data["dataset_id"],
                 data.get("eval_dataset_id"), json.dumps(data["config"]), str(output_dir), created_at),
            )
        self.add_event(job_id, level="INFO", stage="queued", progress=0, message="Training run queued")
        return self.get_job(job_id)  # type: ignore[return-value]

    @staticmethod
    def _job_dict(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["config"] = json.loads(value.pop("config_json"))
        value["cancel_requested"] = bool(value["cancel_requested"])
        value["kept"] = bool(value.get("kept", False))
        return value

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM training_jobs WHERE id=?", (job_id,)).fetchone()
        return self._job_dict(row) if row else None

    def list_jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM training_jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._job_dict(row) for row in rows]

    def keep_job(self, job_id: str) -> dict[str, Any]:
        with self._connect() as db:
            cursor = db.execute("UPDATE training_jobs SET kept=1 WHERE id=?", (job_id,))
        if not cursor.rowcount:
            raise KeyError(job_id)
        return self.get_job(job_id)  # type: ignore[return-value]

    def delete_job(self, job_id: str) -> None:
        """Delete one terminal run, its events, model records, and its own artifacts."""
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        terminal = {"built_evaluation_required", "built_evaluated", "failed", "cancelled"}
        if job["status"] not in terminal:
            raise ValueError("Cancel and wait for the run to stop before deleting it")

        artifact_group = "experts" if job["target"] == "expert" else "orchestrators"
        expected_output = (self.artifact_root / artifact_group / job_id).resolve()
        output_dir = Path(job["output_dir"]).resolve()
        if output_dir != expected_output:
            raise ValueError("Run artifact path is outside its expected training folder")

        with self._connect() as db:
            current = db.execute("SELECT status FROM training_jobs WHERE id=?", (job_id,)).fetchone()
            if current is None:
                raise KeyError(job_id)
            if current["status"] not in terminal:
                raise ValueError("Cancel and wait for the run to stop before deleting it")
            versions = db.execute("SELECT id FROM model_versions WHERE job_id=?", (job_id,)).fetchall()
            version_ids = [row["id"] for row in versions]
            if version_ids:
                placeholders = ",".join("?" for _ in version_ids)
                active = db.execute(
                    f"SELECT 1 FROM active_models WHERE model_version_id IN ({placeholders}) LIMIT 1",
                    version_ids,
                ).fetchone()
                if active:
                    raise ValueError("This run provides the active orchestrator model; deactivate it before deleting")
                db.execute(f"DELETE FROM model_evaluations WHERE model_version_id IN ({placeholders})", version_ids)
                db.execute(f"DELETE FROM model_versions WHERE id IN ({placeholders})", version_ids)
            db.execute("DELETE FROM training_events WHERE job_id=?", (job_id,))
            db.execute("DELETE FROM training_jobs WHERE id=?", (job_id,))

        if output_dir.exists():
            shutil.rmtree(output_dir)

    def update_job(self, job_id: str, **changes: Any) -> None:
        allowed = {"status", "stage", "progress", "worker_pid", "error_message", "started_at", "finished_at"}
        if set(changes) - allowed:
            raise ValueError("Unsupported training job update field")
        if changes:
            assignments = ", ".join(f"{key}=?" for key in changes)
            with self._connect() as db:
                db.execute(f"UPDATE training_jobs SET {assignments} WHERE id=?", (*changes.values(), job_id))

    def request_cancel(self, job_id: str) -> bool:
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE training_jobs SET cancel_requested=1 WHERE id=? AND status IN ('queued','running')",
                (job_id,),
            )
        return cursor.rowcount > 0

    def resume_job(self, job_id: str) -> dict[str, Any]:
        """Requeue an interrupted run only when Trainer saved a resumable checkpoint."""
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job["status"] not in {"failed", "cancelled"}:
            raise ValueError("Only failed or cancelled training runs can be resumed")
        checkpoints = sorted((Path(job["output_dir"]) / "trainer").glob("checkpoint-*"))
        if not checkpoints:
            raise ValueError("No saved checkpoint exists for this run. Start a new run instead.")
        with self._connect() as db:
            db.execute(
                "UPDATE training_jobs SET status='queued',stage='queued',cancel_requested=0,worker_pid=NULL,error_message=NULL,started_at=NULL,finished_at=NULL WHERE id=?",
                (job_id,),
            )
        self.add_event(job_id, level="INFO", stage="queued", progress=job["progress"],
                       message=f"Resume queued from saved checkpoint {checkpoints[-1].name}")
        return self.get_job(job_id)  # type: ignore[return-value]

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._connect() as db:
            row = db.execute("SELECT cancel_requested FROM training_jobs WHERE id=?", (job_id,)).fetchone()
        return bool(row and row["cancel_requested"])

    def has_active_job(self) -> bool:
        with self._connect() as db:
            row = db.execute("SELECT 1 FROM training_jobs WHERE status IN ('queued','running') LIMIT 1").fetchone()
        return row is not None

    def add_event(self, job_id: str, *, level: str, stage: str, progress: float | None,
                  message: str, metrics: dict[str, Any] | None = None) -> int:
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO training_events(job_id,created_at,level,stage,progress,message,metrics_json) VALUES (?,?,?,?,?,?,?)",
                (job_id, _now(), level, stage, progress, message,
                 json.dumps(metrics, ensure_ascii=False) if metrics is not None else None),
            )
            event_id = int(cursor.lastrowid)
            db.execute("UPDATE training_jobs SET stage=?,progress=COALESCE(?,progress) WHERE id=?",
                       (stage, progress, job_id))
        return event_id

    def get_events(self, job_id: str, after_id: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM training_events WHERE job_id=? AND id>? ORDER BY id LIMIT ?",
                              (job_id, after_id, limit)).fetchall()
        events = []
        for row in rows:
            item = dict(row)
            item["metrics"] = json.loads(item.pop("metrics_json")) if item["metrics_json"] else None
            events.append(item)
        return events

    def save_model_version(self, job: dict[str, Any], artifact_path: str | Path,
                           evaluation: dict[str, Any]) -> dict[str, Any]:
        model_version_id = f"model-{uuid4().hex[:12]}"
        task_metrics = evaluation.get("task_metrics") or {}
        task_split = str(task_metrics.get("split") or "").casefold()
        try:
            task_examples = int(task_metrics.get("examples_evaluated") or 0)
            test_examples = int(evaluation.get("test_examples_evaluated") or 0)
        except (TypeError, ValueError):
            task_examples, test_examples = 0, 0
        has_saved_heldout_metrics = (
            task_split in {"test", "validation"} and task_examples > 0
        ) or (str(evaluation.get("test_split") or "").casefold() == "test" and test_examples > 0)
        readiness = "ready_to_deploy" if has_saved_heldout_metrics else "evaluation_required"
        with self._connect() as db:
            count = db.execute("SELECT COUNT(*) AS n FROM model_versions WHERE target=? AND model_name=?",
                               (job["target"], job["model_name"])).fetchone()["n"]
            db.execute(
                """INSERT INTO model_versions
                (id,job_id,target,model_name,version,approach,architecture,artifact_path,artifact_uri,evaluation_json,readiness,created_at)
                VALUES (?,?,?,?,?,?,?,?,NULL,?,?,?)""",
                (model_version_id, job["id"], job["target"], job["model_name"], str(int(count) + 1),
                 job["approach"], job["architecture"], str(Path(artifact_path).resolve()),
                 json.dumps(evaluation), readiness, _now()),
            )
        return self.get_model_version(model_version_id)  # type: ignore[return-value]

    def update_artifact_uri(self, model_version_id: str, artifact_uri: str | None) -> None:
        with self._connect() as db:
            cursor = db.execute("UPDATE model_versions SET artifact_uri=? WHERE id=?",
                                (artifact_uri, model_version_id))
        if not cursor.rowcount:
            raise KeyError(model_version_id)

    def list_model_versions(self, target: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as db:
            if target:
                rows = db.execute("SELECT * FROM model_versions WHERE target=? ORDER BY created_at DESC", (target,)).fetchall()
            else:
                rows = db.execute("SELECT * FROM model_versions ORDER BY created_at DESC").fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["evaluation"] = json.loads(item.pop("evaluation_json")) if item["evaluation_json"] else None
            result.append(item)
        return result

    def get_model_version(self, model_version_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM model_versions WHERE id=?", (model_version_id,)).fetchone()
        if not row:
            return None
        value = dict(row)
        value["evaluation"] = json.loads(value.pop("evaluation_json")) if value["evaluation_json"] else None
        return value

    def save_task_metrics(self, model_version_id: str, metrics: dict[str, Any]) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT evaluation_json FROM model_versions WHERE id=?", (model_version_id,)).fetchone()
            if row is None:
                raise KeyError(model_version_id)
            evaluation = json.loads(row["evaluation_json"] or "{}")
            evaluation["task_metrics"] = metrics
            db.execute("UPDATE model_versions SET evaluation_json=? WHERE id=?",
                       (json.dumps(evaluation, ensure_ascii=False), model_version_id))
        return self.get_model_version(model_version_id)  # type: ignore[return-value]

    def mark_model_ready_for_deployment(self, model_version_id: str) -> dict[str, Any]:
        """Persist deployment readiness after the saved in-run evaluation is checked by the API."""
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE model_versions SET readiness='ready_to_deploy' WHERE id=?",
                (model_version_id,),
            )
        if not cursor.rowcount:
            raise KeyError(model_version_id)
        return self.get_model_version(model_version_id)  # type: ignore[return-value]

    def create_evaluation(self, model_version_id: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
        evaluation_id, created_at = f"eval-{uuid4().hex[:12]}", _now()
        with self._connect() as db:
            db.execute(
                "INSERT INTO model_evaluations(id,model_version_id,cases_json,status,created_at) VALUES (?,?,?,'awaiting_review',?)",
                (evaluation_id, model_version_id, json.dumps(cases, ensure_ascii=False), created_at),
            )
        return {"id": evaluation_id, "model_version_id": model_version_id, "cases": cases,
                "status": "awaiting_review", "created_at": created_at}

    def get_evaluation(self, evaluation_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM model_evaluations WHERE id=?", (evaluation_id,)).fetchone()
        if not row:
            return None
        value = dict(row)
        value["cases"] = json.loads(value.pop("cases_json"))
        return value

    def get_evaluation_for_model_version(self, model_version_id: str) -> dict[str, Any] | None:
        """Return the most recent in-run evaluation packet without generating new outputs."""
        with self._connect() as db:
            row = db.execute(
                "SELECT id FROM model_evaluations WHERE model_version_id=? ORDER BY created_at DESC LIMIT 1",
                (model_version_id,),
            ).fetchone()
        return self.get_evaluation(row["id"]) if row else None

    def submit_evaluation_review(self, evaluation_id: str, *, reviewer: str, reviewer_role: str,
                                 ratings: list[bool], threshold: float = 0.85) -> dict[str, Any]:
        evaluation = self.get_evaluation(evaluation_id)
        if evaluation is None:
            raise KeyError(evaluation_id)
        if evaluation["status"] != "awaiting_review":
            raise ValueError("This evaluation has already been reviewed")
        if len(ratings) != len(evaluation["cases"]) or not ratings:
            raise ValueError("Provide one pass/fail rating for every evaluation case")
        passed = sum(bool(value) for value in ratings)
        score = passed / len(ratings)
        reviewed_at = _now()
        with self._connect() as db:
            db.execute(
                "UPDATE model_evaluations SET status='reviewed',reviewer=?,reviewer_role=?,score=?,pass_count=?,case_count=?,reviewed_at=? WHERE id=?",
                (reviewer.strip(), reviewer_role.strip(), score, passed, len(ratings), reviewed_at, evaluation_id),
            )
            version = db.execute("SELECT * FROM model_versions WHERE id=?", (evaluation["model_version_id"],)).fetchone()
            current = json.loads(version["evaluation_json"] or "{}")
            current["manual_review"] = {
                "reviewer": reviewer.strip(), "reviewer_role": reviewer_role.strip(),
                "score": score, "pass_count": passed, "case_count": len(ratings),
                "threshold": threshold, "reviewed_at": reviewed_at,
            }
            readiness = "ready_to_deploy" if score >= threshold else "needs_improvement"
            db.execute("UPDATE model_versions SET evaluation_json=?,readiness=? WHERE id=?",
                       (json.dumps(current, ensure_ascii=False), readiness, evaluation["model_version_id"]))
        return {"evaluation_id": evaluation_id, "model_version_id": evaluation["model_version_id"],
                "score": score, "pass_count": passed, "case_count": len(ratings),
                "threshold": threshold, "readiness": readiness, "reviewer": reviewer.strip(),
                "reviewer_role": reviewer_role.strip(), "reviewed_at": reviewed_at}

    def activate_model_version(self, model_version_id: str) -> dict[str, Any]:
        with self._connect() as db:
            version = db.execute("SELECT * FROM model_versions WHERE id=?", (model_version_id,)).fetchone()
            if version is None:
                raise KeyError(model_version_id)
            if version["readiness"] != "ready_to_deploy":
                raise ValueError("Model version must pass reviewed evaluation before activation")
            if version["target"] == "orchestrator":
                db.execute(
                    "INSERT INTO active_models(target,model_version_id,activated_at) VALUES (?,?,?) "
                    "ON CONFLICT(target) DO UPDATE SET model_version_id=excluded.model_version_id,activated_at=excluded.activated_at",
                    (version["target"], model_version_id, _now()),
                )
        return self.get_model_version(model_version_id)  # type: ignore[return-value]

    def update_artifact_path(self, model_version_id: str, path: str | Path) -> None:
        with self._connect() as db:
            db.execute("UPDATE model_versions SET artifact_path=? WHERE id=?",
                       (str(Path(path).resolve()), model_version_id))

    def get_active_model(self, target: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT model_version_id FROM active_models WHERE target=?", (target,)
            ).fetchone()
        return self.get_model_version(row["model_version_id"]) if row else None

    def deactivate_model_version(self, model_version_id: str) -> bool:
        """Deactivate a currently selected model version, if it owns its target slot."""
        with self._connect() as db:
            version = db.execute("SELECT target FROM model_versions WHERE id=?", (model_version_id,)).fetchone()
            if version is None:
                raise KeyError(model_version_id)
            cursor = db.execute(
                "DELETE FROM active_models WHERE target=? AND model_version_id=?",
                (version["target"], model_version_id),
            )
        return bool(cursor.rowcount)

