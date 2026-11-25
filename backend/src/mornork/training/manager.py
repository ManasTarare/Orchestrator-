"""One-at-a-time local training worker supervisor."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .store import TrainingStore


class TrainingManager:
    """Run durable queued jobs in separate Python processes on the local machine."""

    def __init__(self, store: TrainingStore, project_root: str | Path, log_root: str | Path,
                 before_gpu_job: Callable[[], None] | None = None,
                 after_gpu_job: Callable[[], None] | None = None) -> None:
        self.store = store
        self.project_root = Path(project_root).resolve()
        self.log_root = Path(log_root).resolve()
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.before_gpu_job = before_gpu_job
        self.after_gpu_job = after_gpu_job
        self._lock = threading.RLock()
        self._active_process: subprocess.Popen | None = None
        self._active_job_id: str | None = None
        self._active_uses_gpu = False
        self._gpu_prepared = False
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stopping.clear()
            self._thread = threading.Thread(target=self._run, name="mornork-training-supervisor", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stopping.set()

    def enqueue(self, job_id: str) -> None:
        if not self.store.get_job(job_id):
            raise KeyError(job_id)
        self.start()

    def cancel(self, job_id: str) -> bool:
        """Stop a queued or active worker and persist cancellation before returning."""
        with self._lock:
            job = self.store.get_job(job_id)
            if job is None or job["status"] not in {"queued", "running"}:
                return False
            if not self.store.request_cancel(job_id):
                return False

            if job["status"] == "running":
                process = self._active_process if self._active_job_id == job_id else None
                pid = process.pid if process is not None else job.get("worker_pid")
                if process is not None:
                    try:
                        if process.poll() is None:
                            process.kill()
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                elif pid and self._pid_alive(int(pid)):
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    deadline = time.monotonic() + 3
                    while self._pid_alive(int(pid)) and time.monotonic() < deadline:
                        time.sleep(0.05)
                    if self._pid_alive(int(pid)):
                        force_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
                        try:
                            os.kill(int(pid), force_signal)
                        except ProcessLookupError:
                            pass
                        deadline = time.monotonic() + 2
                        while self._pid_alive(int(pid)) and time.monotonic() < deadline:
                            time.sleep(0.05)
                        if self._pid_alive(int(pid)):
                            raise RuntimeError("Training worker did not stop after the terminate signal")

                if self._active_job_id == job_id:
                    self._active_process = None
                    self._active_job_id = None
                    self._active_uses_gpu = False

            self.store.update_job(job_id, status="cancelled", stage="cancelled",
                                  finished_at=datetime.now(timezone.utc).isoformat(), worker_pid=None)
            self.store.add_event(job_id, level="WARN", stage="cancelled", progress=job.get("progress"),
                                 message="Training worker stopped immediately at the user's request")
            return True

    @staticmethod
    def _pid_alive(pid: int | None) -> bool:
        if not pid:
            return False
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def _prepare_gpu(self) -> None:
        if not self._gpu_prepared and self.before_gpu_job:
            self.before_gpu_job()
        self._gpu_prepared = True

    def _restore_gpu(self) -> None:
        if self._gpu_prepared and self.after_gpu_job:
            try:
                self.after_gpu_job()
            finally:
                self._gpu_prepared = False

    def _fail_interrupted(self, job_id: str, reason: str) -> None:
        self.store.update_job(job_id, status="failed", stage="failed", error_message=reason,
                              finished_at=datetime.now(timezone.utc).isoformat(),
                              worker_pid=None)
        self.store.add_event(job_id, level="ERROR", stage="failed", progress=None, message=reason)

    def _run(self) -> None:
        while not self._stopping.wait(0.75):
            try:
                self._tick()
            except Exception:
                # Keep the supervisor alive; worker-level failures are persisted by the worker.
                continue

    def _tick(self) -> None:
        with self._lock:
            if self._active_process is not None:
                if self._active_process.poll() is None:
                    return
                job_id = self._active_job_id
                code = self._active_process.returncode
                self._active_process = None
                self._active_job_id = None
                if job_id:
                    job = self.store.get_job(job_id)
                    if job and job["status"] in {"queued", "running"}:
                        self._fail_interrupted(job_id, f"Training worker exited unexpectedly (code {code})")
                if self._active_uses_gpu:
                    self._restore_gpu()
                self._active_uses_gpu = False

            jobs = self.store.list_jobs(limit=1000)
            running = next((job for job in jobs if job["status"] == "running"), None)
            if running:
                if self._pid_alive(running.get("worker_pid")):
                    if str(running["config"].get("device", "auto")) != "cpu":
                        self._active_job_id = running["id"]
                        self._active_uses_gpu = True
                        self._prepare_gpu()
                    return
                self._fail_interrupted(running["id"], "Training worker stopped before the run finished; resume by starting a new run")
                if self._gpu_prepared:
                    self._restore_gpu()
            elif self._gpu_prepared:
                # A worker can finish while this process is temporarily offline; restore the chat model.
                self._restore_gpu()

            queued = [job for job in jobs if job["status"] == "queued"]
            if not queued:
                return
            job = queued[-1]  # list is newest first; run the oldest queued item first.
            if job["cancel_requested"]:
                self.store.update_job(job["id"], status="cancelled", stage="cancelled", progress=100,
                                      finished_at=datetime.now(timezone.utc).isoformat())
                self.store.add_event(job["id"], level="WARN", stage="cancelled", progress=100,
                                     message="Queued run cancelled by user")
                return
            uses_gpu = str(job["config"].get("device", "auto")) != "cpu"
            try:
                if uses_gpu:
                    self._prepare_gpu()
                self.store.update_job(job["id"], status="running", stage="starting")
                self.store.add_event(job["id"], level="INFO", stage="starting", progress=1,
                                     message="Starting local training worker")
                log_path = self.log_root / f"{job['id']}.log"
                command = [
                    sys.executable, "-m", "mornork.training.worker",
                    "--database", str(self.store.database),
                    "--datasets", str(self.store.dataset_root),
                    "--artifacts", str(self.store.artifact_root),
                    "--job-id", job["id"],
                ]
                creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                with log_path.open("ab") as output:
                    self._active_process = subprocess.Popen(
                        command, cwd=self.project_root, stdin=subprocess.DEVNULL,
                        stdout=output, stderr=subprocess.STDOUT, creationflags=creation_flags,
                    )
                self._active_job_id = job["id"]
                self._active_uses_gpu = uses_gpu
                self.store.update_job(job["id"], worker_pid=self._active_process.pid)
            except Exception as error:
                self._active_process = None
                self._active_job_id = None
                self._fail_interrupted(job["id"], f"Could not start local worker: {type(error).__name__}: {error}")
                if uses_gpu:
                    self._restore_gpu()
                self._active_uses_gpu = False
