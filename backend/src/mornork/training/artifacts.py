"""Optional versioned S3 storage with an on-disk inference cache."""

from __future__ import annotations

import os
import json
from pathlib import Path, PurePosixPath
import shutil
from urllib.parse import urlparse

from ..paths import project_root


def _s3_client():
    """Create an S3 client from the active AWS profile and environment settings."""
    try:
        import boto3
    except ImportError as error:
        raise RuntimeError("S3 artifact storage requires boto3; run: python -m pip install -e .") from error

    profile = os.environ.get("AWS_PROFILE") or None
    region = (
        os.environ.get("AWS_DEFAULT_REGION")
        or os.environ.get("AWS_REGION")
        or None
    )
    session = boto3.Session(profile_name=profile, region_name=region)
    return session.client("s3", region_name=region) if region else session.client("s3")


class ModelArtifactStore:
    def __init__(self) -> None:
        self.storage = os.environ.get("MORNORK_ARTIFACT_STORAGE", "local").strip().casefold()
        self.bucket = os.environ.get("MORNORK_S3_BUCKET", "").strip()
        self.prefix = os.environ.get("MORNORK_S3_PREFIX", "mornork/models").strip("/")
        root = project_root()
        cache = Path(os.environ.get("MORNORK_MODEL_CACHE_DIR", "artifacts/model-cache"))
        self.cache = cache.resolve() if cache.is_absolute() else (root / cache).resolve()
        self.upload_enabled = os.environ.get("MORNORK_UPLOAD_TRAINED_MODELS", "false").casefold() in {"1", "true", "yes", "on"}
        self.load_enabled = os.environ.get("MORNORK_LOAD_MODELS_FROM_S3", "true").casefold() in {"1", "true", "yes", "on"}

    def upload_directory(
        self,
        directory: str | Path,
        target: str,
        model_version_id: str,
        *,
        manifest: dict[str, object] | None = None,
        force: bool = False,
    ) -> str | None:
        if not self.upload_enabled and not force:
            return None
        if self.storage != "s3":
            raise ValueError("Set MORNORK_ARTIFACT_STORAGE=s3 to upload trained model artifacts")
        if not self.bucket:
            raise ValueError("MORNORK_S3_BUCKET is required when S3 artifact upload is enabled")
        root = Path(directory).resolve()
        key_root = PurePosixPath(self.prefix, target, model_version_id)
        client = _s3_client()
        for path in root.rglob("*"):
            if path.is_file():
                if manifest is not None and path.name == "mornork-model.json":
                    continue
                key = str(key_root / path.relative_to(root).as_posix())
                client.upload_file(str(path), self.bucket, key)
        if manifest is not None:
            manifest_key = str(key_root / "mornork-model.json")
            client.put_object(
                Bucket=self.bucket,
                Key=manifest_key,
                Body=json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
                ContentType="application/json",
            )
        return f"s3://{self.bucket}/{key_root.as_posix()}"

    def update_manifest(self, artifact_uri: str, manifest: dict[str, object]) -> None:
        parsed = urlparse(artifact_uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError("Model artifact URI must be an s3://bucket/prefix URL")
        prefix = parsed.path.lstrip("/").rstrip("/")
        _s3_client().put_object(
            Bucket=parsed.netloc,
            Key=f"{prefix}/mornork-model.json",
            Body=json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

    def list_model_artifacts(self) -> list[dict[str, object]]:
        """Return model manifests that are actually present under the configured S3 prefix."""
        if self.storage != "s3" or not self.bucket:
            raise ValueError("Configure MORNORK_ARTIFACT_STORAGE=s3 and MORNORK_S3_BUCKET to list S3 models")
        client = _s3_client()
        prefix = f"{self.prefix}/" if self.prefix else ""
        artifacts: list[dict[str, object]] = []
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                key = str(item.get("Key", ""))
                if not key.endswith("/mornork-model.json"):
                    continue
                relative = key[len(prefix):] if prefix and key.startswith(prefix) else key
                parts = relative.split("/")
                if len(parts) < 3 or parts[0] not in {"expert", "orchestrator"}:
                    continue
                target, version_id = parts[0], parts[1]
                try:
                    response = client.get_object(Bucket=self.bucket, Key=key)
                    manifest = json.loads(response["Body"].read().decode("utf-8"))
                except (KeyError, UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if not isinstance(manifest, dict):
                    continue
                uri = f"s3://{self.bucket}/{prefix}{target}/{version_id}"
                artifacts.append({
                    "id": version_id,
                    "target": target,
                    "artifact_uri": uri,
                    "manifest": manifest,
                    "last_modified": item.get("LastModified").isoformat() if item.get("LastModified") else None,
                })
        return sorted(artifacts, key=lambda item: str(item.get("last_modified") or ""), reverse=True)

    def materialize(self, artifact_path: str | Path | None, artifact_uri: str | None) -> Path:
        path = Path(artifact_path).resolve() if artifact_path else None
        if (path is not None and path.is_dir()) or not artifact_uri or not self.load_enabled:
            if path is None or not path.is_dir():
                raise FileNotFoundError(f"Model artifact directory not found: {path or artifact_uri}")
            return path
        parsed = urlparse(artifact_uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError("Model artifact URI must be an s3://bucket/prefix URL")
        self.cache.mkdir(parents=True, exist_ok=True)
        destination = self.cache / parsed.netloc / parsed.path.lstrip("/").replace("/", "_")
        if (destination / "config.json").is_file():
            return destination
        destination.mkdir(parents=True, exist_ok=True)
        client = _s3_client()
        bucket, prefix = parsed.netloc, parsed.path.lstrip("/").rstrip("/") + "/"
        paginator = client.get_paginator("list_objects_v2")
        found = False
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                relative = key[len(prefix):]
                if not relative or ".." in Path(relative).parts:
                    continue
                target = (destination / relative).resolve()
                target.relative_to(destination.resolve())
                target.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(bucket, key, str(target))
                found = True
        if not found:
            raise FileNotFoundError(f"No model objects found at {artifact_uri}")
        return destination

    def delete_artifact(self, artifact_uri: str | None) -> None:
        if not artifact_uri:
            return
        parsed = urlparse(artifact_uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError("Model artifact URI must be an s3://bucket/prefix URL")
        client = _s3_client()
        bucket, prefix = parsed.netloc, parsed.path.lstrip("/").rstrip("/") + "/"
        paginator = client.get_paginator("list_objects_v2")
        batch: list[dict[str, str]] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                batch.append({"Key": item["Key"]})
                if len(batch) == 1000:
                    client.delete_objects(Bucket=bucket, Delete={"Objects": batch})
                    batch = []
        if batch:
            client.delete_objects(Bucket=bucket, Delete={"Objects": batch})
        cache_path = self.cache / parsed.netloc / parsed.path.lstrip("/").replace("/", "_")
        if cache_path.is_dir():
            shutil.rmtree(cache_path)
