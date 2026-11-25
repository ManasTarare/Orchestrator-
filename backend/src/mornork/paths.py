"""Project paths shared by the API and artifact storage."""

from pathlib import Path


def project_root() -> Path:
    """Locate the checkout root that contains .env, frontend, and persistent data."""
    module = Path(__file__).resolve()
    for candidate in module.parents:
        if (candidate / ".env").is_file() and (candidate / "frontend").is_dir():
            return candidate
    return module.parents[3]
