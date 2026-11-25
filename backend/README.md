# Mornork backend

The FastAPI service, model routing, training workers, evaluation, S3 artifact handling and workflow APIs live here. HTML pages and CSS live in the sibling frontend/ directory. Persistent user data remains in the project root, including the SQLite database, datasets, checkpoints, logs and .env file.

## Run the backend

In a PowerShell window:

    cd D:\ork\backend
    ..\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    python -m uvicorn mornork.api:app --app-dir src --host 127.0.0.1 --port 8765

The install command is needed once per environment, or when dependencies change. FastAPI listens at http://127.0.0.1:8765 and reads settings from D:\ork\.env.

## Run the frontend separately

In a second PowerShell window:

    cd D:\ork\frontend
    & D:\ork\.venv\Scripts\python.exe -m http.server 5500 --bind 127.0.0.1

Open http://127.0.0.1:5500. Keep both processes running. The frontend uses frontend/api-config.js to call FastAPI. Set MORNORK_FRONTEND_ORIGINS in .env if the frontend is served from a different origin.

## Project layout

- backend/src/mornork/: FastAPI app and application modules.
- backend/tests/: backend tests.
- frontend/: static HTML pages, interaction scripts and CSS.
- artifacts/, config/, data/, logs/: persistent model, configuration, dataset, database and log files outside the code folders.

S3 bucket, prefix, profile and region are read from .env and the AWS profile. The Models page displays model artifacts discovered in the configured S3 catalog, with saved details and evaluation results.
