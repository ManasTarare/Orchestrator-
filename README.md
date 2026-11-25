# Mornork

Mornork is a local model-training and routing workspace. The HTML/CSS frontend and FastAPI backend run as separate processes; runtime data and configuration remain in the project root.

## Layout

- frontend/: Chat, Models, Training, Workflows pages and CSS.
- backend/: FastAPI application, model training, evaluation, routing and tests.
- .env: local settings, including S3 bucket, prefix, AWS profile and region.
- artifacts/, config/, data/, logs/: persistent checkpoints, configuration, training data/database and logs.

## Start the backend

Open a PowerShell window:

    cd D:\ork\backend
    ..\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    python -m uvicorn mornork.api:app --app-dir src --host 127.0.0.1 --port 8765

The install command is needed once per environment, or when dependencies change. FastAPI listens at http://127.0.0.1:8765 and reads configuration from D:\ork\.env.

## Start the frontend

Open a second PowerShell window:

    cd D:\ork\frontend
    & D:\ork\.venv\Scripts\python.exe -m http.server 5500 --bind 127.0.0.1

Open http://127.0.0.1:5500. Leave both terminal windows running. To use another backend address, open the frontend with ?api=http://HOST:PORT; the address is saved in this browser. Add any changed frontend origin to MORNORK_FRONTEND_ORIGINS in .env.

Model training runs and evaluation records remain local. The Models page shows the configured S3 catalog and the details and evaluation metrics saved with its artifacts.
