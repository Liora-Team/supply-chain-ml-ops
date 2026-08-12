# ADR 001: MLflow 3.14 Local Storage Fallback

**Context:** 
MLflow >= 3.14 blocks the local filesystem backend (`file://`) by default, placing it in maintenance mode. Card 2.1 introduces MLflow experiment tracking for the training pipelines, and we need a robust offline fallback that works across operating systems (including Windows) without relying on deprecated backends. 

**Decision:**
Instead of globally overriding `MLFLOW_ALLOW_FILE_STORE=true` via environment variables (which risks polluting production state), we fallback to using a local SQLite database (`sqlite:///mlruns/mlflow.db`) when `MLFLOW_TRACKING_URI` is unset.

The SQLite URI is constructed from `ROOT / "mlruns" / "mlflow.db"` and normalised to a forward-slash path so that it remains valid on Windows as well as on Unix-like systems.

This fallback is implemented in `scripts/build_pipelines.py` when `HAS_MLFLOW` is true and `MLFLOW_TRACKING_URI` is unset.

**Consequences**  

- Offline training and CI work seamlessly across OS environments without mutating global state or relying on the deprecated `file://` store.
- The MLflow 3.14+ maintenance-mode restriction for the file backend is avoided entirely.
- The behaviour is deterministic and compatible with the existing training scripts and the tracking tests, while keeping the production configuration clean.