"""Card 4.4 — automated retraining control loop (decision logic).

Pure, Airflow-free decision logic used by dags/auto_retrain_pipeline.py to
close the MLOps loop: drift signal -> cooldown/dedupe guard -> next category
slice -> trigger the existing Card 3.2 model pipeline. Lives under src/ and
imports only the stdlib, so it stays unit-testable without Airflow or Evidently.
"""
