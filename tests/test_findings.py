"""Guard the hand-verified constants in src/findings.py against silent drift.

findings.PER_CLASS stores per-class metrics as literals because the test set is
not committed. The macro-F1 headline, however, IS derivable from the committed
pipeline sidecars — so pin them together: if a retrain updates the sidecar but
nobody updates findings.py, this test fails.
"""

import json
from pathlib import Path

from src import findings

ROOT = Path(__file__).resolve().parents[1]
PIPELINES = ROOT / "models" / "pipelines"


def test_per_class_macro_f1_matches_served_sidecar():
    for schema in ("3-class", "5-class"):
        sidecar = PIPELINES / f"LogReg__grid_+_cw=balanced__{schema}.json"
        meta = json.loads(sidecar.read_text())
        assert round(meta["pipeline_macro_f1"], 3) == findings.PER_CLASS[schema]["macro_f1"], (
            f"{schema}: findings.PER_CLASS macro_f1 no longer matches {sidecar.name} — "
            "regenerate the PER_CLASS block (recipe in src/findings.py)."
        )


def test_no_zero_byte_dvc_artifacts():
    """Every .dvc pointer must describe a non-empty file.

    A size: 0 pointer usually means dvc add was run against a
    missing/empty file (e.g. git cat-file on a gitignored path).
    """
    import yaml

    root = Path(__file__).parent.parent
    for dvc_file in root.rglob("*.dvc"):
        if not dvc_file.is_file():
            continue  # skip directories like .dvc/
        data = yaml.safe_load(dvc_file.read_text())
        for out in data.get("outs", []):
            size = out.get("size", 0)
            assert size > 0, f"{dvc_file} points to a 0-byte artifact"


def test_all_loadable_models_have_positive_size():
    """A model advertised as loadable must have a non-empty joblib file."""
    from src.registry import classical

    for entry in classical("5-class") + classical("3-class"):
        if entry.joblib_path is not None:
            assert (
                entry.joblib_path.stat().st_size > 0
            ), f"{entry.model_id} is loadable but {entry.joblib_path} is empty"
