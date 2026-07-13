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
