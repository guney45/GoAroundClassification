"""Load the trained model bundle at startup."""
from __future__ import annotations
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR     = Path(__file__).resolve().parent.parent
MODELS_DIR   = BASE_DIR / "models"
MODEL_PATH   = MODELS_DIR / "final_model.joblib"
SCHEMA_PATH  = MODELS_DIR / "feature_schema.json"

_bundle      = None
_schema      = None


def get_bundle():
    global _bundle, _schema
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
        if SCHEMA_PATH.exists():
            with open(SCHEMA_PATH) as fh:
                _schema = json.load(fh)
        else:
            _schema = _bundle.get("feature_schema", {})
    return _bundle, _schema


def predict(features: dict) -> dict:
    bundle, schema = get_bundle()
    model        = bundle["model"]
    preprocessor = bundle["preprocessor"]

    num_feats = schema.get("numeric_features", [])
    cat_feats = schema.get("categorical_features", [])
    threshold  = float(schema.get("best_threshold", 0.5))

    row = {}
    for f in num_feats:
        val = features.get(f)
        row[f] = float(val) if val is not None else np.nan
    for f in cat_feats:
        val = features.get(f)
        row[f] = str(val) if val is not None else "UNKNOWN"

    X = pd.DataFrame([row])[num_feats + cat_feats]

    from sklearn.pipeline import Pipeline as _Pipeline
    if isinstance(model, _Pipeline):
        proba = model.predict_proba(X)[0]
    else:
        try:
            X_pre = preprocessor.transform(X)
        except Exception:
            X_pre = X.values
        proba = model.predict_proba(X_pre)[0]
    prob_ga_raw = float(proba[1])

    # Apply prior-shift + isotonic calibration if the bundle ships them.
    from src.models.common import apply_calibration
    prob_ga = float(apply_calibration(
        np.array([prob_ga_raw]),
        bundle.get("calibrator"),
        bundle.get("prior_train"),
        bundle.get("prior_test"),
    )[0])
    prob_nl = 1.0 - prob_ga
    pred_class = int(prob_ga >= threshold)
    label = "Go-Around Risk" if pred_class == 1 else "Normal Landing"

    return {
        "predicted_class":          pred_class,
        "predicted_label":          label,
        "probability_go_around":    round(prob_ga, 4),
        "probability_normal_landing": round(prob_nl, 4),
        "threshold":                threshold,
    }
