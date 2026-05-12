"""Shared model utilities: preprocessing, evaluation, serialisation."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import TRAIN_PARQUET, VALID_PARQUET, TEST_PARQUET
from src.features.build_features import load_data, build_feature_matrix


# --------------------------------------------------------------------------- #
# Data loading                                                                 #
# --------------------------------------------------------------------------- #

def load_splits(
    feature_set: str = "context_metar",
    sample_frac: float | None = None,
    neg_ratio: int | None = None,
) -> tuple:
    """Return (X_tr, y_tr, X_va, y_va, X_te, y_te, num_feats, cat_feats).

    Negative undersampling (applied only to training set, all positives kept):
      neg_ratio  – keep at most (neg_ratio × n_positive) negatives.
                   Recommended: 10 for a ~9 % positive rate.
                   Takes precedence over sample_frac when both are given.
      sample_frac – keep this fraction of negatives (legacy flag).
    """
    print(f"  Loading splits (feature_set={feature_set}) ...")
    t0 = time.time()

    train = load_data(TRAIN_PARQUET)
    valid = load_data(VALID_PARQUET)
    test  = load_data(TEST_PARQUET)

    if neg_ratio is not None:
        pos_tr  = train[train["target"] == 1]
        neg_all = train[train["target"] == 0]
        n_neg   = min(len(neg_all), neg_ratio * len(pos_tr))
        neg_tr  = neg_all.sample(n=n_neg, random_state=42)
        train   = pd.concat([pos_tr, neg_tr]).sample(frac=1, random_state=42)
        rate    = 100.0 * len(pos_tr) / len(train)
        print(f"  Train (neg_ratio={neg_ratio}): {len(train):,} rows | "
              f"{len(pos_tr):,} go-arounds ({rate:.1f} %)")
    elif sample_frac is not None and sample_frac < 1.0:
        pos_tr = train[train["target"] == 1]
        neg_tr = train[train["target"] == 0].sample(frac=sample_frac, random_state=42)
        train  = pd.concat([pos_tr, neg_tr]).sample(frac=1, random_state=42)
        rate   = 100.0 * int(train["target"].sum()) / len(train)
        print(f"  Train (sample_frac={sample_frac}): {len(train):,} rows | "
              f"{int(train['target'].sum()):,} go-arounds ({rate:.1f} %)")

    print(f"  Train: {len(train):,}  Valid: {len(valid):,}  Test: {len(test):,}")

    X_tr, y_tr, num_feats, cat_feats = build_feature_matrix(train, feature_set)
    X_va, y_va, _,         _         = build_feature_matrix(valid, feature_set)
    X_te, y_te, _,         _         = build_feature_matrix(test,  feature_set)

    print(f"  Features: {len(num_feats)} numeric + {len(cat_feats)} categorical  [{time.time()-t0:.1f}s]")
    return X_tr, y_tr, X_va, y_va, X_te, y_te, num_feats, cat_feats


# --------------------------------------------------------------------------- #
# Preprocessing                                                                #
# --------------------------------------------------------------------------- #

def create_preprocessor(numeric_features: list[str], categorical_features: list[str],
                        ohe_min_frequency: int = 2000) -> ColumnTransformer:
    """
    ohe_min_frequency: a category must appear at least this many times in the
    training set to get its own OHE column (others → 'infrequent_sklearn').
    2000 out of ~5.6M rows ≈ top airports/runways only → keeps RAM manageable.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=ohe_min_frequency,
            sparse_output=False,
        )),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline,     numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


# --------------------------------------------------------------------------- #
# Evaluation                                                                   #
# --------------------------------------------------------------------------- #

def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    split_name: str = "test",
) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred).tolist()
    metrics = {
        f"{split_name}_accuracy":           round(float(accuracy_score(y_true, y_pred)), 6),
        f"{split_name}_precision":          round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        f"{split_name}_recall":             round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        f"{split_name}_f1":                 round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        f"{split_name}_roc_auc":            round(float(roc_auc_score(y_true, y_prob)), 6),
        f"{split_name}_average_precision":  round(float(average_precision_score(y_true, y_prob)), 6),
        f"{split_name}_confusion_matrix":   cm,
        f"{split_name}_threshold":          threshold,
    }
    return metrics


def tune_threshold_for_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    denom = precisions[:-1] + recalls[:-1]
    f1s = np.where(denom > 0, 2 * precisions[:-1] * recalls[:-1] / np.maximum(denom, 1e-9), 0.0)
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx])


# --------------------------------------------------------------------------- #
# Serialisation                                                                #
# --------------------------------------------------------------------------- #

def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"  Metrics saved → {path.name}")


def save_model_bundle(
    model: Any,
    preprocessor: Any,
    feature_schema: dict,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model":          model,
        "preprocessor":   preprocessor,
        "feature_schema": feature_schema,
    }
    joblib.dump(bundle, path)
    print(f"  Model bundle saved → {path.name}")


def load_model_bundle(path: Path) -> dict:
    return joblib.load(path)
