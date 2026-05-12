"""Shared model utilities: preprocessing, evaluation, calibration, serialisation."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
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
# Data loading                                                                #
# --------------------------------------------------------------------------- #

def _stratified_negative_sample(
    train: pd.DataFrame, neg_ratio: int, random_state: int = 42
) -> pd.DataFrame:
    """
    Keep all positives. Sample negatives stratified by airport so high-volume
    hubs do not crowd out long-tail airports — preserves the per-airport prior
    that the model needs to learn airport-specific decision boundaries.
    """
    pos = train[train["target"] == 1]
    neg = train[train["target"] == 0]

    target_neg = neg_ratio * len(pos)
    if target_neg >= len(neg):
        return pd.concat([pos, neg]).sample(frac=1, random_state=random_state)

    if "airport" in neg.columns:
        # Proportional allocation per airport, at least 1 negative per airport
        airport_counts = neg["airport"].value_counts()
        total_neg = airport_counts.sum()
        per_airport = (airport_counts / total_neg * target_neg).round().astype(int).clip(lower=1)
        sampled = []
        rng = np.random.default_rng(random_state)
        for ap, n_take in per_airport.items():
            grp = neg[neg["airport"] == ap]
            n_take = min(int(n_take), len(grp))
            if n_take > 0:
                idx = rng.choice(grp.index.values, size=n_take, replace=False)
                sampled.append(grp.loc[idx])
        neg_tr = pd.concat(sampled) if sampled else neg.sample(n=target_neg, random_state=random_state)
    else:
        neg_tr = neg.sample(n=target_neg, random_state=random_state)

    return pd.concat([pos, neg_tr]).sample(frac=1, random_state=random_state)


def load_splits(
    feature_set: str = "full",
    sample_frac: float | None = None,
    neg_ratio: int | None = None,
) -> tuple:
    """Return (X_tr, y_tr, X_va, y_va, X_te, y_te, num_feats, cat_feats).

    Validation and test sets are *always* left at natural class prevalence so
    that the threshold tuned on validation and the metrics reported on test
    reflect the deployment distribution. Negative undersampling — if requested
    — is applied only to the training set, stratified by airport.
    """
    print(f"  Loading splits (feature_set={feature_set}) ...")
    t0 = time.time()

    train = load_data(TRAIN_PARQUET)
    valid = load_data(VALID_PARQUET)
    test = load_data(TEST_PARQUET)

    if neg_ratio is not None:
        n_before = len(train)
        train = _stratified_negative_sample(train, neg_ratio=neg_ratio)
        pos = int(train["target"].sum())
        rate = 100.0 * pos / len(train)
        print(f"  Train (stratified neg_ratio={neg_ratio}): "
              f"{n_before:,} → {len(train):,} rows | {pos:,} go-arounds ({rate:.2f} %)")
    elif sample_frac is not None and sample_frac < 1.0:
        pos_tr = train[train["target"] == 1]
        neg_tr = train[train["target"] == 0].sample(frac=sample_frac, random_state=42)
        train = pd.concat([pos_tr, neg_tr]).sample(frac=1, random_state=42)
        rate = 100.0 * int(train["target"].sum()) / len(train)
        print(f"  Train (sample_frac={sample_frac}): {len(train):,} rows | "
              f"{int(train['target'].sum()):,} go-arounds ({rate:.2f} %)")

    print(f"  Train: {len(train):,}  Valid: {len(valid):,}  Test: {len(test):,}")

    X_tr, y_tr, num_feats, cat_feats = build_feature_matrix(train, feature_set)
    X_va, y_va, _, _ = build_feature_matrix(valid, feature_set)
    X_te, y_te, _, _ = build_feature_matrix(test, feature_set)

    print(f"  Features: {len(num_feats)} numeric + {len(cat_feats)} categorical  "
          f"[{time.time() - t0:.1f}s]")
    return X_tr, y_tr, X_va, y_va, X_te, y_te, num_feats, cat_feats


# --------------------------------------------------------------------------- #
# Preprocessing                                                               #
# --------------------------------------------------------------------------- #

def create_preprocessor(numeric_features: list[str], categorical_features: list[str],
                        ohe_min_frequency: int = 2000) -> ColumnTransformer:
    """
    ohe_min_frequency: a category must appear at least this many times in the
    training set to get its own OHE column (others → 'infrequent_sklearn').
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
# Calibration                                                                 #
# --------------------------------------------------------------------------- #

class PriorShiftCorrector:
    """
    Correct probabilities for the prevalence shift introduced by negative
    undersampling.

    If the training prior is π_train but the deployment prior is π_test,
    Bayes-optimal scores must be divided by the likelihood ratio of the priors:

        p_test = (π_test * p_train) /
                 (π_test * p_train + (1 - π_test) * (1 - p_train) * r),
        with r = π_train (1 - π_test) / ((1 - π_train) π_test).

    Equivalently, in log-odds: logit(p_test) = logit(p_train) - log(r).
    """

    def __init__(self, prior_train: float, prior_test: float):
        self.prior_train = float(prior_train)
        self.prior_test = float(prior_test)
        self.log_ratio = (
            np.log(self.prior_train / max(1 - self.prior_train, 1e-12))
            - np.log(self.prior_test / max(1 - self.prior_test, 1e-12))
        )

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-9, 1 - 1e-9)
        logits = np.log(p / (1 - p)) - self.log_ratio
        return 1.0 / (1.0 + np.exp(-logits))


def fit_isotonic_calibration(p_val: np.ndarray, y_val: np.ndarray) -> IsotonicRegression:
    """Fit an isotonic calibrator on a held-out (natural-prevalence) set."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_val, y_val)
    return iso


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
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


def split_validation_for_calibration(
    y_va: pd.Series, p_va: np.ndarray, cal_frac: float = 0.5, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split validation into (calibration, threshold-tuning) halves so that the
    isotonic fit and the F1-optimal threshold do not both fit the same indices.
    Returns (p_cal, y_cal, p_thr, y_thr).
    """
    rng = np.random.default_rng(random_state)
    n = len(y_va)
    idx = rng.permutation(n)
    cut = int(n * cal_frac)
    cal_idx, thr_idx = idx[:cut], idx[cut:]
    return p_va[cal_idx], y_va.values[cal_idx], p_va[thr_idx], y_va.values[thr_idx]


def calibrate_and_score(
    p_va: np.ndarray, y_va: pd.Series,
    p_te: np.ndarray, y_te: pd.Series,
    train_prior: float | None = None,
    test_prior: float | None = None,
) -> dict[str, Any]:
    """
    End-to-end calibration + threshold tuning + evaluation.

    1. (optional) prior-shift correct the raw scores (closed-form).
    2. Fit isotonic on half of the validation set; tune threshold on the other.
    3. Apply the calibrator to test scores and evaluate at the tuned threshold.

    Returns a dict with calibrator, threshold, and val+test metrics.
    """
    p_va_raw, p_te_raw = p_va, p_te
    if train_prior is not None and test_prior is not None and abs(train_prior - test_prior) > 1e-4:
        shifter = PriorShiftCorrector(train_prior, test_prior)
        p_va = shifter.transform(p_va_raw)
        p_te = shifter.transform(p_te_raw)

    p_cal, y_cal, p_thr, y_thr = split_validation_for_calibration(y_va, p_va)
    iso = fit_isotonic_calibration(p_cal, y_cal)

    p_thr_cal = iso.transform(p_thr)
    p_va_cal = iso.transform(p_va)
    p_te_cal = iso.transform(p_te)

    best_thresh = tune_threshold_for_f1(y_thr, p_thr_cal)

    out = {"best_threshold": float(best_thresh)}
    out.update(evaluate_binary_classifier(y_va.values, p_va_cal, threshold=best_thresh, split_name="validation"))
    out.update(evaluate_binary_classifier(y_te.values, p_te_cal, threshold=best_thresh, split_name="test"))
    out["_calibrator"] = iso
    out["_prior_train"] = float(train_prior) if train_prior is not None else None
    out["_prior_test"] = float(test_prior) if test_prior is not None else None
    return out


# --------------------------------------------------------------------------- #
# Serialisation                                                               #
# --------------------------------------------------------------------------- #

def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Remove non-serialisable internals
    clean = {k: v for k, v in metrics.items() if not k.startswith("_")}
    with open(path, "w") as fh:
        json.dump(clean, fh, indent=2)
    print(f"  Metrics saved → {path.name}")


def save_model_bundle(
    model: Any,
    preprocessor: Any,
    feature_schema: dict,
    path: Path,
    calibrator: Any | None = None,
    prior_train: float | None = None,
    prior_test: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model":          model,
        "preprocessor":   preprocessor,
        "feature_schema": feature_schema,
        "calibrator":     calibrator,
        "prior_train":    prior_train,
        "prior_test":     prior_test,
    }
    joblib.dump(bundle, path)
    print(f"  Model bundle saved → {path.name}")


def load_model_bundle(path: Path) -> dict:
    return joblib.load(path)


def apply_calibration(
    p_raw: np.ndarray,
    calibrator: Any | None,
    prior_train: float | None,
    prior_test: float | None,
) -> np.ndarray:
    """Apply optional prior-shift + isotonic calibration to raw probabilities."""
    p = np.asarray(p_raw, dtype=float)
    if prior_train is not None and prior_test is not None and abs(prior_train - prior_test) > 1e-4:
        p = PriorShiftCorrector(prior_train, prior_test).transform(p)
    if calibrator is not None:
        p = calibrator.transform(p)
    return p
