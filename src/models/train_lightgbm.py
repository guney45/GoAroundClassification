"""Train LightGBM (gradient-boosted trees) for go-around classification."""
import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import MODELS_DIR, METRICS_DIR, FIGURES_DIR
from src.models.common import (
    calibrate_and_score, create_preprocessor,
    load_splits, save_metrics, save_model_bundle,
)

FEATURE_SETS = ["context_only", "context_metar", "full"]


class LGBMSklearnWrapper(BaseEstimator, ClassifierMixin):
    """Thin sklearn-compatible wrapper around lgb.Booster."""

    def __init__(self, **params):
        self.params = params
        self.booster_ = None
        self.classes_ = np.array([0, 1])

    def fit(self, X, y, eval_set=None):
        pos = int(np.sum(y))
        neg = len(y) - pos
        scale_pos_weight = neg / max(pos, 1)
        params = {
            "objective":         "binary",
            "metric":            "average_precision",
            "scale_pos_weight":  scale_pos_weight,
            "learning_rate":     0.05,
            "num_leaves":        63,
            "min_child_samples": 50,
            "feature_fraction":  0.9,
            "bagging_fraction":  0.9,
            "bagging_freq":      5,
            "verbose":           -1,
        }
        params.update(self.params)
        train_data = lgb.Dataset(X, label=y)
        callbacks = [
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=50),
        ]
        valid_sets = [train_data]
        if eval_set is not None:
            X_va, y_va = eval_set
            valid_data = lgb.Dataset(X_va, label=y_va, reference=train_data)
            valid_sets = [train_data, valid_data]
        self.booster_ = lgb.train(
            params, train_data,
            num_boost_round=1500,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )
        return self

    def predict_proba(self, X):
        prob1 = self.booster_.predict(X)
        return np.column_stack([1 - prob1, prob1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train_lightgbm(
    feature_set: str = "full",
    sample_frac: float | None = None,
    neg_ratio: int | None = None,
) -> dict:
    print(f"\n=== LightGBM [{feature_set}] ===")
    X_tr, y_tr, X_va, y_va, X_te, y_te, num_feats, cat_feats = load_splits(
        feature_set, sample_frac, neg_ratio
    )

    preprocessor = create_preprocessor(num_feats, cat_feats)

    print("  Preprocessing (fit+transform train) ...")
    t0 = time.time()
    X_tr_pre = preprocessor.fit_transform(X_tr, y_tr)
    X_va_pre = preprocessor.transform(X_va)
    X_te_pre = preprocessor.transform(X_te)
    print(f"  Preprocessed: train{X_tr_pre.shape}  [{time.time() - t0:.1f}s]")

    clf = LGBMSklearnWrapper()
    print("  Training LightGBM (PR-AUC early stopping at 50) ...")
    t0 = time.time()
    clf.fit(X_tr_pre, y_tr.values, eval_set=(X_va_pre, y_va.values))
    print(f"  Training done [{time.time() - t0:.1f}s]")

    va_prob = clf.predict_proba(X_va_pre)[:, 1]
    te_prob = clf.predict_proba(X_te_pre)[:, 1]

    train_prior = float(y_tr.mean())
    test_prior = float(y_te.mean())
    scored = calibrate_and_score(
        va_prob, y_va, te_prob, y_te,
        train_prior=train_prior, test_prior=test_prior,
    )

    metrics = {"model": "lightgbm", "feature_set": feature_set,
               "best_threshold": scored["best_threshold"]}
    metrics.update({k: v for k, v in scored.items() if not k.startswith("_")})

    key = f"lightgbm_{feature_set}"
    save_metrics(metrics, METRICS_DIR / f"{key}.json")
    schema = {"numeric_features": num_feats, "categorical_features": cat_feats,
              "feature_set": feature_set}
    save_model_bundle(
        clf, preprocessor, schema, MODELS_DIR / f"{key}.joblib",
        calibrator=scored["_calibrator"],
        prior_train=train_prior, prior_test=test_prior,
    )

    # Feature importance
    try:
        booster = clf.booster_
        ohe_names = preprocessor.named_transformers_["cat"].named_steps[
            "encoder"
        ].get_feature_names_out(cat_feats).tolist()
        feat_names = num_feats + ohe_names
        importances = booster.feature_importance(importance_type="gain")
        if len(feat_names) == len(importances):
            top_n = min(25, len(feat_names))
            idx = np.argsort(importances)[-top_n:][::-1]
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.barh([feat_names[i] for i in reversed(idx)], importances[idx[::-1]])
            ax.set_title(f"LightGBM Feature Importance (gain) [{feature_set}]")
            ax.set_xlabel("Gain")
            plt.tight_layout()
            fig.savefig(FIGURES_DIR / f"lightgbm_feature_importance_{feature_set}.png", dpi=100)
            plt.close(fig)
    except Exception as e:
        print(f"  Feature importance plot skipped: {e}")

    print(f"  Val  ROC-AUC={metrics['validation_roc_auc']:.4f}  "
          f"PR-AUC={metrics['validation_average_precision']:.4f}  "
          f"F1={metrics['validation_f1']:.4f}")
    print(f"  Test ROC-AUC={metrics['test_roc_auc']:.4f}  "
          f"PR-AUC={metrics['test_average_precision']:.4f}  "
          f"F1={metrics['test_f1']:.4f}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-frac", type=float, default=None)
    parser.add_argument("--neg-ratio", type=int, default=None,
                        help="Optional stratified-by-airport undersampling.")
    parser.add_argument("--feature-sets", nargs="*", default=FEATURE_SETS)
    args = parser.parse_args()
    for fs in args.feature_sets:
        train_lightgbm(fs, args.sample_frac, args.neg_ratio)


if __name__ == "__main__":
    main()
