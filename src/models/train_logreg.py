"""
Train Logistic Regression for go-around classification.

Probabilistic model:  p(y=1|x) = σ(wᵀx + b),  σ(z) = 1/(1+e⁻ᶻ)
Decision rule:        ŷ = 1  if  p(y=1|x) ≥ τ,  else 0.
Loss:                 L = -1/N Σ [y_i log p_i + (1-y_i) log(1-p_i)]
"""
import argparse
import sys
import time
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import MODELS_DIR, METRICS_DIR
from src.models.common import (
    create_preprocessor, evaluate_binary_classifier,
    load_splits, save_metrics, save_model_bundle, tune_threshold_for_f1,
)

FEATURE_SETS = ["context_only", "context_metar"]


def train_logreg(
    feature_set: str = "context_metar",
    sample_frac: float | None = None,
    neg_ratio: int | None = None,
) -> dict:
    print(f"\n=== Logistic Regression [{feature_set}] ===")
    X_tr, y_tr, X_va, y_va, X_te, y_te, num_feats, cat_feats = load_splits(
        feature_set, sample_frac, neg_ratio
    )

    preprocessor = create_preprocessor(num_feats, cat_feats)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs", verbose=0)
    pipe = Pipeline([("pre", preprocessor), ("clf", clf)])

    print(f"  Fitting Logistic Regression (solver=lbfgs) ...")
    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    print(f"  Fit done [{time.time()-t0:.1f}s]")

    print("  Predicting on validation ...")
    va_prob = pipe.predict_proba(X_va)[:, 1]
    print("  Predicting on test ...")
    te_prob = pipe.predict_proba(X_te)[:, 1]
    best_thresh = tune_threshold_for_f1(y_va.values, va_prob)

    metrics = {"model": "logreg", "feature_set": feature_set, "best_threshold": best_thresh}
    metrics.update(evaluate_binary_classifier(y_va.values, va_prob, threshold=0.5,         split_name="validation"))
    metrics.update(evaluate_binary_classifier(y_te.values, te_prob, threshold=best_thresh, split_name="test"))

    key = f"logreg_{feature_set}"
    save_metrics(metrics, METRICS_DIR / f"{key}.json")
    schema = {"numeric_features": num_feats, "categorical_features": cat_feats, "feature_set": feature_set}
    save_model_bundle(pipe, preprocessor, schema, MODELS_DIR / f"{key}.joblib")

    print(f"  Val  ROC-AUC={metrics['validation_roc_auc']:.4f}  PR-AUC={metrics['validation_average_precision']:.4f}")
    print(f"  Test ROC-AUC={metrics['test_roc_auc']:.4f}  PR-AUC={metrics['test_average_precision']:.4f}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-frac", type=float, default=None,
                        help="Fraction of negatives to keep (legacy). Use --neg-ratio instead.")
    parser.add_argument("--neg-ratio", type=int, default=None,
                        help="Keep at most neg_ratio × n_positive negatives (e.g. 10).")
    args = parser.parse_args()
    for fs in FEATURE_SETS:
        train_logreg(fs, args.sample_frac, args.neg_ratio)


if __name__ == "__main__":
    main()
