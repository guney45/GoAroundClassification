"""
Train Linear Discriminant Analysis (LDA) for go-around classification.

LDA assumption: x | y=k ~ N(μ_k, Σ) with shared covariance Σ.
Decision rule:  ŷ = argmax_k  δ_k(x),
where δ_k(x) = xᵀΣ⁻¹μ_k - ½μ_kᵀΣ⁻¹μ_k + log π_k.
"""
import argparse
import sys
import time
from pathlib import Path

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import MODELS_DIR, METRICS_DIR
from src.models.common import (
    create_preprocessor, evaluate_binary_classifier,
    load_splits, save_metrics, save_model_bundle, tune_threshold_for_f1,
)

FEATURE_SETS = ["context_only", "context_metar"]


def train_lda(
    feature_set: str = "context_metar",
    sample_frac: float | None = None,
    neg_ratio: int | None = None,
) -> dict:
    print(f"\n=== LDA [{feature_set}] ===")
    X_tr, y_tr, X_va, y_va, X_te, y_te, num_feats, cat_feats = load_splits(
        feature_set, sample_frac, neg_ratio
    )

    preprocessor = create_preprocessor(num_feats, cat_feats)

    clf = LinearDiscriminantAnalysis(solver="svd", n_components=1)
    pipe = Pipeline([("pre", preprocessor), ("clf", clf)])

    print(f"  Fitting LDA ...")
    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    print(f"  Fit done [{time.time()-t0:.1f}s]")

    print("  Predicting on validation ...")
    va_prob = pipe.predict_proba(X_va)[:, 1]
    print("  Predicting on test ...")
    te_prob = pipe.predict_proba(X_te)[:, 1]

    best_thresh = tune_threshold_for_f1(y_va.values, va_prob)

    metrics = {"model": "lda", "feature_set": feature_set, "best_threshold": best_thresh}
    metrics.update(evaluate_binary_classifier(y_va.values, va_prob, threshold=0.5,          split_name="validation"))
    metrics.update(evaluate_binary_classifier(y_te.values, te_prob, threshold=best_thresh,  split_name="test"))

    key = f"lda_{feature_set}"
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
        train_lda(fs, args.sample_frac, args.neg_ratio)


if __name__ == "__main__":
    main()
