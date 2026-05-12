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
    calibrate_and_score, create_preprocessor,
    load_splits, save_metrics, save_model_bundle,
)

FEATURE_SETS = ["context_only", "context_metar", "full"]


def train_lda(
    feature_set: str = "full",
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

    print("  Fitting LDA ...")
    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    print(f"  Fit done [{time.time() - t0:.1f}s]")

    va_prob = pipe.predict_proba(X_va)[:, 1]
    te_prob = pipe.predict_proba(X_te)[:, 1]

    train_prior = float(y_tr.mean())
    test_prior = float(y_te.mean())
    scored = calibrate_and_score(
        va_prob, y_va, te_prob, y_te,
        train_prior=train_prior, test_prior=test_prior,
    )

    metrics = {"model": "lda", "feature_set": feature_set,
               "best_threshold": scored["best_threshold"]}
    metrics.update({k: v for k, v in scored.items() if not k.startswith("_")})

    key = f"lda_{feature_set}"
    save_metrics(metrics, METRICS_DIR / f"{key}.json")
    schema = {"numeric_features": num_feats, "categorical_features": cat_feats,
              "feature_set": feature_set}
    save_model_bundle(
        pipe, preprocessor, schema, MODELS_DIR / f"{key}.joblib",
        calibrator=scored["_calibrator"],
        prior_train=train_prior, prior_test=test_prior,
    )

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
    parser.add_argument("--neg-ratio", type=int, default=None)
    parser.add_argument("--feature-sets", nargs="*", default=FEATURE_SETS)
    args = parser.parse_args()
    for fs in args.feature_sets:
        train_lda(fs, args.sample_frac, args.neg_ratio)


if __name__ == "__main__":
    main()
