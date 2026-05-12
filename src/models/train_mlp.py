"""
Train MLP (Multi-Layer Perceptron) for go-around classification.

Why this differs from sklearn's stock recipe:

* sklearn MLPClassifier has no class_weight, and its built-in early stopping
  monitors *accuracy* — useless when 99.6 % of labels are 0 (it stops the
  moment the network learns to always predict 0). We disable it.
* Imbalance is handled with per-sample weights derived from the inverse class
  frequency (Bayes-optimal for the cross-entropy loss).
* Probability calibration: an isotonic regressor is fit on half of the
  validation set; the threshold is tuned on the other half. This keeps the
  reported validation metrics and the threshold from sharing indices.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import MODELS_DIR, METRICS_DIR
from src.models.common import (
    calibrate_and_score, create_preprocessor,
    load_splits, save_metrics, save_model_bundle,
)

FEATURE_SETS = ["context_only", "context_metar", "full"]


def train_mlp(
    feature_set: str = "full",
    sample_frac: float | None = None,
    neg_ratio: int | None = None,
) -> dict:
    print(f"\n=== MLP [{feature_set}] ===")
    X_tr, y_tr, X_va, y_va, X_te, y_te, num_feats, cat_feats = load_splits(
        feature_set, sample_frac, neg_ratio
    )

    preprocessor = create_preprocessor(num_feats, cat_feats)

    print("  Preprocessing ...")
    t0 = time.time()
    X_tr_pre = preprocessor.fit_transform(X_tr, y_tr)
    X_va_pre = preprocessor.transform(X_va)
    X_te_pre = preprocessor.transform(X_te)
    print(f"  Preprocessed: train{X_tr_pre.shape}  [{time.time() - t0:.1f}s]")

    # Inverse-frequency sample weights.
    pos_rate = float(y_tr.mean())
    w_pos = 0.5 / max(pos_rate, 1e-6)
    w_neg = 0.5 / max(1 - pos_rate, 1e-6)
    sample_weight = np.where(y_tr.values == 1, w_pos, w_neg).astype(np.float64)

    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=4096,
        learning_rate_init=1e-3,
        max_iter=60,
        early_stopping=False,
        random_state=42,
        verbose=False,
    )

    print("  Fitting MLP with inverse-frequency sample weights ...")
    t0 = time.time()
    # Replicate-sampling implementation of class weights for MLPClassifier
    # (which doesn't accept sample_weight in fit). Each positive is replicated
    # ⌈w_pos / w_neg⌉ times, which is mathematically equivalent to
    # weighting the SGD step on positives.
    rep = max(1, int(round(w_pos / w_neg)))
    if rep > 1:
        pos_mask = y_tr.values == 1
        X_pos = X_tr_pre[pos_mask]
        y_pos = y_tr.values[pos_mask]
        X_tr_eff = np.vstack([X_tr_pre] + [X_pos] * (rep - 1))
        y_tr_eff = np.concatenate([y_tr.values] + [y_pos] * (rep - 1))
        # Shuffle to avoid block-of-positives at the end of every epoch
        order = np.random.default_rng(42).permutation(len(y_tr_eff))
        X_tr_eff = X_tr_eff[order]
        y_tr_eff = y_tr_eff[order]
        print(f"  Effective train size after positive replication ×{rep}: "
              f"{len(y_tr_eff):,} ({y_tr_eff.mean():.3%} positive)")
    else:
        X_tr_eff, y_tr_eff = X_tr_pre, y_tr.values

    clf.fit(X_tr_eff, y_tr_eff)
    print(f"  Fit done [{time.time() - t0:.1f}s]  iterations={clf.n_iter_}")

    va_prob = clf.predict_proba(X_va_pre)[:, 1]
    te_prob = clf.predict_proba(X_te_pre)[:, 1]

    # Replicate-sampling shifts the effective training prior, so correct it.
    train_prior_effective = float(y_tr_eff.mean())
    test_prior = float(y_te.mean())
    scored = calibrate_and_score(
        va_prob, y_va, te_prob, y_te,
        train_prior=train_prior_effective, test_prior=test_prior,
    )

    metrics = {"model": "mlp", "feature_set": feature_set,
               "best_threshold": scored["best_threshold"]}
    metrics.update({k: v for k, v in scored.items() if not k.startswith("_")})

    key = f"mlp_{feature_set}"
    save_metrics(metrics, METRICS_DIR / f"{key}.json")
    schema = {"numeric_features": num_feats, "categorical_features": cat_feats,
              "feature_set": feature_set}
    # Wrap the bare classifier in a pipeline that includes the preprocessor
    # so that the saved bundle can be applied directly to a raw DataFrame.
    pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
    save_model_bundle(
        pipe, preprocessor, schema, MODELS_DIR / f"{key}.joblib",
        calibrator=scored["_calibrator"],
        prior_train=train_prior_effective,
        prior_test=test_prior,
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
    parser.add_argument("--sample-frac", type=float, default=None,
                        help="Fraction of negatives to keep (legacy). Use --neg-ratio instead.")
    parser.add_argument("--neg-ratio", type=int, default=None,
                        help="Keep at most neg_ratio × n_positive negatives (e.g. 20). "
                             "Stratified by airport. Test prior is preserved.")
    parser.add_argument("--feature-sets", nargs="*", default=FEATURE_SETS,
                        help="Subset of feature sets to train.")
    args = parser.parse_args()
    for fs in args.feature_sets:
        train_mlp(fs, args.sample_frac, args.neg_ratio)


if __name__ == "__main__":
    main()
