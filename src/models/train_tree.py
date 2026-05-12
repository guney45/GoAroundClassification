"""Train Random Forest classifier for go-around classification."""
import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import MODELS_DIR, METRICS_DIR, FIGURES_DIR
from src.models.common import (
    create_preprocessor, evaluate_binary_classifier,
    load_splits, save_metrics, save_model_bundle, tune_threshold_for_f1,
)

FEATURE_SETS = ["context_only", "context_metar"]


def train_tree(
    feature_set: str = "context_metar",
    n_estimators: int = 200,
    max_depth: int | None = 15,
    min_samples_leaf: int = 50,
    sample_frac: float | None = None,
    neg_ratio: int | None = None,
) -> dict:
    print(f"\n=== Random Forest [{feature_set}] ===")
    X_tr, y_tr, X_va, y_va, X_te, y_te, num_feats, cat_feats = load_splits(
        feature_set, sample_frac, neg_ratio
    )

    preprocessor = create_preprocessor(num_feats, cat_feats)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    pipe = Pipeline([("pre", preprocessor), ("clf", clf)])

    print(f"  Preprocessing ...")
    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    print(f"  Fit done [{time.time()-t0:.1f}s]")

    print("  Predicting on validation ...")
    va_prob = pipe.predict_proba(X_va)[:, 1]
    print("  Predicting on test ...")
    te_prob = pipe.predict_proba(X_te)[:, 1]
    best_thresh = tune_threshold_for_f1(y_va.values, va_prob)

    metrics = {"model": "random_forest", "feature_set": feature_set, "best_threshold": best_thresh}
    metrics.update(evaluate_binary_classifier(y_va.values, va_prob, threshold=0.5,         split_name="validation"))
    metrics.update(evaluate_binary_classifier(y_te.values, te_prob, threshold=best_thresh, split_name="test"))

    key = f"random_forest_{feature_set}"
    save_metrics(metrics, METRICS_DIR / f"{key}.json")
    schema = {"numeric_features": num_feats, "categorical_features": cat_feats, "feature_set": feature_set}
    save_model_bundle(pipe, preprocessor, schema, MODELS_DIR / f"{key}.joblib")

    # Feature importance plot
    try:
        rf = pipe.named_steps["clf"]
        pre = pipe.named_steps["pre"]
        ohe_names = pre.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(cat_feats).tolist()
        feat_names = num_feats + ohe_names
        importances = rf.feature_importances_
        top_n = 20
        idx = np.argsort(importances)[-top_n:][::-1]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh([feat_names[i] for i in reversed(idx)], importances[idx[::-1]])
        ax.set_title(f"RF Feature Importance [{feature_set}]")
        ax.set_xlabel("Mean Decrease in Impurity")
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / f"rf_feature_importance_{feature_set}.png", dpi=100)
        plt.close(fig)
    except Exception as e:
        print(f"  Feature importance plot skipped: {e}")

    print(f"  Val  ROC-AUC={metrics['validation_roc_auc']:.4f}  PR-AUC={metrics['validation_average_precision']:.4f}")
    print(f"  Test ROC-AUC={metrics['test_roc_auc']:.4f}  PR-AUC={metrics['test_average_precision']:.4f}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-frac", type=float, default=None,
                        help="Fraction of negatives to keep (legacy). Use --neg-ratio instead.")
    parser.add_argument("--neg-ratio", type=int, default=None,
                        help="Keep at most neg_ratio × n_positive negatives (e.g. 10).")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=15)
    args = parser.parse_args()
    for fs in FEATURE_SETS:
        train_tree(fs, n_estimators=args.n_estimators, max_depth=args.max_depth,
                   sample_frac=args.sample_frac, neg_ratio=args.neg_ratio)


if __name__ == "__main__":
    main()
