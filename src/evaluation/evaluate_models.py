"""Generate final evaluation plots for the best model on the test set."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay,
    auc, confusion_matrix, precision_recall_curve, roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import FINAL_MODEL, METRICS_DIR, FIGURES_DIR, FEATURE_SCHEMA
from src.models.common import (
    apply_calibration, evaluate_binary_classifier, load_model_bundle, load_splits,
)


def evaluate_final_model() -> None:
    print("Loading final model ...")
    bundle = load_model_bundle(FINAL_MODEL)
    model        = bundle["model"]
    preprocessor = bundle["preprocessor"]

    # Prefer the standalone JSON schema (contains model_name + best_threshold)
    if FEATURE_SCHEMA.exists():
        with open(FEATURE_SCHEMA) as fh:
            schema = json.load(fh)
    else:
        schema = bundle["feature_schema"]

    feature_set = schema.get("feature_set", "full")
    threshold   = schema.get("best_threshold", 0.5)

    print(f"  Model: {schema.get('model_name', 'unknown')}  [{feature_set}]")

    _, _, _, _, X_te, y_te, num_feats, cat_feats = load_splits(feature_set)

    from sklearn.pipeline import Pipeline as _Pipeline
    if isinstance(model, _Pipeline):
        y_prob_raw = model.predict_proba(X_te)[:, 1]
    else:
        try:
            X_pre = preprocessor.transform(X_te)
        except Exception:
            X_pre = X_te.values
        y_prob_raw = model.predict_proba(X_pre)[:, 1]
    y_prob = apply_calibration(
        y_prob_raw,
        bundle.get("calibrator"),
        bundle.get("prior_train"),
        bundle.get("prior_test"),
    )
    y_true = y_te.values

    # --- Metrics ---
    metrics = evaluate_binary_classifier(y_true, y_prob, threshold=threshold, split_name="test")
    metrics["model_name"]   = schema.get("model_name", "unknown")
    metrics["feature_set"]  = feature_set
    metrics["threshold"]    = threshold

    with open(METRICS_DIR / "final_test_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"  Test ROC-AUC={metrics['test_roc_auc']:.4f}  PR-AUC={metrics['test_average_precision']:.4f}")

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, (y_prob >= threshold).astype(int))
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal Landing", "Go-Around"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix (threshold={threshold:.2f})")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)
    print("  Saved confusion_matrix.png")

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc_val = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC AUC = {roc_auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curve.png", dpi=120)
    plt.close(fig)
    print("  Saved roc_curve.png")

    # --- Precision-Recall Curve ---
    prec, rec, thresholds_pr = precision_recall_curve(y_true, y_prob)
    pr_auc_val = auc(rec, prec)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2, label=f"PR AUC = {pr_auc_val:.4f}")
    ax.axhline(y=y_true.mean(), color="r", linestyle="--", lw=1, label="Baseline (prevalence)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "precision_recall_curve.png", dpi=120)
    plt.close(fig)
    print("  Saved precision_recall_curve.png")

    # --- Calibration Curve ---
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(mean_pred, frac_pos, "s-", label="Model")
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Calibration Curve")
        ax.legend()
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "calibration_curve.png", dpi=120)
        plt.close(fig)
        print("  Saved calibration_curve.png")
    except Exception as e:
        print(f"  Calibration plot skipped: {e}")

    # --- Threshold vs Precision/Recall/F1 ---
    try:
        ths = np.linspace(0.01, 0.99, 100)
        precisions, recalls, f1s = [], [], []
        for th in ths:
            yp = (y_prob >= th).astype(int)
            p = float(np.sum((yp == 1) & (y_true == 1))) / max(np.sum(yp == 1), 1)
            r = float(np.sum((yp == 1) & (y_true == 1))) / max(np.sum(y_true == 1), 1)
            f = 2 * p * r / max(p + r, 1e-9)
            precisions.append(p); recalls.append(r); f1s.append(f)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ths, precisions, label="Precision")
        ax.plot(ths, recalls,    label="Recall")
        ax.plot(ths, f1s,        label="F1")
        ax.axvline(threshold, color="k", linestyle="--", label=f"Selected threshold={threshold:.2f}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title("Threshold vs Precision / Recall / F1")
        ax.legend()
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "threshold_analysis.png", dpi=120)
        plt.close(fig)
        print("  Saved threshold_analysis.png")
    except Exception as e:
        print(f"  Threshold plot skipped: {e}")

    print("\nEvaluation complete. Figures saved to reports/figures/.")


def main() -> None:
    evaluate_final_model()


if __name__ == "__main__":
    main()
