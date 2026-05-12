"""Command-line interface for the go-around classification pipeline."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Go-Around Classification CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  download-data       Download Zenodo dataset files to data/raw/
  verify-data         Check that data/raw/ files exist
  make-interim        Convert raw CSV.GZ files to Parquet
  make-splits         Create train/valid/test splits
  train-lda           Train Linear Discriminant Analysis
  train-logreg        Train Logistic Regression
  train-tree          Train Random Forest
  train-mlp           Train MLP (neural baseline)
  train-lightgbm      Train LightGBM
  select-best-model   Compare all models and pick best
  evaluate            Generate final evaluation plots
  error-analysis      Run error analysis on test set
  serve               Start the FastAPI development server
        """,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("download-data")
    sub.add_parser("verify-data")
    sub.add_parser("make-interim")

    sp = sub.add_parser("make-splits")
    sp.add_argument("--sample-frac", type=float, default=None)
    sp.add_argument("--top-airports", type=int, default=None)

    def _add_train_args(p):
        p.add_argument("--sample-frac", type=float, default=None,
                       help="Fraction of negatives to keep (legacy). Use --neg-ratio instead.")
        p.add_argument("--neg-ratio", type=int, default=None,
                       help="Keep at most neg_ratio × n_positive negatives (e.g. 10).")

    _add_train_args(sub.add_parser("train-lda"))
    _add_train_args(sub.add_parser("train-logreg"))
    sp_tree = sub.add_parser("train-tree")
    _add_train_args(sp_tree)
    sp_tree.add_argument("--n-estimators", type=int, default=200)
    sp_tree.add_argument("--max-depth", type=int, default=15)
    _add_train_args(sub.add_parser("train-mlp"))
    _add_train_args(sub.add_parser("train-lightgbm"))
    sub.add_parser("select-best-model")
    sub.add_parser("evaluate")
    sub.add_parser("error-analysis")
    sub.add_parser("serve")

    args = parser.parse_args()
    cmd  = args.command

    if cmd == "download-data":
        from src.data.download_data import main as fn; fn()
    elif cmd == "verify-data":
        from src.data.verify_local_data import main as fn; fn()
    elif cmd == "make-interim":
        from src.data.make_interim import main as fn; fn()
    elif cmd == "make-splits":
        from src.data.make_splits import make_splits
        make_splits(sample_frac=args.sample_frac, top_airports=args.top_airports)
    elif cmd == "train-lda":
        from src.models.train_lda import train_lda
        for fs in ["context_only", "context_metar"]:
            train_lda(fs, args.sample_frac, args.neg_ratio)
    elif cmd == "train-logreg":
        from src.models.train_logreg import train_logreg
        for fs in ["context_only", "context_metar"]:
            train_logreg(fs, args.sample_frac, args.neg_ratio)
    elif cmd == "train-tree":
        from src.models.train_tree import train_tree
        for fs in ["context_only", "context_metar"]:
            train_tree(fs, n_estimators=args.n_estimators, max_depth=args.max_depth,
                       sample_frac=args.sample_frac, neg_ratio=args.neg_ratio)
    elif cmd == "train-mlp":
        from src.models.train_mlp import train_mlp
        for fs in ["context_only", "context_metar"]:
            train_mlp(fs, args.sample_frac, args.neg_ratio)
    elif cmd == "train-lightgbm":
        from src.models.train_lightgbm import train_lightgbm
        for fs in ["context_only", "context_metar"]:
            train_lightgbm(fs, args.sample_frac, args.neg_ratio)
    elif cmd == "select-best-model":
        from src.models.select_best_model import main as fn; fn()
    elif cmd == "evaluate":
        from src.evaluation.evaluate_models import main as fn; fn()
    elif cmd == "error-analysis":
        from src.evaluation.error_analysis import main as fn; fn()
    elif cmd == "serve":
        import uvicorn
        uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
