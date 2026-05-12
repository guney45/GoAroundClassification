"""Create time-based train/validation/test splits."""
import argparse
import sys
from pathlib import Path

import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import AUGMENTED_PARQUET, TRAIN_PARQUET, VALID_PARQUET, TEST_PARQUET, DATA_PROCESSED
from src.features.build_features import load_data, clean_data, add_time_features


def _print_split_stats(name: str, df: pd.DataFrame) -> None:
    n = len(df)
    pos = int(df["target"].sum()) if "target" in df.columns else 0
    rate = 100.0 * pos / n if n else 0.0
    print(f"  {name}: {n:>10,} rows | {pos:>7,} go-arounds ({rate:.3f}%)")


def make_splits(sample_frac: float | None = None, top_airports: int | None = None) -> None:
    print(f"Loading {AUGMENTED_PARQUET.name} ...")
    raw = load_data(AUGMENTED_PARQUET)
    print(f"  Loaded {len(raw):,} rows, {raw.shape[1]} cols")

    if sample_frac is not None and sample_frac < 1.0:
        raw = raw.sample(frac=sample_frac, random_state=42)
        print(f"  Sampled {len(raw):,} rows (frac={sample_frac})")

    # Preserve raw aircraft id BEFORE clean_data() drops it — needed for the
    # group check below to prevent same-aircraft leakage across splits.
    icao24_col = raw["icao24"].astype(str).copy() if "icao24" in raw.columns else None
    time_col = pd.to_datetime(raw["time"], errors="coerce") if "time" in raw.columns else None

    raw = add_time_features(raw)
    raw = clean_data(raw)

    if time_col is not None:
        raw["_split_time"] = time_col.reindex(raw.index).values
    if icao24_col is not None:
        raw["_icao24"] = icao24_col.reindex(raw.index).values

    if top_airports is not None and "airport" in raw.columns:
        top = (
            raw[raw["target"] == 1]["airport"]
            .value_counts()
            .head(top_airports)
            .index
        )
        raw = raw[raw["airport"].isin(top)]
        print(f"  Filtered to top {top_airports} airports: {len(raw):,} rows")

    if "_split_time" in raw.columns and raw["_split_time"].notna().any():
        # Time-based split (proposal §4: "airport-aware and/or time-aware").
        # Quantile boundaries make this robust to arbitrary dataset windows.
        t = pd.to_datetime(raw["_split_time"], errors="coerce")
        t_train_end = t.quantile(0.70)
        t_valid_end = t.quantile(0.85)
        train = raw[t <= t_train_end]
        valid = raw[(t > t_train_end) & (t <= t_valid_end)]
        test  = raw[t > t_valid_end]
        print(f"  Time split: train≤{t_train_end}  valid≤{t_valid_end}  test>{t_valid_end}")
    else:
        # Fallback: group split by icao24 so the same airframe never appears
        # in both train and test (prevents identity-style leakage).
        print("  No time column — using grouped split by icao24 (70/15/15)")
        from sklearn.model_selection import GroupShuffleSplit
        groups = raw.get("_icao24", pd.Series(range(len(raw)))).astype(str).values
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
        tr_idx, tmp_idx = next(gss1.split(raw, groups=groups))
        train, tmp = raw.iloc[tr_idx], raw.iloc[tmp_idx]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
        va_idx, te_idx = next(gss2.split(tmp, groups=tmp["_icao24"].astype(str).values
                                          if "_icao24" in tmp.columns else range(len(tmp))))
        valid, test = tmp.iloc[va_idx], tmp.iloc[te_idx]

    for split in (train, valid, test):
        for c in ("_split_time", "_icao24"):
            if c in split.columns:
                split.drop(columns=[c], inplace=True)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    pl.from_pandas(train.reset_index(drop=True)).write_parquet(TRAIN_PARQUET)
    pl.from_pandas(valid.reset_index(drop=True)).write_parquet(VALID_PARQUET)
    pl.from_pandas(test.reset_index(drop=True)).write_parquet(TEST_PARQUET)

    print("Split statistics:")
    for name, split in [("train", train), ("valid", valid), ("test", test)]:
        _print_split_stats(name, split)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/valid/test splits")
    parser.add_argument("--sample-frac", type=float, default=None)
    parser.add_argument("--top-airports", type=int, default=None)
    args = parser.parse_args()
    make_splits(sample_frac=args.sample_frac, top_airports=args.top_airports)
    print("Splits saved to data/processed/.")


if __name__ == "__main__":
    main()
