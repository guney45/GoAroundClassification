"""Feature engineering for go-around classification."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# --------------------------------------------------------------------------- #
# Feature groups                                                               #
# --------------------------------------------------------------------------- #

NUMERIC_CONTEXT = [
    # n_approaches and n_rwy_approached are EXCLUDED: they encode the number of
    # approaches a flight ultimately made, which is post-hoc information that
    # leaks the target (a go-around adds an extra approach → n_approaches >= 2).
    # They are used only for filtering calibration flights in clean_data().
    "glide_slope_angle",
    "rwy_length",
    "month",
    "day_of_week",
    "hour_utc",
]

# ADS-B approach-dynamics features (proposal §3, "rolling statistics from the
# final approach segment"). Each value is a per-landing aggregate computed from
# the trajectory ending at touchdown / go-around initiation. Available in the
# enriched augmented file produced by src/data/generate_synthetic_data.py and
# in any pipeline that derives them from raw OpenSky ADS-B states.
NUMERIC_TRAJECTORY = [
    "ias_at_1000ft_kts",
    "ias_at_500ft_kts",
    "vrate_std_5nm_fpm",
    "vrate_mean_5nm_fpm",
    "alt_at_gate_ft",
    "lat_dev_1nm_nm",
    "hdg_change_rate_dps",
    "rwy_align_err_deg",
    "gs_range_5nm_kts",
    "prev_arr_gap_min",
    "approach_dur_s",
    "config_late_score",
]

NUMERIC_METAR = [
    "wind_speed_knts",
    "wind_dir_deg",
    "wind_gust_knts",
    "visibility_m",
    "temperature_deg",
    "press_sea_level_p",
    "press_p",
]

CATEGORICAL_CONTEXT = [
    "airport",
    "runway",
    "typecode",
    "icaoaircrafttype",
    "wtc",
    "has_intersection",
    "airport_country",
    "airport_region",
    "operator_country",
    "operator_region",
]

CATEGORICAL_METAR = [
    "weather_intensity",
    "weather_precipitation",
    "weather_desc",
    "weather_obscuration",
    "weather_other",
]

# Columns that must never be model features (leakage / IDs)
DROP_COLS = {"has_ga", "icao24", "callsign", "registration", "time"}


def get_feature_groups() -> dict:
    return {
        "numeric_context":     NUMERIC_CONTEXT,
        "numeric_trajectory":  NUMERIC_TRAJECTORY,
        "numeric_metar":       NUMERIC_METAR,
        "categorical_context": CATEGORICAL_CONTEXT,
        "categorical_metar":   CATEGORICAL_METAR,
    }


def load_data(path: str | Path) -> pd.DataFrame:
    df = pl.read_parquet(path).to_pandas()
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "time" in df.columns:
        t = pd.to_datetime(df["time"], errors="coerce")
        df = df.copy()
        df["month"]       = t.dt.month.astype("float32")
        df["day_of_week"] = t.dt.dayofweek.astype("float32")
        df["hour_utc"]    = t.dt.hour.astype("float32")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Require target
    if "target" not in df.columns and "has_ga" in df.columns:
        df["target"] = (
            df["has_ga"].astype(str).str.lower()
            .map({"true": 1, "1": 1, "false": 0, "0": 0})
            .astype("Int8")
        )
    df = df[df["target"].notna()].copy()
    df["target"] = df["target"].astype(int)

    # Filter likely training/calibration flights
    if "n_approaches" in df.columns:
        df = df[pd.to_numeric(df["n_approaches"], errors="coerce").fillna(1) <= 2]

    # Fill missing categoricals
    cat_cols = CATEGORICAL_CONTEXT + CATEGORICAL_METAR
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna("UNKNOWN").astype(str)

    # Numeric missings stay as NaN for sklearn imputers

    # Drop identifier columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    return df


def build_feature_matrix(
    df: pd.DataFrame,
    feature_set: Literal["context_only", "context_metar", "full"] = "full",
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Return X, y, numeric_features, categorical_features.

    Feature sets
    ------------
    context_only   : airport / runway / aircraft / time only.
    context_metar  : context + METAR weather (legacy ablation set).
    full           : context + METAR + ADS-B trajectory dynamics (proposal §3).
                     This is the recommended set when trajectory columns are
                     present in the input dataframe — without it the model is
                     limited to weak landing-level aggregates.
    """
    groups = get_feature_groups()

    num_feats = groups["numeric_context"].copy()
    cat_feats = groups["categorical_context"].copy()

    if feature_set in ("context_metar", "full"):
        num_feats += groups["numeric_metar"]
        cat_feats += groups["categorical_metar"]
    if feature_set == "full":
        num_feats += groups["numeric_trajectory"]

    # Keep only columns that actually exist
    num_feats = [c for c in num_feats if c in df.columns]
    cat_feats = [c for c in cat_feats if c in df.columns]

    y = df["target"].astype(int)
    X = df[num_feats + cat_feats].copy()

    return X, y, num_feats, cat_feats
