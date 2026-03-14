import numpy as np
import pandas as pd


def create_features(df, target_col="PJME_MW"):
    """
    Create time-based, cyclical, lag, and rolling features.
    """
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a pandas DatetimeIndex.")

    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in dataframe.")

    # calendar features
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    # cyclical features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    # lag features
    df["lag_1"] = df[target_col].shift(1)
    df["lag_24"] = df[target_col].shift(24)
    df["lag_168"] = df[target_col].shift(168)

    # rolling features
    shifted = df[target_col].shift(1)
    df["rolling_mean_24"] = shifted.rolling(window=24).mean()
    df["rolling_std_24"] = shifted.rolling(window=24).std()

    return df.dropna().copy()
