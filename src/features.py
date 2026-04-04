import numpy as np
import pandas as pd


def create_features(df, target_col="PJME_MW", use_lag1=True):
    """
    Create time-based, cyclical, lag, and rolling features.
    
    Parameters:
    - use_lag1: if False, exclude lag_1 to simulate operational delay
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

    # lag features (energy)
    if use_lag1:
        df["lag_1"] = df[target_col].shift(1)

    df["lag_3"] = df[target_col].shift(3)
    df["lag_24"] = df[target_col].shift(24)
    df["lag_168"] = df[target_col].shift(168)

    # rolling features (energy)
    shifted = df[target_col].shift(1)
    df["rolling_mean_24"] = shifted.rolling(window=24).mean()
    df["rolling_std_24"] = shifted.rolling(window=24).std()

    # weather features
    if "temperature" in df.columns:

        df["temp_lag_1"] = df["temperature"].shift(1)
        df["temp_lag_24"] = df["temperature"].shift(24)
        df["temp_roll_mean_24"] = df["temperature"].rolling(24).mean()

        df["heating_degree"] = (18 - df["temperature"]).clip(lower=0)
        df["cooling_degree"] = (df["temperature"] - 18).clip(lower=0)

    return df.dropna().copy()
