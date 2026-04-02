import pandas as pd


def load_data(path, timestamp_col="timestamp"):
    """
    Load dataset from CSV file and parse timestamp column.
    """
    df = pd.read_csv(path)

    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": timestamp_col})

    if timestamp_col not in df.columns:
        raise ValueError(f"Column '{timestamp_col}' not found in dataset.")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    return df


def sort_by_time(df, timestamp_col="timestamp"):
    """
    Set timestamp column as index and sort chronologically.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Column '{timestamp_col}' not found in dataframe.")

    return df.set_index(timestamp_col).sort_index()


def run_data_checks(df, target_col="PJME_MW"):
    time_deltas = df.index.to_series().diff().value_counts().head()

    full_range = pd.date_range(df.index.min(), df.index.max(), freq="h")
    missing = len(full_range) - len(df)

    checks = {
        "sorted_index": df.index.is_monotonic_increasing,
        "duplicate_timestamps": int(df.index.duplicated().sum()),
        "missing_target_values": int(df[target_col].isna().sum()),
        "negative_target_values": int((df[target_col] < 0).sum()),
        "missing_timestamps": int(missing),
    }

    return checks, time_deltas