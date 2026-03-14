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
