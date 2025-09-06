import pandas as pd
import numpy as np
from joblib import Parallel, delayed

def trail(before_df: pd.DataFrame, transaction_row, run):
    """Run function only on prior rows for a single transaction."""
    return run(before_df, transaction_row)

def hound(df: pd.DataFrame, run=lambda d, t: (d, t)):
    """
    For each row in df, apply run() to all rows before it in time.
    Returns a list of results.
    """
    df_sorted = df.sort_values('time')  # Ensure transactions are time-ordered
    results = []

    # Keep a running 'history' slice without filtering entire df each time
    history = pd.DataFrame(columns=df.columns)

    for row in df_sorted.itertuples(index=False):
        # Call run on history
        results.append(run(history, row._asdict()))
        # Append current row to history
        new_row = pd.DataFrame([row._asdict()])
        if not new_row.dropna(how='all').empty:
            history = pd.concat([history, new_row], ignore_index=True)

    return results


def distance(latA, lonA, latB, lonB):
    # Radius of the earth
    R = 6371

    # Convert degrees to radians
    latA, lonA, latB, lonB = map(np.radians, [latA, lonA, latB, lonB])

    # Difference
    lat_diff = latB - latA
    lon_diff = lonB - lonA

    # Haversine formular
    a = np.sin(lat_diff / 2.0)**2 + np.cos(latA) * np.cos(latB) * np.sin(lon_diff / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def rolling_averages(df, group, features, window):
    df = df.sort_values([group, "date"]).copy()
    data = df[[group, *features]]

    """
        Add rolling averages grouped by `group` back to original DataFrame.

        @params df: The dataframe to use.
        @params group: How the dataframe should be grouped.
        @params features: The features to roll.
        @params window: Number of days to roll.

        @returns df: A dataframe containing the original dataframe and the rolled values.
    """
    columns = []
    for feature in features:
        rolled_col = f"{group}_{feature}_avg_{window}D"
        columns.append(rolled_col)
        df[rolled_col] = (
            data.groupby(group)[feature]
              .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
    
    return df[columns]


def extract_window(df, ref_date, window_size, inclusive="both"):
    """
    Extracts a dataset window ending at a given reference date.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset containing a datetime column.
    date_col : str
        The column name of the datetime column.
    ref_date : str or pd.Timestamp
        The reference date (e.g., '2025-08-28').
    window_size : int
        Number of days to include in the window.
    inclusive : {"both", "neither", "left", "right"}
        Whether the interval is inclusive of start/end (default "both").
        
    Returns:
    --------
    pd.DataFrame
        Subset of the dataframe within the date window.
    """
    ref_date = pd.to_datetime(ref_date)
    start_date = ref_date - pd.Timedelta(days=window_size-1)
    
    mask = df['date'].between(start_date, ref_date, inclusive=inclusive)
    return df.loc[mask].copy()