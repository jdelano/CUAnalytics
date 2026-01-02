"""
Scaling utilities for numeric features.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def scale_data(df, method='standard', columns=None, exclude_cols=None, scaler=None, skip_binary=True):
    """
    Scale numeric columns using a fitted scaler.

    If `scaler` is None, fits a new scaler on the provided data and returns it.
    If `scaler` is provided, applies it to the selected columns.

    Parameters:
    -----------
    df : pd.DataFrame
        Data to scale
    method : str
        Scaling method: 'standard' or 'minmax'
    columns : list[str] | None
        Columns to scale; defaults to numeric columns (excluding exclude_cols)
    exclude_cols : list[str] | None
        Columns to exclude from scaling
    scaler : object | None
        A fitted sklearn scaler to reuse
    skip_binary : bool
        If True, leave 0/1 columns unchanged

    Returns:
    --------
    scaled_df : pd.DataFrame
        Scaled data
    scaler : object
        Fitted scaler (StandardScaler or MinMaxScaler)
    """
    if columns is None:
        columns = list(df.select_dtypes(include='number').columns)
        if exclude_cols:
            columns = [col for col in columns if col not in exclude_cols]

    if skip_binary:
        columns = [
            col for col in columns
            if not set(df[col].dropna().unique()).issubset({0, 1})
        ]

    if method not in ('standard', 'minmax'):
        raise ValueError("method must be 'standard' or 'minmax'")

    if scaler is None:
        scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
        scaler.fit(df[columns])

    scaled_df = df.copy()
    scaled_df[columns] = scaler.transform(df[columns])
    return scaled_df, scaler
