import numpy as np
import pandas as pd
import requests
from io import StringIO
import matplotlib.pyplot as plt

def calculate_entropy(data, target_col=None, split_col=None):
    """
    Calculate Shannon entropy.

    Parameters:
    -----------
    data : pd.Series | list | pd.DataFrame
        Series/list of labels, or a DataFrame when target_col is provided.
    target_col : str | None
        Column to calculate entropy for when data is a DataFrame.
    split_col : str | None
        When provided with a DataFrame, returns weighted entropy after splitting
        target_col by split_col.
    """
    if isinstance(data, pd.DataFrame):
        if target_col is None:
            raise ValueError("target_col is required when data is a DataFrame")
        if split_col is None:
            return calculate_entropy(data[target_col])
        n = len(data)
        weighted_entropy = 0.0
        for val in data[split_col].unique():
            child = data[data[split_col] == val][target_col]
            weight = len(child) / n
            weighted_entropy += weight * calculate_entropy(child)
        return weighted_entropy

    y = pd.Series(data)
    counts = y.value_counts()
    probabilities = counts / len(y)
    probabilities = probabilities[probabilities > 0]
    if len(probabilities) == 0:
        return 0.0
    # Handle p=0 case automatically (0*log(0) = 0 by convention)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def information_gain(df, feature, target_col='class'):
    """Calculate information gain from splitting on a feature."""
    parent_entropy = calculate_entropy(df, target_col=target_col)
    weighted_child_entropy = calculate_entropy(
        df, target_col=target_col, split_col=feature
    )
    return parent_entropy - weighted_child_entropy
