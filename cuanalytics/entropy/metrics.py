import numpy as np
import pandas as pd
import requests
from io import StringIO
import matplotlib.pyplot as plt

def calculate_entropy(y):
    """Calculate Shannon entropy"""
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
    n = len(df)
    parent_entropy = calculate_entropy(df[target_col])
    
    # Calculate weighted average of children entropies
    weighted_child_entropy = 0
    for val in df[feature].unique():
        child = df[df[feature] == val][target_col]
        weight = len(child) / n
        weighted_child_entropy += weight * calculate_entropy(child)
    
    return parent_entropy - weighted_child_entropy
