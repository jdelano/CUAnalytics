import numpy as np
import pandas as pd
import requests
from io import StringIO
import matplotlib.pyplot as plt

def calculate_entropy(y):
    """Calculate Shannon entropy"""
    counts = y.value_counts()
    probabilities = counts / len(y)
    # Handle p=0 case automatically (0*log(0) = 0 by convention)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return entropy

def information_gain(parent, children_list):
    """
    Calculate information gain from a split into multiple subsets
    
    Parameters:
    parent: array-like, the parent dataset (target values)
    children_list: list of array-like children subsets
    
    Returns:
    float: information gain
    """
    n = len(parent)
    parent_entropy = calculate_entropy(parent)
    
    # Calculate weighted average of children entropies
    weighted_child_entropy = 0
    for child in children_list:
        if len(child) > 0:  # Skip empty children
            weight = len(child) / n
            weighted_child_entropy += weight * calculate_entropy(child)
    
    return parent_entropy - weighted_child_entropy