"""
Similarity and distance metrics.
"""

import numpy as np


def _to_array(x):
    return np.asarray(x, dtype=float)


def euclidean(a, b):
    """Compute Euclidean distance between two vectors."""
    a = _to_array(a)
    b = _to_array(b)
    return np.linalg.norm(a - b)


def manhattan(a, b):
    """Compute Manhattan (L1) distance between two vectors."""
    a = _to_array(a)
    b = _to_array(b)
    return np.sum(np.abs(a - b))


def cosine(a, b):
    """Compute cosine distance between two vectors."""
    a = _to_array(a)
    b = _to_array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return 1 - float(np.dot(a, b) / denom)


def jaccard(a, b):
    """Compute Jaccard distance between two binary vectors."""
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    intersection = np.logical_and(a, b).sum()
    return 1 - (intersection / union)
