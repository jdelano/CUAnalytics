"""
k-Nearest Neighbors models.
"""

from .classifier import fit_knn_classifier, KNNClassifierModel
from .regressor import fit_knn_regressor, KNNRegressorModel

__all__ = [
    'fit_knn_classifier',
    'KNNClassifierModel',
    'fit_knn_regressor',
    'KNNRegressorModel',
]
