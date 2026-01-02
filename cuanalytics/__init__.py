"""
ITM 4150: Advanced Business Analytics and Visualization
Python toolkit for course materials at Cedarville University.
"""

__version__ = "0.1.2"
__author__ = "Dr. John D. Delano"

# Import commonly used functions for convenient access
from cuanalytics.datasets.loaders import load_mushroom_data, load_iris_data, load_breast_cancer_data, load_real_estate_data
from cuanalytics.entropy.metrics import calculate_entropy, information_gain
from cuanalytics.entropy.visualization import plot_entropy
from cuanalytics.preprocessing.split import split_data 
from cuanalytics.trees.decision_tree import fit_tree, SimpleDecisionTree
from cuanalytics.lda.discriminant import fit_lda, LDAModel
from cuanalytics.svm import fit_svm, SVMModel
from cuanalytics.regression.linear import fit_lm, LinearRegressionModel
from cuanalytics.regression.logistic import fit_logit, LogisticRegressionModel

# Define what gets imported with "from cuanalytics import *"
__all__ = [
    'load_mushroom_data',
    'load_iris_data',
    'load_breast_cancer_data',
    'load_real_estate_data',
    'calculate_entropy',
    'information_gain',
    'plot_entropy',
    'split_data',
    'fit_tree',
    'SimpleDecisionTree',
    'fit_lda',
    'LDAModel',
    'fit_svm',
    'SVMModel',
    'fit_lm',
    'LinearRegressionModel',
    'fit_logit',
    'LogisticRegressionModel',
]
