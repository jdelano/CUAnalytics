"""
ITM 4150: Advanced Business Analytics and Visualization
Python toolkit for course materials at Cedarville University.
"""

__version__ = "0.2.1"
__author__ = "Dr. John D. Delano"

# Import commonly used functions for convenient access
from cuanalytics.datasets.loaders import load_mushroom_data, load_iris_data, load_breast_cancer_data, load_real_estate_data
from cuanalytics.entropy.metrics import calculate_entropy, information_gain
from cuanalytics.entropy.visualization import plot_entropy
from cuanalytics.preprocessing.split import split_data
from cuanalytics.preprocessing.scale import scale_data
from cuanalytics.trees.decision_tree import fit_tree, SimpleDecisionTree
from cuanalytics.lda.discriminant import fit_lda, LDAModel
from cuanalytics.svm import fit_svm, SVMModel
from cuanalytics.regression.linear import fit_lm, LinearRegressionModel
from cuanalytics.regression.logistic import fit_logit, LogisticRegressionModel
from cuanalytics.neuralnet import fit_nn, NeuralNetModel
from cuanalytics.similarity import euclidean, manhattan, cosine, jaccard
from cuanalytics.knn import fit_knn_classifier, KNNClassifierModel, fit_knn_regressor, KNNRegressorModel
from cuanalytics.clustering import fit_kmeans, KMeansModel, fit_hierarchical, HierarchicalClusteringModel

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
    'scale_data',
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
    'fit_nn',
    'NeuralNetModel',
    'euclidean',
    'manhattan',
    'cosine',
    'jaccard',
    'fit_knn_classifier',
    'KNNClassifierModel',
    'fit_knn_regressor',
    'KNNRegressorModel',
    'fit_kmeans',
    'KMeansModel',
    'fit_hierarchical',
    'HierarchicalClusteringModel',
]
