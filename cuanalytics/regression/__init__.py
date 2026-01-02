"""
Regression module for ITM 4150.

This module provides simplified interfaces for regression tasks.
"""

from cuanalytics.regression.linear import fit_lm, LinearRegressionModel
from cuanalytics.regression.logistic import fit_logit, LogisticRegressionModel

__all__ = ['fit_lm', 'LinearRegressionModel', 'fit_logit', 'LogisticRegressionModel']
