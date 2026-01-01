# cuanalytics/svm/__init__.py
"""
Support Vector Machine module for ITM 4150.

This module provides a simplified interface to Support Vector Machines (SVM)
for binary classification tasks.
"""

from cuanalytics.svm.classifier import fit_svm, SVMModel

__all__ = ['fit_svm', 'SVMModel']