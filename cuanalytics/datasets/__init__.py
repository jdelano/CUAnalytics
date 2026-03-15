"""
Dataset loaders for ITM 4150 course examples.
"""

from .loaders import (
    load_movie_reviews_data,
    load_sms_spam_data,
    load_mushroom_data,
    load_iris_data,
    load_breast_cancer_data,
    load_real_estate_data,
    list_datasets,
    load_dataset,
    AVAILABLE_DATASETS,
)

__all__ = [
    'load_movie_reviews_data',
    'load_sms_spam_data',
    'load_mushroom_data',
    'load_iris_data',
    'load_breast_cancer_data',
    'load_real_estate_data',
    'list_datasets',
    'load_dataset',
    'AVAILABLE_DATASETS',
]
