import numpy as np
import pandas as pd
import pytest

from cuanalytics import fit_hierarchical
from cuanalytics.clustering.hierarchical import HierarchicalClusteringModel


@pytest.fixture
def cluster_data():
    np.random.seed(42)
    x1 = np.random.normal(0, 1, 60)
    x2 = np.random.normal(0, 1, 60)
    return pd.DataFrame({'x1': x1, 'x2': x2})


def test_fit_hierarchical(cluster_data):
    model = fit_hierarchical(cluster_data, formula='~ x1 + x2', n_clusters=3)
    assert isinstance(model, HierarchicalClusteringModel)


def test_predict_returns_series(cluster_data):
    model = fit_hierarchical(cluster_data, formula='~ x1 + x2', n_clusters=3)
    labels = model.predict()
    assert isinstance(labels, pd.Series)
    assert len(labels) == len(cluster_data)


def test_score_returns_metrics(cluster_data):
    model = fit_hierarchical(cluster_data, formula='~ x1 + x2', n_clusters=3)
    metrics = model.score()
    assert isinstance(metrics, dict)


def test_get_metrics(cluster_data):
    model = fit_hierarchical(cluster_data, formula='~ x1 + x2', n_clusters=3)
    metrics = model.get_metrics()
    assert metrics['model_type'] == 'hierarchical'


def test_predict_new_data_error(cluster_data):
    model = fit_hierarchical(cluster_data, formula='~ x1 + x2', n_clusters=3)
    with pytest.raises(ValueError, match="does not support predicting new data"):
        model.predict(cluster_data)


def test_missing_formula_raises(cluster_data):
    with pytest.raises(ValueError, match="Must provide 'formula'"):
        fit_hierarchical(cluster_data, formula=None)


def test_non_numeric_feature_raises(cluster_data):
    df = cluster_data.copy()
    df['cat'] = ['a', 'b', 'c'] * 20
    with pytest.raises(ValueError, match="All features must be numeric"):
        fit_hierarchical(df, formula='~ x1 + cat', n_clusters=3)


def test_unfitted_predict_raises(cluster_data):
    model = HierarchicalClusteringModel.__new__(HierarchicalClusteringModel)
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.predict()
