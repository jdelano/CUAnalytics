import numpy as np
import pandas as pd
import pytest

from cuanalytics import fit_kmeans
from cuanalytics.clustering.kmeans import KMeansModel


@pytest.fixture
def cluster_data():
    np.random.seed(42)
    x1 = np.random.normal(0, 1, 60)
    x2 = np.random.normal(0, 1, 60)
    return pd.DataFrame({'x1': x1, 'x2': x2})


def test_fit_kmeans(cluster_data):
    model = fit_kmeans(cluster_data, formula='~ x1 + x2', n_clusters=3)
    assert isinstance(model, KMeansModel)


def test_predict_returns_series(cluster_data):
    model = fit_kmeans(cluster_data, formula='~ x1 + x2', n_clusters=3)
    preds = model.predict(cluster_data)
    assert isinstance(preds, pd.Series)
    assert len(preds) == len(cluster_data)


def test_score_returns_metrics(cluster_data):
    model = fit_kmeans(cluster_data, formula='~ x1 + x2', n_clusters=3)
    metrics = model.score(cluster_data)
    assert 'inertia' in metrics


def test_get_metrics(cluster_data):
    model = fit_kmeans(cluster_data, formula='~ x1 + x2', n_clusters=3)
    metrics = model.get_metrics()
    assert metrics['model_type'] == 'kmeans'
    assert all(isinstance(k, int) for k in metrics['cluster_counts'].keys())
    assert all(isinstance(v, int) for v in metrics['cluster_counts'].values())


def test_missing_formula_raises(cluster_data):
    with pytest.raises(ValueError, match="Must provide 'formula'"):
        fit_kmeans(cluster_data, formula=None)


def test_non_numeric_feature_raises(cluster_data):
    df = cluster_data.copy()
    df['cat'] = ['a', 'b', 'c'] * 20
    with pytest.raises(ValueError, match="All features must be numeric"):
        fit_kmeans(df, formula='~ x1 + cat', n_clusters=3)


def test_unfitted_predict_raises(cluster_data):
    model = KMeansModel.__new__(KMeansModel)
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.predict(cluster_data)


def test_visualize_runs(cluster_data, monkeypatch):
    model = fit_kmeans(cluster_data, formula='~ x1 + x2', n_clusters=3)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    model.visualize()


def test_describe_clusters_returns_labeled_dataframe(cluster_data):
    model = fit_kmeans(cluster_data, formula='~ x1 + x2', n_clusters=3, random_state=42)
    described = model.describe_clusters(max_depth=2)
    assert isinstance(described, pd.DataFrame)
    assert len(described) == len(cluster_data)
    assert 'cluster' in described.columns
    assert 'cluster_rule' in described.columns
    assert described['cluster'].notna().all()
    assert described['cluster_rule'].notna().all()


def test_describe_clusters_invalid_criterion_raises(cluster_data):
    model = fit_kmeans(cluster_data, formula='~ x1 + x2', n_clusters=3, random_state=42)
    with pytest.raises(ValueError, match="criterion must be 'entropy' or 'gini'"):
        model.describe_clusters(criterion='bad_criterion')
