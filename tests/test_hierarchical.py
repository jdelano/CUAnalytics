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
    assert all(isinstance(k, int) for k in metrics['cluster_counts'].keys())
    assert all(isinstance(v, int) for v in metrics['cluster_counts'].values())


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


def test_visualize_dendrogram_runs(cluster_data, monkeypatch):
    model = fit_hierarchical(cluster_data, formula='~ x1 + x2', n_clusters=3)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    model.visualize()


def test_visualize_dendrogram_with_cutoff_runs(cluster_data, monkeypatch):
    model = fit_hierarchical(cluster_data, formula='~ x1 + x2', n_clusters=3)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    model.visualize(cutoff=5)


def test_visualize_dendrogram_with_level_truncation_runs(cluster_data, monkeypatch):
    model = fit_hierarchical(cluster_data, formula='~ x1 + x2', n_clusters=3)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    model.visualize(cutoff=3, truncate_mode='level')


def test_visualize_dendrogram_invalid_cutoff_raises(cluster_data):
    model = fit_hierarchical(cluster_data, formula='~ x1 + x2', n_clusters=3)
    with pytest.raises(ValueError, match="cutoff must be >= 2 for truncate_mode='lastp'"):
        model.visualize(cutoff=1)


def test_visualize_dendrogram_level_cutoff_one_runs(cluster_data, monkeypatch):
    model = fit_hierarchical(cluster_data, formula='~ x1 + x2', n_clusters=3)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    model.visualize(cutoff=1, truncate_mode='level')


def test_visualize_dendrogram_level_cutoff_zero_runs(cluster_data, monkeypatch):
    model = fit_hierarchical(cluster_data, formula='~ x1 + x2', n_clusters=3)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    model.visualize(cutoff=0, truncate_mode='level')


def test_visualize_dendrogram_invalid_truncate_mode_raises(cluster_data):
    model = fit_hierarchical(cluster_data, formula='~ x1 + x2', n_clusters=3)
    with pytest.raises(ValueError, match="truncate_mode must be one of"):
        model.visualize(cutoff=3, truncate_mode='bad_mode')


def test_visualize_all_features_runs(cluster_data, monkeypatch):
    model = fit_hierarchical(cluster_data, formula='~ x1 + x2', n_clusters=3)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    model.visualize_all_features()
