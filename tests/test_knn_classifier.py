import numpy as np
import pandas as pd
import pytest

from cuanalytics import fit_knn_classifier
from cuanalytics.knn.classifier import KNNClassifierModel


@pytest.fixture
def binary_data():
    np.random.seed(42)
    x1 = np.random.normal(0, 1, 120)
    x2 = np.random.normal(0, 1, 120)
    y = np.where(x1 + x2 > 0, 'A', 'B')
    return pd.DataFrame({'x1': x1, 'x2': x2, 'class': y})


def test_fit_knn_classifier(binary_data):
    model = fit_knn_classifier(binary_data, formula='class ~ x1 + x2', k=3)
    assert isinstance(model, KNNClassifierModel)


def test_predict_returns_series(binary_data):
    model = fit_knn_classifier(binary_data, formula='class ~ x1 + x2', k=3)
    preds = model.predict(binary_data)
    assert isinstance(preds, pd.Series)
    assert len(preds) == len(binary_data)


def test_score_returns_report(binary_data, monkeypatch):
    model = fit_knn_classifier(binary_data, formula='class ~ x1 + x2', k=3)
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
    report = model.score(binary_data)
    assert 'accuracy' in report
    assert 'confusion_matrix' in report


def test_get_metrics(binary_data):
    model = fit_knn_classifier(binary_data, formula='class ~ x1 + x2', k=3)
    metrics = model.get_metrics()
    assert metrics['model_type'] == 'knn_classifier'


def test_visualize_invalid_feature_count(binary_data):
    df = binary_data.copy()
    df['x3'] = np.random.normal(0, 1, len(df))
    model = fit_knn_classifier(df, formula='class ~ x1 + x2 + x3', k=3)
    with pytest.raises(ValueError, match="exactly 2"):
        model.visualize()


def test_missing_formula_raises(binary_data):
    with pytest.raises(ValueError, match="Must provide 'formula'"):
        fit_knn_classifier(binary_data, formula=None)


def test_invalid_formula_raises(binary_data):
    with pytest.raises(ValueError, match="Formula must include a target"):
        fit_knn_classifier(binary_data, formula='class x1 + x2')


def test_non_numeric_feature_raises(binary_data):
    df = binary_data.copy()
    df['cat'] = ['a', 'b'] * 60
    with pytest.raises(ValueError, match="All features must be numeric"):
        fit_knn_classifier(df, formula='class ~ x1 + cat')


def test_unfitted_predict_raises(binary_data):
    model = KNNClassifierModel.__new__(KNNClassifierModel)
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.predict(binary_data)
