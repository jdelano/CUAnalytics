import numpy as np
import pandas as pd
import pytest

from cuanalytics import fit_knn_regressor
from cuanalytics.knn.regressor import KNNRegressorModel


@pytest.fixture
def regression_data():
    np.random.seed(7)
    x1 = np.random.normal(0, 1, 120)
    x2 = np.random.normal(0, 1, 120)
    y = 2.5 * x1 - 1.2 * x2 + np.random.normal(0, 0.1, 120)
    return pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})


def test_fit_knn_regressor(regression_data):
    model = fit_knn_regressor(regression_data, formula='y ~ x1 + x2', k=3)
    assert isinstance(model, KNNRegressorModel)


def test_predict_returns_series(regression_data):
    model = fit_knn_regressor(regression_data, formula='y ~ x1 + x2', k=3)
    preds = model.predict(regression_data)
    assert isinstance(preds, pd.Series)
    assert len(preds) == len(regression_data)


def test_score_returns_metrics(regression_data):
    model = fit_knn_regressor(regression_data, formula='y ~ x1 + x2', k=3)
    report = model.score(regression_data)
    assert 'r2' in report
    assert 'rmse' in report
    assert 'mae' in report


def test_get_metrics(regression_data):
    model = fit_knn_regressor(regression_data, formula='y ~ x1 + x2', k=3)
    metrics = model.get_metrics()
    assert metrics['model_type'] == 'knn_regressor'


def test_missing_formula_raises(regression_data):
    with pytest.raises(ValueError, match="Must provide 'formula'"):
        fit_knn_regressor(regression_data, formula=None)


def test_invalid_formula_raises(regression_data):
    with pytest.raises(ValueError, match="Formula must include a target"):
        fit_knn_regressor(regression_data, formula='y x1 + x2')


def test_non_numeric_feature_raises(regression_data):
    df = regression_data.copy()
    df['cat'] = ['a', 'b'] * 60
    with pytest.raises(ValueError, match="All features must be numeric"):
        fit_knn_regressor(df, formula='y ~ x1 + cat')


def test_unfitted_predict_raises(regression_data):
    model = KNNRegressorModel.__new__(KNNRegressorModel)
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.predict(regression_data)
