import numpy as np
import pandas as pd
import pytest

from cuanalytics import fit_nn
from cuanalytics.neuralnet import NeuralNetModel


@pytest.fixture
def binary_data():
    np.random.seed(42)
    x1 = np.random.normal(0, 1, 120)
    x2 = np.random.normal(0, 1, 120)
    logits = 0.6 * x1 - 0.4 * x2
    probs = 1 / (1 + np.exp(-logits))
    y = np.where(probs > 0.5, 'A', 'B')
    return pd.DataFrame({'x1': x1, 'x2': x2, 'class': y})


@pytest.fixture
def regression_data():
    np.random.seed(7)
    x1 = np.random.normal(0, 1, 120)
    x2 = np.random.normal(0, 1, 120)
    y = 2.5 * x1 - 1.2 * x2 + np.random.normal(0, 0.1, 120)
    return pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})


def test_fit_nn_returns_model(binary_data):
    model = fit_nn(binary_data, formula='class ~ x1 + x2', hidden_layers=[3, 2], solver='lbfgs', max_iter=1000)
    assert isinstance(model, NeuralNetModel)


def test_predict_returns_series(binary_data):
    model = fit_nn(binary_data, formula='class ~ x1 + x2', hidden_layers=[3], solver='lbfgs', max_iter=1000)
    preds = model.predict(binary_data)
    assert isinstance(preds, pd.Series)
    assert len(preds) == len(binary_data)


def test_score_classification(binary_data, monkeypatch):
    model = fit_nn(binary_data, formula='class ~ x1 + x2', hidden_layers=[4], solver='lbfgs', max_iter=1000)
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
    report = model.score(binary_data)
    assert isinstance(report, dict)
    assert 'accuracy' in report


def test_score_regression(regression_data):
    model = fit_nn(regression_data, formula='y ~ x1 + x2', hidden_layers=[4], solver='lbfgs', max_iter=1000)
    report = model.score(regression_data)
    assert 'r2' in report
    assert 'rmse' in report
    assert 'mae' in report


def test_predict_proba_classification(binary_data):
    model = fit_nn(binary_data, formula='class ~ x1 + x2', hidden_layers=[3], solver='lbfgs', max_iter=1000)
    probs = model.predict_proba(binary_data)
    assert probs.shape[0] == len(binary_data)
    assert probs.shape[1] == len(model.classes)


def test_predict_proba_regression_error(regression_data):
    model = fit_nn(regression_data, formula='y ~ x1 + x2', hidden_layers=[3], solver='lbfgs', max_iter=1000)
    with pytest.raises(ValueError, match="predict_proba is only available"):
        model.predict_proba(regression_data)


def test_get_metrics(binary_data):
    model = fit_nn(binary_data, formula='class ~ x1 + x2', hidden_layers=[3], solver='lbfgs', max_iter=1000)
    metrics = model.get_metrics()
    assert metrics['model_type'] == 'neural_network'
    assert metrics['task_type'] == 'classification'


def test_summary_returns_none(binary_data, monkeypatch):
    model = fit_nn(binary_data, formula='class ~ x1 + x2', hidden_layers=[3], solver='lbfgs', max_iter=1000)
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
    result = model.summary()
    assert result is None


def test_visualize_runs(binary_data, monkeypatch):
    model = fit_nn(binary_data, formula='class ~ x1 + x2', hidden_layers=[3], solver='lbfgs', max_iter=1000)
    monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
    model.visualize()


def test_missing_formula_raises(binary_data):
    with pytest.raises(ValueError, match="Must provide 'formula'"):
        fit_nn(binary_data, formula=None)


def test_invalid_formula_raises(binary_data):
    with pytest.raises(ValueError, match="Formula must include a target"):
        fit_nn(binary_data, formula='class x1 + x2')


def test_missing_target_column_raises(binary_data):
    with pytest.raises(ValueError, match="Target 'missing' not found"):
        fit_nn(binary_data, formula='missing ~ x1 + x2')


def test_non_numeric_feature_raises(binary_data):
    df = binary_data.copy()
    df['cat'] = ['a', 'b'] * 60
    with pytest.raises(ValueError, match="All features must be numeric"):
        fit_nn(df, formula='class ~ x1 + cat')


def test_unfitted_predict_raises(binary_data):
    model = NeuralNetModel.__new__(NeuralNetModel)
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.predict(binary_data)


def test_transform_missing_model_spec_raises(binary_data):
    model = fit_nn(binary_data, formula='class ~ x1 + x2', hidden_layers=[3], solver='lbfgs', max_iter=1000)
    model.model_spec = None
    with pytest.raises(RuntimeError, match="Formula metadata missing"):
        model._transform_data_with_formula(binary_data)


def test_transform_rhs_attribute_branch(binary_data):
    model = fit_nn(binary_data, formula='class ~ x1 + x2', hidden_layers=[3], solver='lbfgs', max_iter=1000)

    class DummyMatrices:
        def __init__(self, rhs):
            self.rhs = rhs

    class DummySpec:
        def get_model_matrix(self, data, output='pandas'):
            rhs = pd.DataFrame(
                {
                    'Intercept': [1] * len(data),
                    'x1': data['x1'].values,
                    'x2': data['x2'].values
                },
                index=data.index
            )
            return DummyMatrices(rhs)

    model.model_spec = DummySpec()
    X_new = model._transform_data_with_formula(binary_data)
    assert list(X_new.columns) == ['x1', 'x2']
