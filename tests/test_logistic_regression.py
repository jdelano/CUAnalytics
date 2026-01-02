import numpy as np
import pandas as pd
import pytest

from cuanalytics import fit_logit
from cuanalytics.regression.logistic import LogisticRegressionModel


@pytest.fixture
def binary_data():
    np.random.seed(42)
    x1 = np.random.normal(0, 1, 100)
    x2 = np.random.normal(0, 1, 100)
    logits = 0.5 * x1 - 0.3 * x2
    probs = 1 / (1 + np.exp(-logits))
    y = np.where(probs > 0.5, 'A', 'B')
    return pd.DataFrame({'x1': x1, 'x2': x2, 'class': y})


def test_fit_logit_returns_model(binary_data):
    model = fit_logit(binary_data, formula='class ~ x1 + x2')
    assert isinstance(model, LogisticRegressionModel)

def test_missing_formula_raises(binary_data):
    with pytest.raises(ValueError, match="Must provide 'formula'"):
        fit_logit(binary_data, formula=None)


def test_model_missing_formula_raises(binary_data):
    with pytest.raises(ValueError, match="Must provide 'formula'"):
        LogisticRegressionModel(binary_data, formula=None)


def test_invalid_formula_raises(binary_data):
    with pytest.raises(ValueError, match="Formula must include a target"):
        fit_logit(binary_data, formula='class x1 + x2')


def test_missing_target_column_raises(binary_data):
    with pytest.raises(ValueError, match="Target 'missing' not found"):
        fit_logit(binary_data, formula='missing ~ x1 + x2')


def test_formulaic_missing_raises(binary_data, monkeypatch):
    import builtins
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == 'formulaic':
            raise ImportError("No module named 'formulaic'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', mock_import)
    with pytest.raises(ImportError, match="requires the 'formulaic' library"):
        fit_logit(binary_data, formula='class ~ x1 + x2')


def test_non_numeric_feature_raises(binary_data):
    df = binary_data.copy()
    df['cat'] = ['a', 'b'] * 50
    with pytest.raises(ValueError, match="All features must be numeric"):
        fit_logit(df, formula='class ~ x1 + cat')


def test_unfitted_predict_raises(binary_data):
    model = LogisticRegressionModel.__new__(LogisticRegressionModel)
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.predict(binary_data)


def test_predict_returns_series(binary_data):
    model = fit_logit(binary_data, formula='class ~ x1 + x2')
    preds = model.predict(binary_data)
    assert isinstance(preds, pd.Series)
    assert len(preds) == len(binary_data)


def test_score_returns_report(binary_data, monkeypatch):
    model = fit_logit(binary_data, formula='class ~ x1 + x2')
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
    report = model.score(binary_data)
    assert isinstance(report, dict)
    assert 'accuracy' in report
    assert 'confusion_matrix' in report


def test_get_metrics(binary_data):
    model = fit_logit(binary_data, formula='class ~ x1 + x2')
    metrics = model.get_metrics()
    assert metrics['model_type'] == 'logistic_regression'
    assert 'coefficients' in metrics


def test_summary_returns_none(binary_data, monkeypatch):
    model = fit_logit(binary_data, formula='class ~ x1 + x2')
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
    result = model.summary()
    assert result is None


def test_predict_proba(binary_data):
    model = fit_logit(binary_data, formula='class ~ x1 + x2')
    probs = model.predict_proba(binary_data)
    assert probs.shape[0] == len(binary_data)
    assert probs.shape[1] == len(model.classes)


def test_transform_missing_model_spec_raises(binary_data):
    model = fit_logit(binary_data, formula='class ~ x1 + x2')
    model.model_spec = None
    with pytest.raises(RuntimeError, match="Formula metadata missing"):
        model._transform_data_with_formula(binary_data)


def test_transform_rhs_attribute_branch(binary_data, monkeypatch):
    model = fit_logit(binary_data, formula='class ~ x1 + x2')

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


def test_init_rhs_attribute_branch(binary_data, monkeypatch):
    class DummySpec:
        def __init__(self):
            self.variables = set()

    class DummyMatrices:
        def __init__(self, rhs):
            self.rhs = rhs
            self.model_spec = DummySpec()

    def mock_model_matrix(formula, data, output='pandas'):
        rhs = pd.DataFrame(
            {
                'Intercept': [1] * len(data),
                'x1': data['x1'].values,
                'x2': data['x2'].values
            },
            index=data.index
        )
        return DummyMatrices(rhs)

    import formulaic
    monkeypatch.setattr(formulaic, 'model_matrix', mock_model_matrix)

    model = fit_logit(binary_data, formula='class ~ x1 + x2')
    assert model.feature_names == ['x1', 'x2']


def test_visualize_runs(binary_data, monkeypatch):
    model = fit_logit(binary_data, formula='class ~ x1 + x2')
    monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
    model.visualize()


def test_visualize_multiclass_coefficients(monkeypatch):
    np.random.seed(7)
    x1 = np.random.normal(0, 1, 90)
    x2 = np.random.normal(0, 1, 90)
    y = np.array(['A'] * 30 + ['B'] * 30 + ['C'] * 30)
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'class': y})
    model = fit_logit(df, formula='class ~ x1 + x2')
    monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
    model.visualize()


def test_visualize_features_runs(binary_data, monkeypatch):
    model = fit_logit(binary_data, formula='class ~ x1 + x2')
    monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
    model.visualize_features('x1', 'x2')


def test_visualize_features_invalid_feature(binary_data):
    model = fit_logit(binary_data, formula='class ~ x1 + x2')
    with pytest.raises(ValueError, match="Features must be from"):
        model.visualize_features('x1', 'missing')


@pytest.mark.filterwarnings("ignore:.*penalty.*:FutureWarning")
@pytest.mark.filterwarnings("ignore:.*penalty.*:UserWarning")
def test_visualize_features_penalty_non_default(binary_data, monkeypatch):
    model = fit_logit(binary_data, formula='class ~ x1 + x2', penalty='l1', solver='liblinear')
    monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
    model.visualize_features('x1', 'x2')


def test_visualize_all_features_runs(binary_data, monkeypatch):
    model = fit_logit(binary_data, formula='class ~ x1 + x2')
    monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
    model.visualize_all_features(cols=1)


def test_visualize_all_features_single_axis(monkeypatch):
    np.random.seed(11)
    x1 = np.random.normal(0, 1, 50)
    logits = 0.8 * x1
    probs = 1 / (1 + np.exp(-logits))
    y = np.where(probs > 0.5, 'A', 'B')
    df = pd.DataFrame({'x1': x1, 'class': y})
    model = fit_logit(df, formula='class ~ x1')
    monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
    model.visualize_all_features(cols=1)


def test_visualize_all_features_hides_unused_axes(binary_data, monkeypatch):
    model = fit_logit(binary_data, formula='class ~ x1 + x2')
    monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
    model.visualize_all_features(cols=3)


def test_visualize_all_features_multiclass_error(monkeypatch):
    np.random.seed(8)
    x1 = np.random.normal(0, 1, 90)
    x2 = np.random.normal(0, 1, 90)
    y = np.array(['A'] * 30 + ['B'] * 30 + ['C'] * 30)
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'class': y})
    model = fit_logit(df, formula='class ~ x1 + x2')
    with pytest.raises(ValueError, match="only supported for binary"):
        model.visualize_all_features()


def test_coeff_table_handles_singular_matrix(binary_data, monkeypatch):
    model = fit_logit(binary_data, formula='class ~ x1 + x2')

    def mock_inv(*args, **kwargs):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(np.linalg, 'inv', mock_inv)
    table = model.get_coefficients()
    assert table['z_score'].isna().any()
    assert table['significance'].isin(['', '***', '**', '*', '.']).all()


def test_coeff_table_multiclass_branch(monkeypatch):
    np.random.seed(9)
    x1 = np.random.normal(0, 1, 90)
    x2 = np.random.normal(0, 1, 90)
    y = np.array(['A'] * 30 + ['B'] * 30 + ['C'] * 30)
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'class': y})
    model = fit_logit(df, formula='class ~ x1 + x2')
    table = model.get_coefficients()
    assert set(table['class']) == set(model.classes)


def test_coeff_table_multiclass_singular(monkeypatch):
    np.random.seed(10)
    x1 = np.random.normal(0, 1, 90)
    x2 = np.random.normal(0, 1, 90)
    y = np.array(['A'] * 30 + ['B'] * 30 + ['C'] * 30)
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'class': y})
    model = fit_logit(df, formula='class ~ x1 + x2')

    def mock_inv(*args, **kwargs):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(np.linalg, 'inv', mock_inv)
    table = model.get_coefficients()
    assert table['z_score'].isna().any()


def test_significance_stars(binary_data):
    model = fit_logit(binary_data, formula='class ~ x1 + x2')
    assert model._get_significance_stars(0.0005) == '***'
    assert model._get_significance_stars(0.005) == '**'
    assert model._get_significance_stars(0.03) == '*'
    assert model._get_significance_stars(0.08) == '.'
    assert model._get_significance_stars(0.5) == ''
