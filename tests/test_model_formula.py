import pandas as pd
import pytest

from cuanalytics.formula import ModelFormula


def test_model_formula_supervised_builds_xy():
    df = pd.DataFrame({'x1': [1, 2], 'x2': [3, 4], 'y': [5, 6]})
    mf = ModelFormula(df, 'y ~ x1 + x2')
    assert mf.target == 'y'
    assert mf.y.tolist() == [5, 6]
    assert set(mf.feature_names) == {'x1', 'x2'}
    assert 'Intercept' not in mf.feature_names


def test_model_formula_unsupervised_rhs_only():
    df = pd.DataFrame({'x1': [1, 2], 'x2': [3, 4]})
    mf = ModelFormula(df, 'x1 + x2')
    assert mf.target is None
    assert mf.y is None
    assert set(mf.feature_names) == {'x1', 'x2'}


def test_model_formula_transform():
    df = pd.DataFrame({'x1': [1, 2], 'x2': [3, 4], 'y': [5, 6]})
    mf = ModelFormula(df, 'y ~ x1 + x2')
    new_df = pd.DataFrame({'x1': [10, 20], 'x2': [30, 40]})
    X_new = mf.transform(new_df)
    assert list(X_new.columns) == ['x1', 'x2']
    assert len(X_new) == 2


def test_model_formula_missing_target_raises():
    df = pd.DataFrame({'x1': [1, 2], 'x2': [3, 4]})
    with pytest.raises(ValueError, match="Target 'y' not found"):
        ModelFormula(df, 'y ~ x1 + x2')


def test_model_formula_validate_regression_requires_target():
    df = pd.DataFrame({'x1': [1, 2], 'x2': [3, 4]})
    mf = ModelFormula(df, 'x1 + x2')
    with pytest.raises(ValueError, match="Formula must include a target"):
        mf.validate_regression_target(df)



def test_model_formula_validate_regression_non_numeric_target():
    df = pd.DataFrame({'x1': [1, 2], 'y': ['a', 'b']})
    mf = ModelFormula(df, 'y ~ x1')
    with pytest.raises(ValueError, match="must be numeric"):
        mf.validate_regression_target(df)
