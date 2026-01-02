import numpy as np
import pandas as pd
import pytest

from cuanalytics.preprocessing.scale import scale_data


def test_scale_data_standard_excludes_cols():
    df = pd.DataFrame({
        'x1': [1.0, 2.0, 3.0],
        'x2': [10.0, 20.0, 30.0],
        'y': [0, 1, 0],
    })
    scaled, scaler = scale_data(df, exclude_cols=['y'])
    assert 'y' in scaled.columns
    assert np.allclose(scaled['y'], df['y'])
    assert not np.allclose(scaled['x1'], df['x1'])
    assert scaler is not None


def test_scale_data_minmax():
    df = pd.DataFrame({
        'x1': [0.0, 5.0, 10.0],
        'y': [1, 0, 1],
    })
    scaled, _ = scale_data(df, method='minmax', exclude_cols=['y'])
    assert scaled['x1'].min() == 0.0
    assert scaled['x1'].max() == 1.0


def test_scale_data_with_existing_scaler():
    train = pd.DataFrame({'x1': [0.0, 1.0, 2.0], 'y': [0, 1, 0]})
    test = pd.DataFrame({'x1': [1.0, 3.0], 'y': [1, 0]})
    train_scaled, scaler = scale_data(train, exclude_cols=['y'])
    test_scaled, _ = scale_data(test, exclude_cols=['y'], scaler=scaler)
    assert train_scaled['x1'].mean() == pytest.approx(0.0, abs=1e-7)
    assert 'x1' in test_scaled.columns


def test_scale_data_invalid_method():
    df = pd.DataFrame({'x1': [1.0, 2.0], 'y': [0, 1]})
    with pytest.raises(ValueError, match="method must be"):
        scale_data(df, method='unknown', exclude_cols=['y'])


def test_scale_data_skip_binary_default():
    df = pd.DataFrame({
        'x1': [1.0, 2.0, 3.0],
        'flag': [0, 1, 0],
        'y': [0, 1, 0],
    })
    scaled, _ = scale_data(df, exclude_cols=['y'])
    assert np.allclose(scaled['flag'], df['flag'])
