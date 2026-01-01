# tests/test_lda.py
"""
Tests for Linear Discriminant Analysis
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cuanalytics.lda import fit_lda, LDAModel


@pytest.fixture
def multiclass_data():
    """Simple 3-class classification dataset"""
    np.random.seed(42)
    df = pd.DataFrame({
        'x1': np.concatenate([
            np.random.randn(30) + 3,
            np.random.randn(30),
            np.random.randn(30) - 3
        ]),
        'x2': np.concatenate([
            np.random.randn(30) + 3,
            np.random.randn(30),
            np.random.randn(30) - 3
        ]),
        'class': ['A'] * 30 + ['B'] * 30 + ['C'] * 30
    })
    return df


@pytest.fixture
def binary_data():
    """Binary classification dataset"""
    np.random.seed(42)
    df = pd.DataFrame({
        'x1': np.concatenate([np.random.randn(40) + 2, np.random.randn(40) - 2]),
        'x2': np.concatenate([np.random.randn(40) + 2, np.random.randn(40) - 2]),
        'class': ['A'] * 40 + ['B'] * 40
    })
    return df


@pytest.fixture(autouse=True)
def close_plots():
    """Automatically close all plots after each test"""
    yield
    plt.close('all')


class TestLDABasic:
    """Test basic LDA functionality"""
    
    def test_fit_lda_basic(self, multiclass_data):
        """Test basic LDA fitting"""
        lda = fit_lda(multiclass_data, formula='class ~ .')
        
        assert hasattr(lda, 'lda')
        assert hasattr(lda, 'feature_names')
        assert hasattr(lda, 'classes')
        assert len(lda.classes) == 3
    
    def test_fit_with_n_components(self, multiclass_data):
        """Test fitting with specified n_components"""
        lda = fit_lda(multiclass_data, formula='class ~ .', n_components=1)
        assert lda.n_components == 1
    
    def test_fit_with_solver(self, multiclass_data):
        """Test fitting with different solver"""
        lda = fit_lda(multiclass_data, formula='class ~ .', solver='eigen')
        assert lda.solver == 'eigen'
    
    def test_direct_class_instantiation(self, multiclass_data):
        """Test creating LDAModel directly"""
        lda = LDAModel(multiclass_data, formula='class ~ .')
        assert lda is not None

    def test_fit_with_formula(self, multiclass_data):
        """Test fitting with formula input"""
        lda = fit_lda(multiclass_data, formula='class ~ x1 + x2')
        assert lda.target == 'class'
        assert set(lda.feature_names) == {'x1', 'x2'}

    def test_get_metrics(self, multiclass_data):
        """Test getting classification metrics"""
        lda = fit_lda(multiclass_data, formula='class ~ .')
        metrics = lda.get_metrics()

        assert metrics['model_type'] == 'lda'
        assert 'score' in metrics
        assert 'accuracy' in metrics['score']
        assert 'confusion_matrix' in metrics['score']
        assert 'per_class' in metrics['score']
        assert 'kappa' in metrics['score']
        assert 0 <= metrics['score']['accuracy'] <= 1
        assert isinstance(metrics['score']['confusion_matrix'], np.ndarray)
        assert metrics['score']['confusion_matrix'].shape == (3, 3)
        assert metrics['classes'] == lda.classes
        assert 'feature_importance' in metrics


class TestLDAValidation:
    """Test input validation"""
    
    def test_non_numeric_features_error(self, multiclass_data):
        """Test error when features are non-numeric"""
        df = multiclass_data.copy()
        df['categorical'] = ['cat1', 'cat2', 'cat3'] * 30
        
        with pytest.raises(ValueError, match="must be numeric"):
            fit_lda(df, formula='class ~ I(categorical)')
    
    def test_unfitted_model_error(self):
        """Test error when using unfitted model"""
        lda = LDAModel.__new__(LDAModel)
        # Don't set lda.lda
        
        with pytest.raises(RuntimeError, match="not been fitted"):
            lda.predict(pd.DataFrame({'x1': [1], 'x2': [2]}))


class TestLDAPrediction:
    """Test prediction methods"""
    
    def test_predict(self, multiclass_data):
        """Test basic prediction"""
        lda = fit_lda(multiclass_data, formula='class ~ .')
        predictions = lda.predict(multiclass_data)
        
        assert len(predictions) == len(multiclass_data)
        assert all(pred in lda.classes for pred in predictions)
    
    def test_predict_without_target(self, multiclass_data):
        """Test prediction on data without target column"""
        lda = fit_lda(multiclass_data, formula='class ~ .')
        X_test = multiclass_data.drop('class', axis=1)
        
        predictions = lda.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_predict_with_formula(self, multiclass_data):
        """Test prediction when fit with formula"""
        lda = fit_lda(multiclass_data, formula='class ~ x1 + x2')
        X_test = multiclass_data.drop('class', axis=1)

        predictions = lda.predict(X_test)
        assert len(predictions) == len(X_test)
    
    def test_predict_proba(self, multiclass_data):
        """Test probability prediction"""
        lda = fit_lda(multiclass_data, formula='class ~ .')
        probabilities = lda.predict_proba(multiclass_data)
        
        assert probabilities.shape == (len(multiclass_data), len(lda.classes))
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1
    
    def test_predict_proba_without_target(self, multiclass_data):
        """Test probability prediction without target column"""
        lda = fit_lda(multiclass_data, formula='class ~ .')
        X_test = multiclass_data.drop('class', axis=1)
        
        probabilities = lda.predict_proba(X_test)
        assert probabilities.shape[0] == len(X_test)
    
    def test_score(self, multiclass_data):
        """Test scoring"""
        lda = fit_lda(multiclass_data, formula='class ~ .')
        report = lda.score(multiclass_data)
        
        assert 0 <= report['accuracy'] <= 1
        assert report['accuracy'] > 0.7  # Should fit training data reasonably


class TestLDATransform:
    """Test transformation and dimensionality reduction"""
    
    def test_transform(self, multiclass_data):
        """Test transforming to LDA space"""
        lda = fit_lda(multiclass_data, formula='class ~ .')
        transformed = lda.transform(multiclass_data)
        
        assert transformed.shape[0] == len(multiclass_data)
        assert transformed.shape[1] <= len(lda.classes) - 1
    
    def test_transform_without_target(self, multiclass_data):
        """Test transform on data without target column"""
        lda = fit_lda(multiclass_data, formula='class ~ .')
        X_test = multiclass_data.drop('class', axis=1)
        
        transformed = lda.transform(X_test)
        assert transformed.shape[0] == len(X_test)
    
    def test_get_feature_importance(self, multiclass_data):
        """Test getting feature importance"""
        lda = fit_lda(multiclass_data, formula='class ~ .')
        importance = lda.get_feature_importance()
        
        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
        assert len(importance) == len(lda.feature_names)


class TestLDAVisualization:
    """Test visualization methods"""
    
    def test_visualize_multiclass(self, multiclass_data, monkeypatch):
        """Test visualization with 3 classes"""
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        
        lda = fit_lda(multiclass_data, formula='class ~ .')
        lda.visualize()
    
    def test_visualize_binary(self, binary_data, monkeypatch):
        """Test visualization with 2 classes (1D projection)"""
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        
        lda = fit_lda(binary_data, formula='class ~ .')
        lda.visualize()
    
    def test_visualize_custom_figsize(self, multiclass_data, monkeypatch):
        """Test visualization with custom figsize"""
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        
        lda = fit_lda(multiclass_data, formula='class ~ .')
        lda.visualize(figsize=(10, 6))
    
    def test_visualize_features(self, multiclass_data, monkeypatch):
        """Test feature-specific visualization"""
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        
        lda = fit_lda(multiclass_data, formula='class ~ .')
        lda.visualize_features('x1', 'x2')
    
    def test_visualize_features_invalid_features(self, multiclass_data):
        """Test error with invalid feature names"""
        lda = fit_lda(multiclass_data, formula='class ~ .')
        
        with pytest.raises(ValueError, match="Features must be from"):  # Changed from "original features"
            lda.visualize_features('invalid1', 'invalid2')
    
    def test_visualize_features_custom_figsize(self, multiclass_data, monkeypatch):
        """Test feature visualization with custom figsize"""
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        
        lda = fit_lda(multiclass_data, formula='class ~ .')
        lda.visualize_features('x1', 'x2', figsize=(12, 10))

    def test_visualize_features_formula_runs(self, multiclass_data, monkeypatch):
        """Test visualize_features runs with formula usage"""
        lda = fit_lda(multiclass_data, formula='class ~ x1 + x2')

        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        lda.visualize_features('x1', 'x2')


class TestLDASummary:
    """Test summary method"""
    
    def test_summary_multiclass(self, multiclass_data, monkeypatch):
        """Test summary with multiclass data"""
        monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
        
        lda = fit_lda(multiclass_data, formula='class ~ .')
        result = lda.summary()
        
        # Summary is display-only, should return None
        assert result is None
    
    def test_summary_binary(self, binary_data, monkeypatch):
        """Test summary with binary classification"""
        monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
        
        lda = fit_lda(binary_data, formula='class ~ .')
        result = lda.summary()
        
        assert result is None
    
    def test_summary_contents(self, multiclass_data, monkeypatch):
        """Test that summary runs without error"""
        monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
        
        lda = fit_lda(multiclass_data, formula='class ~ .')
        
        # Just verify it runs without error
        lda.summary()
        
        # Verify model attributes are accessible
        assert len(lda.classes) == 3
        assert len(lda.feature_names) == 2
    
    def test_summary_no_return(self, multiclass_data, monkeypatch):
        """Test that summary returns None (display only)"""
        monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
        
        lda = fit_lda(multiclass_data, formula='class ~ .')
        result = lda.summary()
        
        assert result is None


class TestLDAEdgeCases:
    """Test edge cases"""
    
    def test_two_classes(self, binary_data, monkeypatch):
        """Test with 2 classes (minimum for LDA)"""
        monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
        
        lda = fit_lda(binary_data, formula='class ~ .')
        assert len(lda.classes) == 2
        
        # Binary LDA has 1 discriminant
        transformed = lda.transform(binary_data)
        assert transformed.shape[1] == 1
    
    def test_many_classes(self, monkeypatch):
        """Test with many classes"""
        monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
        
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(150),
            'x2': np.random.randn(150),
            'class': ['A'] * 30 + ['B'] * 30 + ['C'] * 30 + ['D'] * 30 + ['E'] * 30
        })
        
        lda = fit_lda(df, formula='class ~ .')
        assert len(lda.classes) == 5
        
        # n_components = min(n_features, n_classes - 1)
        transformed = lda.transform(df)
        assert transformed.shape[1] == min(2, 4)  # min(2 features, 5-1 classes)
    
    def test_many_features(self, monkeypatch):
        """Test with many features"""
        monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
        
        np.random.seed(42)
        df = pd.DataFrame({
            f'x{i}': np.concatenate([
                np.random.randn(20) + i,
                np.random.randn(20),
                np.random.randn(20) - i
            ]) for i in range(10)
        })
        df['class'] = ['A'] * 20 + ['B'] * 20 + ['C'] * 20
        
        lda = fit_lda(df, formula='class ~ .')
        assert len(lda.feature_names) == 10
    
    def test_perfect_separation(self, monkeypatch):
        """Test with perfectly separable classes"""
        monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
        
        df = pd.DataFrame({
            'x1': [1, 2, 3, 10, 11, 12, 20, 21, 22],
            'x2': [1, 2, 3, 10, 11, 12, 20, 21, 22],
            'class': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
        })
        
        lda = fit_lda(df, formula='class ~ .')
        accuracy = lda.score(df)['accuracy']
        
        assert accuracy == 1.0  # Perfect separation


class TestLDAFeatureImportance:
    """Test feature importance without coefficients"""
    
    def test_feature_importance_no_coef(self, multiclass_data, monkeypatch):
        """Test that feature importance works when coef_ exists"""
        monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
        
        lda = fit_lda(multiclass_data, formula='class ~ .')
        
        # LDA should have coef_ after fitting
        assert hasattr(lda.lda, 'coef_')
        
        importance = lda.get_feature_importance()
        assert len(importance) == len(lda.feature_names)


class TestLDASolvers:
    """Test different solver options"""
    
    def test_svd_solver(self, multiclass_data, monkeypatch):
        """Test with SVD solver"""
        monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
        
        lda = fit_lda(multiclass_data, formula='class ~ .', solver='svd')
        assert lda.solver == 'svd'
    
    def test_lsqr_solver(self, multiclass_data, monkeypatch):
        """Test with LSQR solver"""
        monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
        
        lda = fit_lda(multiclass_data, formula='class ~ .', solver='lsqr')
        assert lda.solver == 'lsqr'
    
    def test_eigen_solver(self, multiclass_data, monkeypatch):
        """Test with eigen solver"""
        monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
        
        lda = fit_lda(multiclass_data, formula='class ~ .', solver='eigen')
        assert lda.solver == 'eigen'
