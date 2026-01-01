# tests/test_svm.py
"""
Tests for SVM classifier
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cuanalytics.svm import fit_svm, SVMModel


@pytest.fixture
def binary_classification_data():
    """Simple binary classification dataset"""
    np.random.seed(42)
    df = pd.DataFrame({
        'x1': np.concatenate([np.random.randn(50) + 2, np.random.randn(50) - 2]),
        'x2': np.concatenate([np.random.randn(50) + 2, np.random.randn(50) - 2]),
        'class': ['A'] * 50 + ['B'] * 50
    })
    return df


@pytest.fixture(autouse=True)
def close_plots():
    """Automatically close all plots after each test"""
    yield
    plt.close('all')


class TestSVMBasic:
    """Test basic SVM functionality"""
    
    def test_fit_svm_basic(self, binary_classification_data):
        """Test basic SVM fitting"""
        svm = fit_svm(binary_classification_data, formula='class ~ .')
        
        assert hasattr(svm, 'svm')
        assert hasattr(svm, 'feature_names')
        assert hasattr(svm, 'classes')
        assert len(svm.classes) == 2
    
    def test_fit_with_custom_C(self, binary_classification_data):
        """Test fitting with custom C parameter"""
        svm = fit_svm(binary_classification_data, formula='class ~ .', C=0.5)
        assert svm.C == 0.5
    
    def test_direct_class_instantiation(self, binary_classification_data):
        """Test creating SVMModel directly"""
        svm = SVMModel(binary_classification_data, formula='class ~ .', C=2.0)
        assert svm.C == 2.0

    def test_fit_with_formula(self, binary_classification_data):
        """Test fitting with formula input"""
        svm = fit_svm(binary_classification_data, formula='class ~ x1 + x2')
        assert svm.target == 'class'
        assert set(svm.feature_names) == {'x1', 'x2'}


class TestSVMValidation:
    """Test input validation"""
    
    def test_non_binary_classification_error(self):
        """Test error when more than 2 classes"""
        df = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5, 6],
            'x2': [2, 3, 4, 5, 6, 7],
            'class': ['A', 'B', 'C', 'A', 'B', 'C']
        })
        
        with pytest.raises(ValueError, match="binary classification"):
            fit_svm(df, formula='class ~ .')
    
    def test_non_numeric_features_error(self, binary_classification_data):
        """Test error when features are non-numeric"""
        df = binary_classification_data.copy()
        df['categorical'] = ['cat1', 'cat2'] * 50
        
        with pytest.raises(ValueError, match="must be numeric"):
            fit_svm(df, formula='class ~ I(categorical)')
    
    def test_unfitted_model_error(self):
        """Test error when using unfitted model"""
        svm = SVMModel.__new__(SVMModel)
        # Don't set svm.svm
        
        with pytest.raises(RuntimeError, match="not been fitted"):
            svm.predict(pd.DataFrame({'x1': [1], 'x2': [2]}))


class TestSVMPrediction:
    """Test prediction methods"""
    
    def test_predict(self, binary_classification_data):
        """Test basic prediction"""
        svm = fit_svm(binary_classification_data, formula='class ~ .')
        predictions = svm.predict(binary_classification_data)
        
        assert len(predictions) == len(binary_classification_data)
        assert all(pred in svm.classes for pred in predictions)
    
    def test_predict_without_target(self, binary_classification_data):
        """Test prediction on data without target column"""
        svm = fit_svm(binary_classification_data, formula='class ~ .')
        X_test = binary_classification_data.drop('class', axis=1)
        
        predictions = svm.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_predict_with_formula(self, binary_classification_data):
        """Test prediction when fit with formula"""
        svm = fit_svm(binary_classification_data, formula='class ~ x1 + x2')
        X_test = binary_classification_data.drop('class', axis=1)

        predictions = svm.predict(X_test)
        assert len(predictions) == len(X_test)
    
    def test_score(self, binary_classification_data):
        """Test scoring"""
        svm = fit_svm(binary_classification_data, formula='class ~ .')
        report = svm.score(binary_classification_data)
        
        assert 0 <= report['accuracy'] <= 1
        assert report['accuracy'] > 0.8  # Should fit training data well

    def test_get_metrics(self, binary_classification_data):
        """Test getting classification metrics"""
        svm = fit_svm(binary_classification_data, formula='class ~ .')
        metrics = svm.get_metrics()

        assert metrics['model_type'] == 'svm'
        assert 'score' in metrics
        assert 'accuracy' in metrics['score']
        assert 'confusion_matrix' in metrics['score']
        assert 'per_class' in metrics['score']
        assert 'kappa' in metrics['score']
        assert 0 <= metrics['score']['accuracy'] <= 1
        assert isinstance(metrics['score']['confusion_matrix'], np.ndarray)
        assert metrics['score']['confusion_matrix'].shape == (2, 2)
        assert metrics['classes'] == svm.classes
        assert 'coefficients' in metrics
        assert 'margin_width' in metrics


class TestSVMSupportVectors:
    """Test support vector methods"""
    
    def test_get_support_vectors(self, binary_classification_data):
        """Test retrieving support vectors"""
        svm = fit_svm(binary_classification_data, formula='class ~ .')
        support_vectors = svm.get_support_vectors()
        
        assert isinstance(support_vectors, pd.DataFrame)
        assert len(support_vectors) > 0
        assert len(support_vectors) <= len(binary_classification_data)
    
    def test_get_coefficients(self, binary_classification_data):
        """Test getting model coefficients"""
        svm = fit_svm(binary_classification_data, formula='class ~ .')
        coef_df = svm.get_coefficients()
        
        assert isinstance(coef_df, pd.DataFrame)
        assert 'feature' in coef_df.columns
        assert 'coefficient' in coef_df.columns
        assert len(coef_df) == len(svm.feature_names)
    
    def test_get_margin_width(self, binary_classification_data):
        """Test margin width calculation"""
        svm = fit_svm(binary_classification_data, formula='class ~ .')
        margin = svm.get_margin_width()
        
        assert isinstance(margin, float)
        assert margin > 0


class TestSVMVisualization:
    """Test visualization methods"""
    
    def test_visualize(self, binary_classification_data, monkeypatch):
        """Test basic visualization"""
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        
        svm = fit_svm(binary_classification_data, formula='class ~ .')
        svm.visualize()
    
    def test_visualize_custom_figsize(self, binary_classification_data, monkeypatch):
        """Test visualization with custom figsize"""
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        
        svm = fit_svm(binary_classification_data, formula='class ~ .')
        svm.visualize(figsize=(10, 6))
    
    def test_visualize_features(self, binary_classification_data, monkeypatch):
        """Test feature-specific visualization"""
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        
        svm = fit_svm(binary_classification_data, formula='class ~ .')
        svm.visualize_features('x1', 'x2')
    
    def test_visualize_features_invalid_features(self, binary_classification_data):
        """Test error with invalid feature names"""
        svm = fit_svm(binary_classification_data, formula='class ~ .')
        
        with pytest.raises(ValueError, match="Features must be from"):
            svm.visualize_features('invalid1', 'invalid2')
    
    def test_visualize_features_custom_figsize(self, binary_classification_data, monkeypatch):
        """Test feature visualization with custom figsize"""
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        
        svm = fit_svm(binary_classification_data, formula='class ~ .')
        svm.visualize_features('x1', 'x2', figsize=(12, 10))


class TestSVMSummary:
    """Test summary method"""
    
    def test_summary(self, binary_classification_data, monkeypatch):
        """Test summary output"""
        monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
        
        svm = fit_svm(binary_classification_data, formula='class ~ .')
        result = svm.summary()
        
        # Summary is display-only, should return None
        assert result is None
    
    def test_summary_contents(self, binary_classification_data, monkeypatch):
        """Test that summary runs without error"""
        monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
        
        svm = fit_svm(binary_classification_data, formula='class ~ .', C=2.0)
        
        # Just verify it runs without error
        svm.summary()
        
        # Verify model attributes are accessible
        assert svm.C == 2.0
        assert svm.get_margin_width() > 0
        assert len(svm.feature_names) == 2


class TestSVMEdgeCases:
    """Test edge cases"""
    
    def test_perfect_separation(self, monkeypatch):
        """Test with perfectly separable data"""
        monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
        
        df = pd.DataFrame({
            'x1': [1, 2, 3, 10, 11, 12],
            'x2': [1, 2, 3, 10, 11, 12],
            'class': ['A', 'A', 'A', 'B', 'B', 'B']
        })
        
        svm = fit_svm(df, formula='class ~ .')
        accuracy = svm.score(df)['accuracy']
        
        assert accuracy == 1.0  # Perfect separation
    
    def test_high_C_value(self, binary_classification_data, monkeypatch):
        """Test with high C (strict margin)"""
        monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
        
        svm = fit_svm(binary_classification_data, formula='class ~ .', C=100.0)
        assert svm.C == 100.0
    
    def test_low_C_value(self, binary_classification_data, monkeypatch):
        """Test with low C (flexible margin)"""
        monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
        
        svm = fit_svm(binary_classification_data, formula='class ~ .', C=0.01)
        assert svm.C == 0.01
    
    def test_many_features(self, monkeypatch):
        """Test with many features"""
        monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
        
        np.random.seed(42)
        df = pd.DataFrame({
            f'x{i}': np.concatenate([
                np.random.randn(30) + 1,
                np.random.randn(30) - 1
            ]) for i in range(10)
        })
        df['class'] = ['A'] * 30 + ['B'] * 30
        
        svm = fit_svm(df, formula='class ~ .')
        assert len(svm.feature_names) == 10


class TestSVMFitFailure:
    """Test handling of fit failures"""
    
    def test_fit_with_insufficient_data(self, monkeypatch):
        """Test with very small dataset"""
        monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
        
        df = pd.DataFrame({
            'x1': [1, 2],
            'x2': [1, 2],
            'class': ['A', 'B']
        })
        
        # Should still work with minimal data
        svm = fit_svm(df, formula='class ~ .')
        assert svm is not None
