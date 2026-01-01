"""
Tests for decision tree module
"""
import pytest
import pandas as pd
import numpy as np
from cuanalytics import fit_tree, split_data
from cuanalytics.trees.decision_tree import SimpleDecisionTree
from cuanalytics.datasets import load_mushroom_data


class TestSimpleDecisionTree:
    """Tests for SimpleDecisionTree class"""
    
    def setup_method(self):
        """Set up test data before each test"""
        # Simple categorical dataset
        self.simple_df = pd.DataFrame({
            'feature1': ['A', 'A', 'B', 'B', 'C', 'C'],
            'feature2': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
            'class': ['yes', 'yes', 'no', 'no', 'yes', 'no']
        })
        
        # Mixed numeric and categorical dataset
        self.mixed_df = pd.DataFrame({
            'age': [25, 30, 35, 40, 45, 50],
            'income': [50000, 60000, 55000, 70000, 65000, 80000],
            'color': ['red', 'blue', 'red', 'green', 'blue', 'green'],
            'size': ['S', 'M', 'L', 'M', 'S', 'L'],
            'approved': ['yes', 'yes', 'no', 'yes', 'no', 'yes']
        })
        
        # Pure numeric dataset
        self.numeric_df = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5, 6],
            'x2': [2, 4, 6, 8, 10, 12],
            'y': ['A', 'A', 'B', 'B', 'A', 'B']
        })
    
    def test_initialization(self):
        """Test that tree initializes correctly"""
        tree = SimpleDecisionTree(self.simple_df, formula='class ~ .', max_depth=3)
        
        assert tree.target == 'class'  # Changed from target_col
        assert tree.max_depth == 3
        assert tree.criterion == 'entropy'
        assert len(tree.classes) == 2  # Changed from class_names
        assert tree.tree is not None  # Changed from clf
    
    def test_categorical_features_identification(self):
        """Test that categorical features are identified correctly"""
        tree = SimpleDecisionTree(self.mixed_df, formula='approved ~ .')
        
        assert 'age' in tree.numeric_features
        assert 'income' in tree.numeric_features
        assert 'color' in tree.categorical_features
        assert 'size' in tree.categorical_features
    
    def test_numeric_features_identification(self):
        """Test that numeric features are identified correctly"""
        tree = SimpleDecisionTree(self.numeric_df, formula='y ~ .')
        
        assert 'x1' in tree.numeric_features
        assert 'x2' in tree.numeric_features
        assert len(tree.categorical_features) == 0
    
    def test_one_hot_encoding(self):
        """Test that categorical features are one-hot encoded"""
        tree = SimpleDecisionTree(self.simple_df, formula='class ~ .')
        
        # Should have expanded categorical columns
        assert len(tree.feature_names) > len(tree.feature_names_original)
        assert any('feature1' in name for name in tree.feature_names)
        assert any('feature2' in name for name in tree.feature_names)
    
    def test_mixed_features_encoding(self):
        """Test that mixed features are encoded correctly"""
        tree = SimpleDecisionTree(self.mixed_df, formula='approved ~ .')
        
        # Numeric features should remain as-is
        assert 'age' in tree.feature_names
        assert 'income' in tree.feature_names
        
        # Categorical features should be one-hot encoded
        assert any('color' in name for name in tree.feature_names)
        assert any('size' in name for name in tree.feature_names)
    
    def test_target_encoding(self):
        """Test that target is label encoded"""
        tree = SimpleDecisionTree(self.simple_df, formula='class ~ .')
        
        # Target should be encoded as 0, 1
        assert set(tree.y_encoded) == {0, 1}
        assert len(tree.y_encoded) == len(self.simple_df)
    
    def test_predict_returns_original_labels(self):
        """Test that predictions return original class labels"""
        tree = SimpleDecisionTree(self.simple_df, formula='class ~ .')
        
        predictions = tree.predict(self.simple_df)  # Can include target now
        
        # Predictions should be original labels, not encoded
        assert all(pred in ['yes', 'no'] for pred in predictions)
        assert len(predictions) == len(self.simple_df)
    
    def test_score_method(self):
        """Test that score method returns accuracy"""
        tree = SimpleDecisionTree(self.simple_df, formula='class ~ .', max_depth=3)
        
        report = tree.score(self.simple_df)
        
        assert 0 <= report['accuracy'] <= 1
        assert isinstance(report['confusion_matrix'], np.ndarray)

    def test_get_metrics(self):
        """Test getting classification metrics"""
        tree = SimpleDecisionTree(self.simple_df, formula='class ~ .', max_depth=3)
        metrics = tree.get_metrics()

        assert metrics['model_type'] == 'decision_tree'
        assert 'score' in metrics
        assert 'accuracy' in metrics['score']
        assert 'confusion_matrix' in metrics['score']
        assert 'per_class' in metrics['score']
        assert 'kappa' in metrics['score']
        assert 0 <= metrics['score']['accuracy'] <= 1
        assert isinstance(metrics['score']['confusion_matrix'], np.ndarray)
        assert metrics['score']['confusion_matrix'].shape == (2, 2)
        assert metrics['classes'] == tree.classes
        assert 'feature_importance' in metrics
    
    def test_perfect_classification(self):
        """Test that deep tree can perfectly classify training data"""
        # Create perfectly separable data
        perfect_df = pd.DataFrame({
            'feature': ['A', 'A', 'B', 'B'],
            'class': ['yes', 'yes', 'no', 'no']
        })
        
        tree = SimpleDecisionTree(perfect_df, formula='class ~ .', max_depth=None)
        accuracy = tree.score(perfect_df)['accuracy']
        
        assert accuracy == 1.0

    def test_get_rules_single_leaf(self):
        """Test get_rules when tree has a single leaf node."""
        df = pd.DataFrame({
            'feature': [1, 1, 1, 1],
            'class': ['yes', 'no', 'yes', 'yes']
        })
        tree = SimpleDecisionTree(df, formula='class ~ .', max_depth=1)
        rules = tree.get_rules()

        assert "IF (Always) THEN Class=" in rules

    def test_summary(self, monkeypatch):
        """Test summary output"""
        monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
        
        tree = SimpleDecisionTree(self.simple_df, formula='class ~ .', max_depth=3)
        result = tree.summary()
        
        assert result is None
    
    def test_max_depth_constraint(self):
        """Test that max_depth is respected"""
        tree_shallow = SimpleDecisionTree(self.simple_df, formula='class ~ .', max_depth=1)
        tree_deep = SimpleDecisionTree(self.simple_df, formula='class ~ .', max_depth=5)
        
        assert tree_shallow.tree.get_depth() <= 1  # Changed from clf
        assert tree_deep.tree.get_depth() <= 5
    
    def test_criterion_entropy(self):
        """Test that entropy criterion works"""
        tree = SimpleDecisionTree(self.simple_df, formula='class ~ .', criterion='entropy')
        
        assert tree.criterion == 'entropy'
        assert tree.tree.criterion == 'entropy'
    
    def test_criterion_gini(self):
        """Test that gini criterion works"""
        tree = SimpleDecisionTree(self.simple_df, formula='class ~ .', criterion='gini')
        
        assert tree.criterion == 'gini'
        assert tree.tree.criterion == 'gini'
    
    def test_feature_importances(self):
        """Test that feature importances are calculated"""
        tree = SimpleDecisionTree(self.simple_df, formula='class ~ .')
        
        importances = tree.tree.feature_importances_
        
        assert len(importances) == len(tree.feature_names)
        assert all(0 <= imp <= 1 for imp in importances)
        assert np.isclose(np.sum(importances), 1.0)
    
    def test_predict_with_new_data(self):
        """Test predictions on new data"""
        tree = SimpleDecisionTree(self.simple_df, formula='class ~ .')
        
        new_data = pd.DataFrame({
            'feature1': ['A', 'B'],
            'feature2': ['X', 'Y']
        })
        
        predictions = tree.predict(new_data)
        
        assert len(predictions) == 2
        assert all(pred in ['yes', 'no'] for pred in predictions)

    def test_formula_fit_tree(self):
        """Test fitting with formula input"""
        df = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5, 6],
            'x2': [2, 1, 2, 1, 2, 1],
            'class': ['yes', 'yes', 'no', 'no', 'yes', 'no']
        })
        tree = fit_tree(df, formula='class ~ x1 + x2')

        assert tree.target == 'class'
        assert set(tree.feature_names) == {'x1', 'x2'}

        predictions = tree.predict(df[['x1', 'x2']])
        assert len(predictions) == len(df)

    def test_formula_visualize_features_runs(self, monkeypatch):
        """Test visualize_features runs with formula usage"""
        df = pd.DataFrame({
            'x1': [1, 2, 3, 4],
            'x2': [2, 3, 4, 5],
            'class': ['yes', 'yes', 'no', 'no']
        })
        tree = fit_tree(df, formula='class ~ x1 + x2')

        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        tree.visualize_features('x1', 'x2')
    
    def test_numeric_features_not_encoded(self):
        """Test that numeric features remain unchanged"""
        tree = SimpleDecisionTree(self.numeric_df, formula='y ~ .')
        
        # Original numeric values should be in X_encoded
        assert tree.X_encoded['x1'].tolist() == [1, 2, 3, 4, 5, 6]
        assert tree.X_encoded['x2'].tolist() == [2, 4, 6, 8, 10, 12]


class TestFitTree:
    """Tests for fit_tree function"""
    
    def test_fit_tree_returns_simple_decision_tree(self):
        """Test that fit_tree returns SimpleDecisionTree instance"""
        df = pd.DataFrame({
            'feature': ['A', 'B', 'A', 'B'],
            'class': ['yes', 'no', 'yes', 'no']
        })
        
        tree = fit_tree(df, formula='class ~ .')
        
        assert isinstance(tree, SimpleDecisionTree)
    
    def test_fit_tree_with_custom_params(self):
        """Test fit_tree with custom parameters"""
        df = pd.DataFrame({
            'feature': ['A', 'B', 'A', 'B'],
            'class': ['yes', 'no', 'yes', 'no']
        })
        
        tree = fit_tree(df, formula='class ~ .', max_depth=2, criterion='gini')
        
        assert tree.max_depth == 2
        assert tree.criterion == 'gini'


class TestGetFeatureImportance:
    """Tests for get_feature_importance method"""
    
    def test_returns_dataframe(self):
        """Test that method returns DataFrame"""
        df = pd.DataFrame({
            'f1': ['A', 'B', 'A', 'B'],
            'f2': ['X', 'Y', 'Y', 'X'],
            'class': ['yes', 'no', 'yes', 'no']
        })
        
        tree = fit_tree(df, formula='class ~ .')
        importance = tree.get_feature_importance()  # Now a method
        
        assert isinstance(importance, pd.DataFrame)
    
    def test_has_correct_columns(self):
        """Test that DataFrame has correct columns"""
        df = pd.DataFrame({
            'feature': ['A', 'B', 'A', 'B'],
            'class': ['yes', 'no', 'yes', 'no']
        })
        
        tree = fit_tree(df, formula='class ~ .')
        importance = tree.get_feature_importance()
        
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
    
    def test_sorted_by_importance(self):
        """Test that results are sorted by importance"""
        df = pd.DataFrame({
            'f1': ['A', 'B', 'A', 'B', 'C', 'C'],
            'f2': ['X', 'Y', 'Y', 'X', 'X', 'Y'],
            'f3': ['P', 'P', 'Q', 'Q', 'P', 'Q'],
            'class': ['yes', 'no', 'yes', 'no', 'yes', 'no']
        })
        
        tree = fit_tree(df, formula='class ~ .')
        importance = tree.get_feature_importance()
        
        # Check that importance values are in descending order
        importances = importance['importance'].tolist()
        assert importances == sorted(importances, reverse=True)


class TestGetRules:
    """Tests for get_rules method"""

    def test_returns_string(self):
        """Test that method returns a string"""
        df = pd.DataFrame({
            'feature': ['A', 'B', 'A', 'B'],
            'class': ['yes', 'no', 'yes', 'no']
        })

        tree = fit_tree(df, formula='class ~ .', max_depth=2)
        rules = tree.get_rules()  # Now a method

        assert isinstance(rules, str)
        assert len(rules) > 0

    def test_contains_feature_names(self):
        """Test that rules contain feature names"""
        df = pd.DataFrame({
            'my_feature': ['A', 'B', 'A', 'B'],
            'class': ['yes', 'no', 'yes', 'no']
        })

        tree = fit_tree(df, formula='class ~ .', max_depth=2)
        rules = tree.get_rules()

        # Should contain the one-hot encoded feature name
        assert 'my_feature' in rules

    def test_rules_in_sentence_format(self):
        """Test that rules are in IF-THEN sentence format"""
        df = pd.DataFrame({
            'age': [25, 30, 35, 40, 45, 50],
            'balance': [30000, 40000, 60000, 70000, 45000, 55000],
            'class': ['approved', 'approved', 'denied', 'denied', 'approved', 'denied']
        })

        tree = fit_tree(df, formula='class ~ .', max_depth=2)
        rules = tree.get_rules()

        # Check for IF-THEN format
        assert 'IF' in rules
        assert 'THEN' in rules
        assert 'Class=' in rules

    def test_rules_contain_logical_operators(self):
        """Test that rules contain AND operators for multiple conditions"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
            'feature2': [10, 20, 30, 40, 50, 60, 70, 80],
            'class': ['A', 'A', 'B', 'B', 'A', 'B', 'A', 'B']
        })

        tree = fit_tree(df, formula='class ~ .', max_depth=3)
        rules = tree.get_rules()

        # With depth > 1, should have AND operators
        assert 'AND' in rules

    def test_rules_contain_class_predictions(self):
        """Test that rules show predicted classes"""
        df = pd.DataFrame({
            'x': [1, 2, 3, 4],
            'status': ['pass', 'fail', 'pass', 'fail']
        })

        tree = fit_tree(df, formula='status ~ .', max_depth=1)
        rules = tree.get_rules()

        # Should contain both class labels
        assert 'Class=pass' in rules or 'Class=fail' in rules

    def test_rules_numbered(self):
        """Test that rules are numbered"""
        df = pd.DataFrame({
            'x': [1, 2, 3, 4],
            'y': ['A', 'B', 'A', 'B']
        })

        tree = fit_tree(df, formula='y ~ .', max_depth=2)
        rules = tree.get_rules()

        # Should have numbered rules
        assert 'Rule 1:' in rules

    def test_rules_contain_conditions(self):
        """Test that rules contain threshold conditions"""
        df = pd.DataFrame({
            'value': [10, 20, 30, 40, 50],
            'label': ['low', 'low', 'high', 'high', 'high']
        })

        tree = fit_tree(df, formula='label ~ .', max_depth=1)
        rules = tree.get_rules()

        # Should contain comparison operators
        assert '<=' in rules or '>' in rules


class TestVisualization:
    """Tests for visualization methods"""
    
    def test_visualize_method_exists(self):
        """Test that visualize method exists"""
        df = pd.DataFrame({
            'feature': ['A', 'B', 'A', 'B'],
            'class': ['yes', 'no', 'yes', 'no']
        })
        
        tree = fit_tree(df, formula='class ~ .')
        
        assert hasattr(tree, 'visualize')
    
    def test_visualize_features_method_exists(self):
        """Test that visualize_features method exists"""
        df = pd.DataFrame({
            'f1': ['A', 'B', 'A', 'B'],
            'f2': ['X', 'Y', 'Y', 'X'],
            'class': ['yes', 'no', 'yes', 'no']
        })
        
        tree = fit_tree(df, formula='class ~ .')
        
        assert hasattr(tree, 'visualize_features')
    
    def test_visualize_runs_without_error(self, monkeypatch):
        """Test that visualize runs without error"""
        df = pd.DataFrame({
            'feature': ['A', 'B', 'A', 'B'],
            'class': ['yes', 'no', 'yes', 'no']
        })
        
        tree = fit_tree(df, formula='class ~ .')
        
        # Mock plt.show() to prevent display
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        
        # Should not raise exception
        tree.visualize()
    
    def test_visualize_features_runs_without_error(self, monkeypatch):
        """Test that visualize_features runs without error"""
        df = pd.DataFrame({
            'f1': ['A', 'B', 'C', 'D'],
            'f2': ['X', 'Y', 'X', 'Y'],
            'class': ['yes', 'no', 'yes', 'no']
        })
        
        tree = fit_tree(df, formula='class ~ .')
        
        # Mock plt.show()
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        
        # Should not raise exception
        tree.visualize_features('f1', 'f2')


class TestWithMushroomData:
    """Integration tests using real mushroom dataset"""
    
    def test_mushroom_classification(self):
        """Test tree on mushroom dataset"""
        df = load_mushroom_data()
        
        tree = fit_tree(df, formula='class ~ .', max_depth=5)
        accuracy = tree.score(df)['accuracy']
        
        # Should get reasonable accuracy
        assert accuracy > 0.5
    
    def test_shallow_mushroom_tree(self):
        """Test shallow tree on mushroom data"""
        df = load_mushroom_data()
        
        tree = fit_tree(df, formula='class ~ .', max_depth=2)
        
        assert tree.tree.get_depth() <= 2
    
    def test_feature_importance_on_mushroom(self):
        """Test feature importance on mushroom data"""
        df = load_mushroom_data()
        
        tree = fit_tree(df, formula='class ~ .', max_depth=3)
        importance = tree.get_feature_importance()
        
        # Should have importance scores for all features
        assert len(importance) > 0
        assert all(importance['importance'] >= 0)
    
    def test_train_test_split(self):
        """Test with train/test split"""
        df = load_mushroom_data()
        
        train, test = split_data(df, test_size=0.2)
        tree = fit_tree(train, formula='class ~ .', max_depth=5)
        
        train_acc = tree.score(train)['accuracy']
        test_acc = tree.score(test)['accuracy']
        
        assert train_acc > 0.5
        assert test_acc > 0.5
        assert 0 <= train_acc <= 1
        assert 0 <= test_acc <= 1


class TestEdgeCases:
    """Tests for edge cases and error handling"""
    
    def test_single_feature(self):
        """Test tree with single feature"""
        df = pd.DataFrame({
            'feature': ['A', 'B', 'A', 'B'],
            'class': ['yes', 'no', 'yes', 'no']
        })
        
        tree = fit_tree(df, formula='class ~ .')
        
        assert tree is not None
        assert tree.score(df)['accuracy'] > 0
    
    def test_binary_target(self):
        """Test with binary target"""
        df = pd.DataFrame({
            'feature': ['A', 'B', 'C', 'D'],
            'class': [0, 1, 0, 1]
        })
        
        tree = fit_tree(df, formula='class ~ .')
        predictions = tree.predict(df)
        
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_multiclass_target(self):
        """Test with multi-class target"""
        df = pd.DataFrame({
            'feature': ['A', 'B', 'C', 'D', 'E', 'F'],
            'class': ['X', 'Y', 'Z', 'X', 'Y', 'Z']
        })
        
        tree = fit_tree(df, formula='class ~ .')
        
        assert len(tree.classes) == 3
        assert set(tree.classes) == {'X', 'Y', 'Z'}
    
    def test_predict_with_missing_encoded_columns(self):
        """Test that prediction handles missing one-hot columns"""
        df_train = pd.DataFrame({
            'color': ['red', 'blue', 'green'],
            'class': ['A', 'B', 'A']
        })
        
        tree = fit_tree(df_train, formula='class ~ .')
        
        # Test data missing 'green' category
        df_test = pd.DataFrame({
            'color': ['red', 'blue']
        })
        
        predictions = tree.predict(df_test)
        
        assert len(predictions) == 2
        assert all(pred in ['A', 'B'] for pred in predictions)

class TestDecisionTreeMissingCoverage:
    """Tests to cover missing branches in decision tree code"""
    
    def test_check_fitted_raises_error(self):
        """Test _check_fitted raises RuntimeError when tree not fitted (lines 106-109)"""
        # Create an unfitted instance
        df = pd.DataFrame({
            'feature': ['A', 'B', 'C'],
            'class': ['X', 'Y', 'X']
        })
        
        tree = SimpleDecisionTree.__new__(SimpleDecisionTree)
        tree.df = df
        tree.target = 'class'
        # Don't set tree.tree - this makes it unfitted
        
        with pytest.raises(RuntimeError, match="Model has not been fitted"):
            tree.predict(df)
    
    def test_predict_with_numeric_features(self):
        """Test predict with numeric features (line 129)"""
        # Current tests only use categorical features
        # Need a test with numeric features
        df = pd.DataFrame({
            'height': [1.5, 2.0, 1.8, 2.2, 1.6],
            'weight': [50, 70, 60, 80, 55],
            'class': ['A', 'B', 'A', 'B', 'A']
        })
        
        tree = fit_tree(df, formula='class ~ .', max_depth=2)
        
        # Predict on new numeric data
        test_df = pd.DataFrame({
            'height': [1.7, 2.1],
            'weight': [58, 75],
            'class': ['A', 'B']  # Include target to test
        })
        
        predictions = tree.predict(test_df)
        assert len(predictions) == 2
    
    def test_predict_with_mixed_features_and_numeric(self):
        """Test predict path with both numeric and categorical (line 149)"""
        df = pd.DataFrame({
            'numeric_feat': [1, 2, 3, 4, 5],
            'category': ['A', 'B', 'A', 'B', 'A'],
            'class': ['X', 'Y', 'X', 'Y', 'X']
        })
        
        tree = fit_tree(df, formula='class ~ .', max_depth=2)
        
        # Predict - should trigger line 149 (concat with numeric features)
        test_df = pd.DataFrame({
            'numeric_feat': [2.5, 3.5],
            'category': ['A', 'B']
        })
        
        predictions = tree.predict(test_df)
        assert len(predictions) == 2
    
    def test_visualize_with_probabilities(self, monkeypatch):
        """Test visualize with show_probabilities=True (lines 245-255)"""
        df = pd.DataFrame({
            'feature': ['A', 'B', 'A', 'B'],
            'class': ['X', 'Y', 'X', 'Y']
        })
        
        tree = fit_tree(df, formula='class ~ .', max_depth=2)
        
        # Mock plt.show()
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        
        # Call with show_probabilities=True
        tree.visualize(show_probabilities=True)
    
    def test_visualize_features_with_defaults(self, monkeypatch):
        """Test visualize_features with default feature selection (lines 302-306)"""
        # Need at least 2 features
        df = pd.DataFrame({
            'feature1': ['A', 'B', 'A', 'B', 'A'],
            'feature2': ['X', 'Y', 'X', 'Y', 'X'],
            'class': ['P', 'N', 'P', 'N', 'P']
        })
        
        tree = fit_tree(df, formula='class ~ .', max_depth=2)
        
        # Mock plt.show()
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        
        # Call without specifying features - should use first two
        tree.visualize_features()
    
    def test_visualize_features_with_less_than_2_features(self):
        """Test visualize_features error with < 2 features (line 306)"""
        df = pd.DataFrame({
            'feature1': ['A', 'B', 'A', 'B'],
            'class': ['X', 'Y', 'X', 'Y']
        })
        
        tree = fit_tree(df, formula='class ~ .', max_depth=2)
        
        # Should raise error - need at least 2 features
        with pytest.raises(ValueError, match="Need at least 2 features"):
            tree.visualize_features()
    
    def test_visualize_features_with_invalid_features(self):
        """Test visualize_features with invalid feature names (lines 309-312)"""
        df = pd.DataFrame({
            'feature1': ['A', 'B', 'A', 'B'],
            'feature2': ['X', 'Y', 'X', 'Y'],
            'class': ['P', 'N', 'P', 'N']
        })
        
        tree = fit_tree(df, formula='class ~ .', max_depth=2)
        
        # Provide invalid feature names
        with pytest.raises(ValueError, match="Features must be from"):
            tree.visualize_features('invalid1', 'invalid2')
    
    def test_visualize_features_with_numeric_features(self, monkeypatch):
        """Test visualize_features with numeric features (lines 320-321, 329-330, etc.)"""
        df = pd.DataFrame({
            'height': [1.5, 2.0, 1.8, 2.2, 1.6, 1.9, 2.1, 1.7],
            'weight': [50, 70, 60, 80, 55, 65, 75, 58],
            'class': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
        })
        
        tree = fit_tree(df, formula='class ~ .', max_depth=2)
        
        # Mock plt.show()
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        
        # Call with numeric features
        tree.visualize_features('height', 'weight')
    
    def test_visualize_features_mesh_with_numeric_in_base_data(self, monkeypatch):
        """Test visualize_features mesh creation with numeric features (line 355)"""
        df = pd.DataFrame({
            'num1': [1, 2, 3, 4, 5],
            'num2': [10, 20, 30, 40, 50],
            'cat1': ['A', 'B', 'A', 'B', 'A'],
            'cat2': ['X', 'Y', 'X', 'Y', 'X'],
            'class': ['P', 'N', 'P', 'N', 'P']
        })
        
        tree = fit_tree(df, formula='class ~ .', max_depth=2)
        
        # Mock plt.show()
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        
        # Visualize two categorical features
        # This will use numeric median for num1 and num2 in base_data (triggers line 355)
        tree.visualize_features('cat1', 'cat2')
    
    def test_visualize_features_mesh_replacement_numeric(self, monkeypatch):
        """Test mesh replacement for numeric features (lines 363, 370)"""
        df = pd.DataFrame({
            'num1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'num2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'class': ['X', 'Y', 'X', 'Y', 'X']
        })
        
        tree = fit_tree(df, formula='class ~ .', max_depth=2)
        
        # Mock plt.show()
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        
        # This should trigger numeric mesh replacement (lines 363, 370)
        tree.visualize_features('num1', 'num2')
    
    def test_visualize_features_mesh_encoding_with_numeric(self, monkeypatch):
        """Test mesh encoding with numeric features present (lines 379, 399)"""
        df = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5, 6],
            'cat1': ['A', 'B', 'A', 'B', 'A', 'B'],
            'cat2': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
            'class': ['P', 'N', 'P', 'N', 'P', 'N']
        })
        
        tree = fit_tree(df, formula='class ~ .', max_depth=2)
        
        # Mock plt.show()
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        
        # Visualize categorical vs categorical
        # This creates mesh with numeric features present (triggers line 379, 399)
        tree.visualize_features('cat1', 'cat2')
    
    def test_visualize_features_missing_cols_in_mesh(self, monkeypatch):
        """Test mesh with missing one-hot columns (lines 394-395)"""
        # This is tricky - need categorical data where mesh doesn't have all categories
        df = pd.DataFrame({
            'cat1': ['A', 'B', 'C'] * 3,
            'cat2': ['X', 'Y', 'Z'] * 3,
            'class': ['P', 'N', 'P', 'N', 'P', 'N', 'P', 'N', 'P']
        })
        
        tree = fit_tree(df, formula='class ~ .', max_depth=2)
        
        # Mock plt.show()
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        
        # Visualize - mesh grid might not have all combinations
        tree.visualize_features('cat1', 'cat2')
    
    def test_visualize_features_skip_tick_labels_long_list(self, monkeypatch):
        """Test visualize_features skips tick labels when >10 categories (lines 438, 442)"""
        # Create data with more than 10 categories
        categories1 = [f'Cat{i}' for i in range(15)]
        categories2 = [f'Type{i}' for i in range(15)]
        
        df = pd.DataFrame({
            'feature1': categories1 * 2,
            'feature2': categories2 * 2,
            'class': ['X', 'Y'] * 15
        })
        
        tree = fit_tree(df, formula='class ~ .', max_depth=2)
        
        # Mock plt.show()
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        
        # Should skip setting tick labels (>10 categories)
        tree.visualize_features('feature1', 'feature2')

    def test_visualize_features_missing_onehot_column(self, monkeypatch):
        """Test visualize_features when mesh is missing a one-hot column (lines 394-395)"""
        # Create data where one category won't appear in mesh
        df = pd.DataFrame({
            'cat1': ['A', 'A', 'A', 'B'],  # 'B' is rare
            'cat2': ['X', 'Y', 'X', 'Y'],
            'class': ['P', 'N', 'P', 'N']
        })
        
        tree = fit_tree(df, formula='class ~ .', max_depth=2)
        
        # Mock plt.show()
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        
        # When visualizing, the mesh grid might not include 'B' 
        # because it's created from a range that might not hit it
        tree.visualize_features('cat1', 'cat2')

    def test_visualize_features_all_onehot_columns_present(self, monkeypatch):
        """Test when all one-hot columns are present (line 392 else path)"""
        df = pd.DataFrame({
            'cat1': ['A', 'B', 'A', 'B'],
            'cat2': ['X', 'Y', 'X', 'Y'],
            'class': ['P', 'N', 'P', 'N']
        })
        
        tree = fit_tree(df, formula='class ~ .', max_depth=2)
        
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        
        # All categories should be present in mesh, no missing columns
        tree.visualize_features('cat1', 'cat2')

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
