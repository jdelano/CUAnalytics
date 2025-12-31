"""
Tests for decision tree module
"""
import pytest
import pandas as pd
import numpy as np
from itm4150.trees import build_tree, visualize_tree, get_feature_importance, get_tree_rules
from itm4150.trees.decision_tree import SimpleDecisionTree
from itm4150.datasets import load_mushroom_data


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
        tree = SimpleDecisionTree(self.simple_df, 'class', max_depth=3)
        
        assert tree.target_col == 'class'
        assert tree.max_depth == 3
        assert tree.criterion == 'entropy'
        assert len(tree.class_names) == 2
        assert tree.clf is not None
    
    def test_categorical_features_identification(self):
        """Test that categorical features are identified correctly"""
        tree = SimpleDecisionTree(self.mixed_df, 'approved')
        
        assert 'age' in tree.numeric_features
        assert 'income' in tree.numeric_features
        assert 'color' in tree.categorical_features
        assert 'size' in tree.categorical_features
    
    def test_numeric_features_identification(self):
        """Test that numeric features are identified correctly"""
        tree = SimpleDecisionTree(self.numeric_df, 'y')
        
        assert 'x1' in tree.numeric_features
        assert 'x2' in tree.numeric_features
        assert len(tree.categorical_features) == 0
    
    def test_one_hot_encoding(self):
        """Test that categorical features are one-hot encoded"""
        tree = SimpleDecisionTree(self.simple_df, 'class')
        
        # Should have one-hot encoded columns
        assert 'feature1_A' in tree.feature_names
        assert 'feature1_B' in tree.feature_names
        assert 'feature1_C' in tree.feature_names
        assert 'feature2_X' in tree.feature_names
        assert 'feature2_Y' in tree.feature_names
    
    def test_mixed_features_encoding(self):
        """Test that mixed features are encoded correctly"""
        tree = SimpleDecisionTree(self.mixed_df, 'approved')
        
        # Numeric features should remain as-is
        assert 'age' in tree.feature_names
        assert 'income' in tree.feature_names
        
        # Categorical features should be one-hot encoded
        assert 'color_red' in tree.feature_names
        assert 'color_blue' in tree.feature_names
        assert 'size_S' in tree.feature_names
        assert 'size_M' in tree.feature_names
    
    def test_target_encoding(self):
        """Test that target is label encoded"""
        tree = SimpleDecisionTree(self.simple_df, 'class')
        
        # Target should be encoded as 0, 1
        assert set(tree.y_encoded) == {0, 1}
        assert len(tree.y_encoded) == len(self.simple_df)
    
    def test_predict_returns_original_labels(self):
        """Test that predictions return original class labels"""
        tree = SimpleDecisionTree(self.simple_df, 'class')
        
        predictions = tree.predict(self.simple_df.drop('class', axis=1))
        
        # Predictions should be original labels, not encoded
        assert all(pred in ['yes', 'no'] for pred in predictions)
        assert len(predictions) == len(self.simple_df)
    
    def test_score_method(self):
        """Test that score method returns accuracy"""
        tree = SimpleDecisionTree(self.simple_df, 'class', max_depth=3)
        
        accuracy = tree.score(self.simple_df)
        
        assert 0 <= accuracy <= 1
        assert isinstance(accuracy, (float, np.floating))
    
    def test_perfect_classification(self):
        """Test that deep tree can perfectly classify training data"""
        # Create perfectly separable data
        perfect_df = pd.DataFrame({
            'feature': ['A', 'A', 'B', 'B'],
            'class': ['yes', 'yes', 'no', 'no']
        })
        
        tree = SimpleDecisionTree(perfect_df, 'class', max_depth=None)
        accuracy = tree.score(perfect_df)
        
        assert accuracy == 1.0
    
    def test_max_depth_constraint(self):
        """Test that max_depth is respected"""
        tree_shallow = SimpleDecisionTree(self.simple_df, 'class', max_depth=1)
        tree_deep = SimpleDecisionTree(self.simple_df, 'class', max_depth=5)
        
        assert tree_shallow.clf.get_depth() <= 1
        assert tree_deep.clf.get_depth() <= 5
    
    def test_criterion_entropy(self):
        """Test that entropy criterion works"""
        tree = SimpleDecisionTree(self.simple_df, 'class', criterion='entropy')
        
        assert tree.criterion == 'entropy'
        assert tree.clf.criterion == 'entropy'
    
    def test_criterion_gini(self):
        """Test that gini criterion works"""
        tree = SimpleDecisionTree(self.simple_df, 'class', criterion='gini')
        
        assert tree.criterion == 'gini'
        assert tree.clf.criterion == 'gini'
    
    def test_feature_importances(self):
        """Test that feature importances are calculated"""
        tree = SimpleDecisionTree(self.simple_df, 'class')
        
        importances = tree.clf.feature_importances_
        
        assert len(importances) == len(tree.feature_names)
        assert all(0 <= imp <= 1 for imp in importances)
        assert np.isclose(np.sum(importances), 1.0)
    
    def test_predict_with_new_data(self):
        """Test predictions on new data"""
        tree = SimpleDecisionTree(self.simple_df, 'class')
        
        new_data = pd.DataFrame({
            'feature1': ['A', 'B'],
            'feature2': ['X', 'Y']
        })
        
        predictions = tree.predict(new_data)
        
        assert len(predictions) == 2
        assert all(pred in ['yes', 'no'] for pred in predictions)
    
    def test_numeric_features_not_encoded(self):
        """Test that numeric features remain unchanged"""
        tree = SimpleDecisionTree(self.numeric_df, 'y')
        
        # Original numeric values should be in X_encoded
        assert tree.X_encoded['x1'].tolist() == [1, 2, 3, 4, 5, 6]
        assert tree.X_encoded['x2'].tolist() == [2, 4, 6, 8, 10, 12]


class TestBuildTree:
    """Tests for build_tree function"""
    
    def test_build_tree_returns_simple_decision_tree(self):
        """Test that build_tree returns SimpleDecisionTree instance"""
        df = pd.DataFrame({
            'feature': ['A', 'B', 'A', 'B'],
            'class': ['yes', 'no', 'yes', 'no']
        })
        
        tree = build_tree(df, target='class')
        
        assert isinstance(tree, SimpleDecisionTree)
    
    def test_build_tree_with_custom_params(self):
        """Test build_tree with custom parameters"""
        df = pd.DataFrame({
            'feature': ['A', 'B', 'A', 'B'],
            'class': ['yes', 'no', 'yes', 'no']
        })
        
        tree = build_tree(df, target='class', max_depth=2, criterion='gini')
        
        assert tree.max_depth == 2
        assert tree.criterion == 'gini'


class TestGetFeatureImportance:
    """Tests for get_feature_importance function"""
    
    def test_returns_dataframe(self):
        """Test that function returns DataFrame"""
        df = pd.DataFrame({
            'f1': ['A', 'B', 'A', 'B'],
            'f2': ['X', 'Y', 'Y', 'X'],
            'class': ['yes', 'no', 'yes', 'no']
        })
        
        tree = build_tree(df, target='class')
        importance = get_feature_importance(tree)
        
        assert isinstance(importance, pd.DataFrame)
    
    def test_has_correct_columns(self):
        """Test that DataFrame has correct columns"""
        df = pd.DataFrame({
            'feature': ['A', 'B', 'A', 'B'],
            'class': ['yes', 'no', 'yes', 'no']
        })
        
        tree = build_tree(df, target='class')
        importance = get_feature_importance(tree)
        
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
        
        tree = build_tree(df, target='class')
        importance = get_feature_importance(tree)
        
        # Check that importance values are in descending order
        importances = importance['importance'].tolist()
        assert importances == sorted(importances, reverse=True)


class TestGetTreeRules:
    """Tests for get_tree_rules function"""
    
    def test_returns_string(self):
        """Test that function returns a string"""
        df = pd.DataFrame({
            'feature': ['A', 'B', 'A', 'B'],
            'class': ['yes', 'no', 'yes', 'no']
        })
        
        tree = build_tree(df, target='class', max_depth=2)
        rules = get_tree_rules(tree)
        
        assert isinstance(rules, str)
        assert len(rules) > 0
    
    def test_contains_feature_names(self):
        """Test that rules contain feature names"""
        df = pd.DataFrame({
            'my_feature': ['A', 'B', 'A', 'B'],
            'class': ['yes', 'no', 'yes', 'no']
        })
        
        tree = build_tree(df, target='class', max_depth=2)
        rules = get_tree_rules(tree)
        
        # Should contain the one-hot encoded feature name
        assert 'my_feature' in rules


class TestWithMushroomData:
    """Integration tests using real mushroom dataset"""
    
    def test_mushroom_classification(self):
        """Test tree on mushroom dataset"""
        df = load_mushroom_data()
        
        tree = build_tree(df, target='class', max_depth=5)
        accuracy = tree.score(df)
        
        # Should get reasonable accuracy
        assert accuracy > 0.5
    
    def test_shallow_mushroom_tree(self):
        """Test shallow tree on mushroom data"""
        df = load_mushroom_data()
        
        tree = build_tree(df, target='class', max_depth=2)
        
        assert tree.clf.get_depth() <= 2
    
    def test_feature_importance_on_mushroom(self):
        """Test feature importance on mushroom data"""
        df = load_mushroom_data()
        
        tree = build_tree(df, target='class', max_depth=3)
        importance = get_feature_importance(tree)
        
        # Should have importance scores for all features
        assert len(importance) > 0
        assert all(importance['importance'] >= 0)


class TestEdgeCases:
    """Tests for edge cases and error handling"""
    
    def test_single_feature(self):
        """Test tree with single feature"""
        df = pd.DataFrame({
            'feature': ['A', 'B', 'A', 'B'],
            'class': ['yes', 'no', 'yes', 'no']
        })
        
        tree = build_tree(df, target='class')
        
        assert tree is not None
        assert tree.score(df) > 0
    
    def test_binary_target(self):
        """Test with binary target"""
        df = pd.DataFrame({
            'feature': ['A', 'B', 'C', 'D'],
            'class': [0, 1, 0, 1]
        })
        
        tree = build_tree(df, target='class')
        predictions = tree.predict(df.drop('class', axis=1))
        
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_multiclass_target(self):
        """Test with multi-class target"""
        df = pd.DataFrame({
            'feature': ['A', 'B', 'C', 'D', 'E', 'F'],
            'class': ['X', 'Y', 'Z', 'X', 'Y', 'Z']
        })
        
        tree = build_tree(df, target='class')
        
        assert len(tree.class_names) == 3
        assert set(tree.class_names) == {'X', 'Y', 'Z'}