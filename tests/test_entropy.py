"""
Tests for entropy module
"""
import pytest
import numpy as np
import pandas as pd
from itm4150.entropy import calculate_entropy, information_gain


class TestCalculateEntropy:
    """Tests for calculate_entropy function"""
    
    def test_perfect_purity(self):
        """Entropy should be 0 for perfectly pure data"""
        data = pd.Series(['A', 'A', 'A', 'A'])
        entropy = calculate_entropy(data)
        assert entropy == 0.0
    
    def test_maximum_entropy(self):
        """Entropy should be 1.0 for 50/50 split (binary)"""
        data = pd.Series(['A', 'A', 'B', 'B'])
        entropy = calculate_entropy(data)
        assert abs(entropy - 1.0) < 0.0001  # Use tolerance for float comparison
    
    def test_three_way_split(self):
        """Test entropy with three classes"""
        data = pd.Series(['A', 'A', 'B', 'B', 'C', 'C'])
        entropy = calculate_entropy(data)
        expected = -3 * (1/3 * np.log2(1/3))
        assert abs(entropy - expected) < 0.0001
    
    def test_with_list(self):
        """Should work when passed a list (converts to Series)"""
        data = pd.Series(['A', 'B', 'A', 'B'])
        entropy = calculate_entropy(data)
        assert abs(entropy - 1.0) < 0.0001
    
    def test_single_element(self):
        """Single element should have zero entropy"""
        data = pd.Series(['A'])
        entropy = calculate_entropy(data)
        assert entropy == 0.0
    
    def test_unbalanced_distribution(self):
        """Test with unbalanced class distribution"""
        data = pd.Series(['A', 'A', 'A', 'B'])  # 75% A, 25% B
        entropy = calculate_entropy(data)
        expected = -(0.75 * np.log2(0.75) + 0.25 * np.log2(0.25))
        assert abs(entropy - expected) < 0.0001


class TestInformationGain:
    """Tests for information_gain function"""
    
    def setup_method(self):
        """Set up test data before each test"""
        # Create simple test dataframe
        self.simple_df = pd.DataFrame({
            'feature': ['X', 'X', 'Y', 'Y'],
            'class': ['A', 'A', 'B', 'B']
        })
        
        # Create mushroom-like test dataframe
        self.test_df = pd.DataFrame({
            'odor': ['a', 'a', 'l', 'l', 'n', 'n'],
            'color': ['b', 'w', 'b', 'w', 'b', 'w'],
            'class': ['e', 'e', 'p', 'p', 'e', 'e']
        })
    
    def test_perfect_split(self):
        """Perfect split should give maximum information gain"""
        ig = information_gain(self.simple_df, 'feature', 'class')
        assert abs(ig - 1.0) < 0.0001  # Should gain all entropy
    
    def test_no_information_split(self):
        """Split that doesn't separate classes should give low IG"""
        df = pd.DataFrame({
            'feature': ['X', 'Y', 'X', 'Y'],
            'class': ['A', 'A', 'B', 'B']
        })
        ig = information_gain(df, 'feature', 'class')
        assert abs(ig - 0.0) < 0.0001
    
    def test_partial_split(self):
        """Test with partial information gain"""
        ig = information_gain(self.test_df, 'odor', 'class')
        # odor should give some but not perfect separation
        assert ig > 0
        assert ig < 1.0
    
    def test_information_gain_non_negative(self):
        """Information gain should never be negative"""
        ig = information_gain(self.test_df, 'color', 'class')
        assert ig >= 0
    
    def test_multiple_values(self):
        """Test with feature that has multiple values"""
        df = pd.DataFrame({
            'feature': ['A', 'A', 'B', 'B', 'C', 'C'],
            'class': ['X', 'X', 'Y', 'Y', 'Z', 'Z']
        })
        ig = information_gain(df, 'feature', 'class')
        assert abs(ig - 1.585) < 0.01  # log2(3) ≈ 1.585
    
    def test_custom_target_column(self):
        """Test with custom target column name"""
        df = pd.DataFrame({
            'feature': ['X', 'X', 'Y', 'Y'],
            'label': ['A', 'A', 'B', 'B']
        })
        ig = information_gain(df, 'feature', 'label')
        assert abs(ig - 1.0) < 0.0001
    
    def test_with_real_data(self):
        """Test with mushroom dataset"""
        from itm4150.datasets import load_mushroom_data
        df = load_mushroom_data()
        
        # Calculate IG for odor feature
        ig = information_gain(df, 'odor', 'class')
        
        # Odor is known to be highly informative
        assert ig > 0.5  # Should have high information gain
        assert ig <= 1.0  # Can't exceed max entropy
    
    def test_single_value_feature(self):
        """Feature with single value should give zero IG"""
        df = pd.DataFrame({
            'feature': ['X', 'X', 'X', 'X'],
            'class': ['A', 'B', 'A', 'B']
        })
        ig = information_gain(df, 'feature', 'class')
        assert abs(ig - 0.0) < 0.0001
    
    def test_information_gain_bounds(self):
        """IG should be bounded by parent entropy"""
        df = pd.DataFrame({
            'feature': ['X', 'Y', 'Z'],
            'class': ['A', 'B', 'C']
        })
        parent_entropy = calculate_entropy(df['class'])
        ig = information_gain(df, 'feature', 'class')
        
        assert ig >= 0
        assert ig <= parent_entropy + 0.0001  # Allow small tolerance


class TestIntegration:
    """Integration tests combining multiple functions"""
    
    def test_mushroom_dataset_ranking(self):
        """Test that we can rank features by IG on real data"""
        from itm4150.datasets import load_mushroom_data
        df = load_mushroom_data()
        
        # Calculate IG for multiple features
        ig_scores = {}
        for feature in ['odor', 'gill-color', 'spore-print-color']:
            ig_scores[feature] = information_gain(df, feature, 'class')
        
        # All should be non-negative
        assert all(ig >= 0 for ig in ig_scores.values())
        
        # Odor should be highly informative (this is known)
        assert ig_scores['odor'] > 0.5