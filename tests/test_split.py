"""
Tests for data splitting utilities
"""
import pytest
import pandas as pd
import numpy as np
from itm4150.preprocessing import split_data
from itm4150.datasets import load_mushroom_data


class TestSplitDataBasic:
    """Basic tests for split_data function"""
    
    def setup_method(self):
        """Set up test data before each test"""
        # Simple test dataset
        self.simple_df = pd.DataFrame({
            'feature1': range(100),
            'feature2': range(100, 200),
            'class': ['A'] * 50 + ['B'] * 50
        })
        
        # Imbalanced dataset
        self.imbalanced_df = pd.DataFrame({
            'feature': range(100),
            'class': ['A'] * 80 + ['B'] * 20
        })
    
    def test_basic_split_returns_two_dataframes(self):
        """Test that basic split returns two DataFrames"""
        train, test = split_data(self.simple_df, target='class')
        
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
    
    def test_basic_split_sizes(self):
        """Test that split sizes are correct (default 80/20)"""
        train, test = split_data(self.simple_df, target='class', test_size=0.2)
        
        assert len(train) == 80
        assert len(test) == 20
        assert len(train) + len(test) == len(self.simple_df)
    
    def test_custom_test_size(self):
        """Test custom test_size parameter"""
        train, test = split_data(self.simple_df, target='class', test_size=0.3)
        
        assert len(test) == 30
        assert len(train) == 70
    
    def test_no_data_loss(self):
        """Test that no data is lost in splitting"""
        train, test = split_data(self.simple_df, target='class')
        
        assert len(train) + len(test) == len(self.simple_df)
    
    def test_no_overlap(self):
        """Test that train and test sets don't overlap"""
        train, test = split_data(self.simple_df, target='class')
        
        # Check indices don't overlap
        train_indices = set(train.index)
        test_indices = set(test.index)
        
        assert len(train_indices.intersection(test_indices)) == 0
    
    def test_random_state_reproducibility(self):
        """Test that same random_state gives same split"""
        train1, test1 = split_data(self.simple_df, target='class', random_state=42)
        train2, test2 = split_data(self.simple_df, target='class', random_state=42)
        
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(test1, test2)
    
    def test_different_random_state_different_split(self):
        """Test that different random_state gives different split"""
        train1, test1 = split_data(self.simple_df, target='class', random_state=42)
        train2, test2 = split_data(self.simple_df, target='class', random_state=99)
        
        # Indices should be different
        assert not train1.index.equals(train2.index)


class TestSplitDataStratification:
    """Tests for stratification in split_data"""
    
    def setup_method(self):
        """Set up test data"""
        self.balanced_df = pd.DataFrame({
            'feature': range(100),
            'class': ['A'] * 50 + ['B'] * 50
        })
        
        self.imbalanced_df = pd.DataFrame({
            'feature': range(100),
            'class': ['A'] * 80 + ['B'] * 20
        })
    
    def test_stratification_maintains_proportions(self):
        """Test that stratification maintains class proportions"""
        train, test = split_data(self.imbalanced_df, target='class', stratify=True)
        
        # Original: 80% A, 20% B
        original_ratio = self.imbalanced_df['class'].value_counts(normalize=True)
        train_ratio = train['class'].value_counts(normalize=True)
        test_ratio = test['class'].value_counts(normalize=True)
        
        # Ratios should be approximately the same (within 5%)
        assert abs(original_ratio['A'] - train_ratio['A']) < 0.05
        assert abs(original_ratio['A'] - test_ratio['A']) < 0.05
    
    def test_stratification_with_balanced_data(self):
        """Test stratification with balanced classes"""
        train, test = split_data(self.balanced_df, target='class', stratify=True)
        
        # Should be approximately 50/50 in both sets
        train_counts = train['class'].value_counts()
        test_counts = test['class'].value_counts()
        
        assert abs(train_counts['A'] - train_counts['B']) <= 1
        assert abs(test_counts['A'] - test_counts['B']) <= 1
    
    def test_no_stratification(self):
        """Test that stratify=False works"""
        # This should not raise an error
        train, test = split_data(self.balanced_df, target='class', stratify=False)
        
        assert len(train) + len(test) == len(self.balanced_df)
    
    def test_stratification_without_target(self):
        """Test that stratify is ignored when no target specified"""
        # Should work without error even though stratify=True
        train, test = split_data(self.balanced_df, stratify=True)
        
        assert len(train) + len(test) == len(self.balanced_df)


class TestSplitDataThreeWay:
    """Tests for three-way splits (train/test/val)"""
    
    def setup_method(self):
        """Set up test data"""
        self.df = pd.DataFrame({
            'feature': range(100),
            'class': ['A'] * 50 + ['B'] * 50
        })
    
    def test_three_way_split_returns_three_dataframes(self):
        """Test that three-way split returns three DataFrames"""
        train, test, val = split_data(self.df, target='class', test_size=0.2, val_size=0.2)
        
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        assert isinstance(val, pd.DataFrame)
    
    def test_three_way_split_sizes(self):
        """Test that three-way split sizes are correct"""
        train, test, val = split_data(self.df, target='class', test_size=0.2, val_size=0.2)
        
        # Should be 60/20/20
        assert len(train) == 60
        assert len(test) == 20
        assert len(val) == 20
    
    def test_three_way_no_overlap(self):
        """Test that train/test/val sets don't overlap"""
        train, test, val = split_data(self.df, target='class', test_size=0.2, val_size=0.2)
        
        train_indices = set(train.index)
        test_indices = set(test.index)
        val_indices = set(val.index)
        
        # No overlaps
        assert len(train_indices.intersection(test_indices)) == 0
        assert len(train_indices.intersection(val_indices)) == 0
        assert len(test_indices.intersection(val_indices)) == 0
    
    def test_three_way_no_data_loss(self):
        """Test that no data is lost in three-way split"""
        train, test, val = split_data(self.df, target='class', test_size=0.2, val_size=0.2)
        
        assert len(train) + len(test) + len(val) == len(self.df)
    
    def test_three_way_custom_sizes(self):
        """Test three-way split with custom sizes"""
        train, test, val = split_data(self.df, target='class', test_size=0.15, val_size=0.15)
        
        # Should be approximately 70/15/15 (allow for rounding)
        assert abs(len(test) - 15) <= 1  # Within 1 sample
        assert abs(len(val) - 15) <= 1   # Within 1 sample
        assert abs(len(train) - 70) <= 2  # Within 2 samples
        
        # Verify no data loss
        assert len(train) + len(test) + len(val) == len(self.df)
    
    def test_three_way_stratification(self):
        """Test that three-way split maintains stratification"""
        train, test, val = split_data(self.df, target='class', test_size=0.2, val_size=0.2, stratify=True)
        
        # All splits should have approximately 50/50 A/B
        for split in [train, test, val]:
            counts = split['class'].value_counts()
            # Within 10% of expected (since samples are small)
            assert abs(counts['A'] - len(split)/2) <= len(split) * 0.1


class TestSplitDataValidation:
    """Tests for input validation and error handling"""
    
    def setup_method(self):
        """Set up test data"""
        self.df = pd.DataFrame({
            'feature': range(100),
            'class': ['A'] * 50 + ['B'] * 50
        })
    
    def test_invalid_test_size_too_small(self):
        """Test that test_size <= 0 raises error"""
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            split_data(self.df, test_size=0)
    
    def test_invalid_test_size_too_large(self):
        """Test that test_size >= 1 raises error"""
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            split_data(self.df, test_size=1.0)
    
    def test_invalid_val_size_too_small(self):
        """Test that val_size <= 0 raises error"""
        with pytest.raises(ValueError, match="val_size must be between 0 and 1"):
            split_data(self.df, test_size=0.2, val_size=0)
    
    def test_invalid_val_size_too_large(self):
        """Test that val_size >= 1 raises error"""
        with pytest.raises(ValueError, match="val_size must be between 0 and 1"):
            split_data(self.df, test_size=0.2, val_size=1.0)
    
    def test_test_and_val_size_too_large(self):
        """Test that test_size + val_size >= 1 raises error"""
        with pytest.raises(ValueError, match="test_size \\+ val_size must be less than 1"):
            split_data(self.df, test_size=0.6, val_size=0.5)
    
    def test_valid_edge_case_sizes(self):
        """Test valid edge case sizes"""
        # This should work: 10/10/80
        train, test, val = split_data(self.df, test_size=0.1, val_size=0.1)
        
        assert len(train) == 80
        assert len(test) == 10
        assert len(val) == 10


class TestSplitDataNoTarget:
    """Tests for split_data without target column"""
    
    def setup_method(self):
        """Set up test data"""
        self.df = pd.DataFrame({
            'feature1': range(100),
            'feature2': range(100, 200),
            'feature3': range(200, 300)
        })
    
    def test_split_without_target(self):
        """Test that split works without target column"""
        train, test = split_data(self.df, test_size=0.2)
        
        assert len(train) == 80
        assert len(test) == 20
    
    def test_split_without_target_three_way(self):
        """Test three-way split without target column"""
        train, test, val = split_data(self.df, test_size=0.2, val_size=0.2)
        
        assert len(train) == 60
        assert len(test) == 20
        assert len(val) == 20
    
    def test_stratify_ignored_without_target(self):
        """Test that stratify is ignored when no target"""
        # Should not raise error
        train, test = split_data(self.df, stratify=True)
        
        assert len(train) + len(test) == len(self.df)


class TestSplitDataWithMushroomDataset:
    """Integration tests using real mushroom dataset"""
    
    def test_mushroom_two_way_split(self):
        """Test two-way split on mushroom data"""
        df = load_mushroom_data()
        train, test = split_data(df, target='class', test_size=0.2)
        
        # Check no data loss
        assert len(train) + len(test) == len(df)
        
        # Check proportions are approximately correct
        test_proportion = len(test) / len(df)
        assert abs(test_proportion - 0.2) < 0.01  # Within 1%
    
    def test_mushroom_three_way_split(self):
        """Test three-way split on mushroom data"""
        df = load_mushroom_data()
        train, test, val = split_data(df, target='class', test_size=0.2, val_size=0.2)
        
        # Check no data loss
        assert len(train) + len(test) + len(val) == len(df)
        
        # Check proportions are approximately correct
        test_proportion = len(test) / len(df)
        val_proportion = len(val) / len(df)
        
        assert abs(test_proportion - 0.2) < 0.01  # Within 1%
        assert abs(val_proportion - 0.2) < 0.01   # Within 1%
    
    def test_mushroom_stratification(self):
        """Test that mushroom data is properly stratified"""
        df = load_mushroom_data()
        train, test = split_data(df, target='class', test_size=0.2, stratify=True)
        
        # Check class proportions are similar
        orig_prop = df['class'].value_counts(normalize=True)
        train_prop = train['class'].value_counts(normalize=True)
        test_prop = test['class'].value_counts(normalize=True)
        
        # Within 2% of original proportions
        for class_label in orig_prop.index:
            assert abs(orig_prop[class_label] - train_prop[class_label]) < 0.02
            assert abs(orig_prop[class_label] - test_prop[class_label]) < 0.02
    
    def test_mushroom_all_columns_preserved(self):
        """Test that all columns are preserved in splits"""
        df = load_mushroom_data()
        train, test = split_data(df, target='class', test_size=0.2)
        
        assert list(train.columns) == list(df.columns)
        assert list(test.columns) == list(df.columns)


class TestSplitDataEdgeCases:
    """Tests for edge cases and special scenarios"""
    
    def test_small_dataset(self):
        """Test splitting very small dataset"""
        small_df = pd.DataFrame({
            'feature': [1, 2, 3, 4, 5],
            'class': ['A', 'B', 'A', 'B', 'A']
        })
        
        train, test = split_data(small_df, target='class', test_size=0.4)
        
        assert len(train) == 3
        assert len(test) == 2
    
    def test_binary_class_column(self):
        """Test with binary numeric class column"""
        df = pd.DataFrame({
            'feature': range(100),
            'class': [0, 1] * 50
        })
        
        train, test = split_data(df, target='class', stratify=True)
        
        # Should maintain 50/50 split
        train_counts = train['class'].value_counts()
        assert abs(train_counts[0] - train_counts[1]) <= 1
    
    def test_multiclass_target(self):
        """Test with multi-class target"""
        df = pd.DataFrame({
            'feature': range(150),
            'class': ['A'] * 50 + ['B'] * 50 + ['C'] * 50
        })
        
        train, test = split_data(df, target='class', stratify=True)
        
        # All classes should be present in both splits
        assert set(train['class'].unique()) == {'A', 'B', 'C'}
        assert set(test['class'].unique()) == {'A', 'B', 'C'}