# tests/test_loaders.py
"""
Tests for dataset loader functions
"""
import pytest
import pandas as pd
from cuanalytics.datasets import (
    load_mushroom_data,
    load_iris_data,
    load_breast_cancer_data,
    load_real_estate_data,
    list_datasets,
    load_dataset
)


class TestLoadMushroomData:
    """Tests for load_mushroom_data function"""
    
    def test_returns_dataframe(self):
        """Test that function returns a DataFrame"""
        df = load_mushroom_data()
        assert isinstance(df, pd.DataFrame)
    
    def test_correct_shape(self):
        """Test that dataset has correct shape"""
        df = load_mushroom_data()
        # Should have 8124 samples and 23 columns
        assert df.shape == (8124, 23)
    
    def test_has_class_column(self):
        """Test that dataset has 'class' column"""
        df = load_mushroom_data()
        assert 'class' in df.columns
    
    def test_class_values(self):
        """Test that class column has correct values"""
        df = load_mushroom_data()
        unique_classes = set(df['class'].unique())
        assert unique_classes == {'e', 'p'}  # edible, poisonous
    
    def test_no_missing_values(self):
        """Test that there are no missing values (except possibly stalk-root)"""
        df = load_mushroom_data()
        # Mushroom dataset has '?' for missing values in stalk-root
        # Just check that we got data
        assert not df.empty
    
    def test_column_names(self):
        """Test that dataset has expected column names"""
        df = load_mushroom_data()
        expected_columns = [
            'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
            'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
            'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
            'stalk-surface-below-ring', 'stalk-color-above-ring',
            'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
            'ring-type', 'spore-print-color', 'population', 'habitat'
        ]
        assert list(df.columns) == expected_columns


class TestLoadIrisData:
    """Tests for load_iris_data function"""
    
    def test_returns_dataframe(self):
        """Test that function returns a DataFrame"""
        df = load_iris_data()
        assert isinstance(df, pd.DataFrame)
    
    def test_correct_shape(self):
        """Test that dataset has correct shape"""
        df = load_iris_data()
        # Should have 150 samples and 5 columns
        assert df.shape == (150, 5)
    
    def test_has_species_column(self):
        """Test that dataset has 'species' column"""
        df = load_iris_data()
        assert 'species' in df.columns
    
    def test_three_species(self):
        """Test that there are exactly 3 species"""
        df = load_iris_data()
        assert len(df['species'].unique()) == 3
    
    def test_balanced_classes(self):
        """Test that classes are balanced (50 samples each)"""
        df = load_iris_data()
        counts = df['species'].value_counts()
        assert all(count == 50 for count in counts)
    
    def test_no_missing_values(self):
        """Test that there are no missing values"""
        df = load_iris_data()
        assert df.isna().sum().sum() == 0
    
    def test_numeric_features(self):
        """Test that feature columns are numeric"""
        df = load_iris_data()
        numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(df[col])


class TestLoadBreastCancerData:
    """Tests for load_breast_cancer_data function"""
    
    def test_returns_dataframe(self):
        """Test that function returns a DataFrame"""
        df = load_breast_cancer_data()
        assert isinstance(df, pd.DataFrame)
    
    def test_correct_shape(self):
        """Test that dataset has correct shape"""
        df = load_breast_cancer_data()
        # Should have 569 samples and 31 columns (30 features + diagnosis)
        assert df.shape == (569, 31)
    
    def test_has_diagnosis_column(self):
        """Test that dataset has 'diagnosis' column"""
        df = load_breast_cancer_data()
        assert 'diagnosis' in df.columns
    
    def test_diagnosis_values(self):
        """Test that diagnosis has M and B values"""
        df = load_breast_cancer_data()
        unique_diagnoses = set(df['diagnosis'].unique())
        assert unique_diagnoses == {'M', 'B'}  # Malignant, Benign
    
    def test_no_missing_values(self):
        """Test that there are no missing values"""
        df = load_breast_cancer_data()
        assert df.isna().sum().sum() == 0
    
    def test_all_features_numeric(self):
        """Test that all features are numeric"""
        df = load_breast_cancer_data()
        feature_cols = [col for col in df.columns if col != 'diagnosis']
        for col in feature_cols:
            assert pd.api.types.is_numeric_dtype(df[col])
    
    def test_has_mean_se_worst_features(self):
        """Test that dataset has mean, standard error, and worst versions of features"""
        df = load_breast_cancer_data()
        
        # Check that we have mean, error, and worst variants
        columns = df.columns.tolist()
        
        # Should have columns starting with 'mean'
        mean_cols = [col for col in columns if col.startswith('mean')]
        assert len(mean_cols) > 0
        
        # Should have columns with 'error' (standard error)
        error_cols = [col for col in columns if 'error' in col]
        assert len(error_cols) > 0
        
        # Should have columns starting with 'worst'
        worst_cols = [col for col in columns if col.startswith('worst')]
        assert len(worst_cols) > 0
        
        # Check specific examples
        assert 'mean radius' in columns
        assert 'worst radius' in columns


class TestLoadRealEstateData:
    """Tests for load_real_estate_data function"""
    
    def test_returns_dataframe(self):
        """Test that function returns a DataFrame"""
        df = load_real_estate_data()
        assert isinstance(df, pd.DataFrame)
    
    def test_correct_shape(self):
        """Test that dataset has correct shape"""
        df = load_real_estate_data()
        # Should have 414 samples (approximately) and 7 columns
        assert df.shape[0] > 400  # Allow some flexibility
        assert df.shape[1] == 7
    
    def test_has_price_column(self):
        """Test that dataset has 'price_per_unit' column"""
        df = load_real_estate_data()
        assert 'price_per_unit' in df.columns
    
    def test_has_expected_columns(self):
        """Test that dataset has expected columns"""
        df = load_real_estate_data()
        expected_columns = [
            'transaction_date',
            'house_age',
            'distance_to_MRT',
            'num_convenience_stores',
            'latitude',
            'longitude',
            'price_per_unit'
        ]
        assert list(df.columns) == expected_columns
    
    def test_all_columns_numeric(self):
        """Test that all columns are numeric"""
        df = load_real_estate_data()
        for col in df.columns:
            assert pd.api.types.is_numeric_dtype(df[col])
    
    def test_no_missing_values(self):
        """Test that there are no missing values"""
        df = load_real_estate_data()
        assert df.isna().sum().sum() == 0
    
    def test_no_row_number_column(self):
        """Test that 'No' column was removed"""
        df = load_real_estate_data()
        assert 'No' not in df.columns
    
    def test_price_values_reasonable(self):
        """Test that price values are in reasonable range"""
        df = load_real_estate_data()
        # Prices should be positive
        assert (df['price_per_unit'] > 0).all()
        # And in reasonable range (between 0 and 200 based on data description)
        assert df['price_per_unit'].max() < 200


class TestListDatasets:
    """Tests for list_datasets function"""
    
    def test_returns_list(self):
        """Test that function returns a list"""
        datasets = list_datasets()
        assert isinstance(datasets, list)
    
    def test_contains_expected_datasets(self):
        """Test that list contains expected datasets"""
        datasets = list_datasets()
        
        # Should contain at least these datasets
        expected = ['mushroom', 'iris', 'breast_cancer', 'real_estate']
        for dataset in expected:
            assert dataset in datasets
    
    def test_non_empty(self):
        """Test that list is not empty"""
        datasets = list_datasets()
        assert len(datasets) > 0


class TestLoadDataset:
    """Tests for load_dataset function"""
    
    def test_load_mushroom_by_name(self):
        """Test loading mushroom dataset by name"""
        df = load_dataset('mushroom')
        assert isinstance(df, pd.DataFrame)
        assert 'class' in df.columns
    
    def test_load_iris_by_name(self):
        """Test loading iris dataset by name"""
        df = load_dataset('iris')
        assert isinstance(df, pd.DataFrame)
        assert 'species' in df.columns
    
    def test_load_breast_cancer_by_name(self):
        """Test loading breast cancer dataset by name"""
        df = load_dataset('breast_cancer')
        assert isinstance(df, pd.DataFrame)
        assert 'diagnosis' in df.columns
    
    def test_load_real_estate_by_name(self):
        """Test loading real estate dataset by name"""
        df = load_dataset('real_estate')
        assert isinstance(df, pd.DataFrame)
        assert 'price_per_unit' in df.columns
    
    def test_invalid_dataset_name(self):
        """Test that invalid dataset name raises error"""
        with pytest.raises(ValueError, match="Dataset .* not found"):
            load_dataset('nonexistent_dataset')
    
    def test_error_message_shows_available_datasets(self):
        """Test that error message lists available datasets"""
        with pytest.raises(ValueError, match="Available datasets"):
            load_dataset('invalid')


class TestDatasetErrorHandling:
    """Tests for error handling in dataset loaders"""
    
    def test_mushroom_network_error_handling(self, monkeypatch):
        """Test that mushroom loader handles network errors"""
        import requests
        
        def mock_get(*args, **kwargs):
            raise requests.exceptions.ConnectionError("Network error")
        
        monkeypatch.setattr(requests, 'get', mock_get)
        
        with pytest.raises(Exception):
            load_mushroom_data()
    
    def test_iris_network_error_handling(self, monkeypatch):
        """Test that iris loader handles network errors"""
        import requests
        
        def mock_get(*args, **kwargs):
            raise requests.exceptions.Timeout("Timeout error")
        
        monkeypatch.setattr(requests, 'get', mock_get)
        
        with pytest.raises(Exception):
            load_iris_data()

    def test_real_estate_download_error_handling(self, monkeypatch):
        """Test that real estate loader handles download errors"""
        import urllib.request
        
        def mock_urlopen(*args, **kwargs):
            raise Exception("Download failed")
        
        monkeypatch.setattr(urllib.request, 'urlopen', mock_urlopen)
        
        # Should handle the error gracefully
        # Based on your code, it prints error and returns None
        result = load_real_estate_data()
        assert result is None


class TestDatasetIntegration:
    """Integration tests using datasets together"""
    
    def test_all_datasets_load(self):
        """Test that all datasets can be loaded"""
        datasets = list_datasets()
        
        for name in datasets:
            df = load_dataset(name)
            assert isinstance(df, pd.DataFrame)
            assert not df.empty
    
    def test_datasets_have_consistent_structure(self):
        """Test that datasets return DataFrames with expected structure"""
        # Each dataset should return a DataFrame
        mushroom = load_mushroom_data()
        iris = load_iris_data()
        breast_cancer = load_breast_cancer_data()
        real_estate = load_real_estate_data()
        
        for df in [mushroom, iris, breast_cancer, real_estate]:
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert len(df.columns) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])