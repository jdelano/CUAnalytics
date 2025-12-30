"""
Tests for dataset loaders
"""
import pytest
import pandas as pd
from itm4150.datasets import (
    load_mushroom_data,
    load_iris_data,
    get_sample_sales_data,
    list_datasets
)


class TestMushroomData:
    """Tests for mushroom dataset loader"""
    
    def test_loads_successfully(self):
        """Should load mushroom data without errors"""
        df = load_mushroom_data()
        assert isinstance(df, pd.DataFrame)
    
    def test_has_correct_shape(self):
        """Should have expected number of rows and columns"""
        df = load_mushroom_data()
        assert df.shape == (8124, 23)
    
    def test_has_class_column(self):
        """Should have 'class' column"""
        df = load_mushroom_data()
        assert 'class' in df.columns
    
    def test_class_values(self):
        """Class column should have 'e' and 'p' values"""
        df = load_mushroom_data()
        unique_classes = set(df['class'].unique())
        assert unique_classes == {'e', 'p'}


class TestIrisData:
    """Tests for iris dataset loader"""
    
    def test_loads_successfully(self):
        """Should load iris data without errors"""
        df = load_iris_data()
        assert isinstance(df, pd.DataFrame)
    
    def test_has_correct_shape(self):
        """Should have 150 rows and 5 columns"""
        df = load_iris_data()
        assert df.shape[0] == 150
        assert df.shape[1] == 5
    
    def test_has_species_column(self):
        """Should have 'species' column"""
        df = load_iris_data()
        assert 'species' in df.columns


class TestSampleSalesData:
    """Tests for sample sales data generator"""
    
    def test_generates_data(self):
        """Should generate sales data"""
        df = get_sample_sales_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1000
    
    def test_has_required_columns(self):
        """Should have expected columns"""
        df = get_sample_sales_data()
        required_cols = ['date', 'region', 'product', 'sales', 'quantity']
        for col in required_cols:
            assert col in df.columns
    
    def test_reproducible(self):
        """Should generate same data with same seed"""
        df1 = get_sample_sales_data()
        df2 = get_sample_sales_data()
        pd.testing.assert_frame_equal(df1, df2)


class TestListDatasets:
    """Tests for list_datasets function"""
    
    def test_returns_list(self):
        """Should return a list"""
        datasets = list_datasets()
        assert isinstance(datasets, list)
    
    def test_contains_mushroom(self):
        """Should include 'mushroom' dataset"""
        datasets = list_datasets()
        assert 'mushroom' in datasets