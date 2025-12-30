"""
Tests for visualization module
"""
import pytest
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from itm4150.entropy.visualization import plot_entropy_rectangles
from itm4150.datasets import load_mushroom_data


class TestPlotEntropyRectangles:
    """Tests for plot_entropy_rectangles function"""
    
    def test_creates_plot(self):
        """Should create a plot without errors"""
        df = load_mushroom_data()
        
        # This should not raise an exception
        plot_entropy_rectangles(df, 'odor')
        
        # Clean up
        plt.close('all')
    
    def test_with_different_features(self):
        """Should work with different feature columns"""
        df = load_mushroom_data()
        
        for feature in ['odor', 'gill-color', 'cap-color']:
            plot_entropy_rectangles(df, feature)
            plt.close('all')
    
    def test_invalid_feature_raises_error(self):
        """Should raise error for non-existent feature"""
        df = load_mushroom_data()
        
        with pytest.raises(KeyError):
            plot_entropy_rectangles(df, 'nonexistent_column')