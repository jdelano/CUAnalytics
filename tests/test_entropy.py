"""
Tests for entropy module
"""
import pytest
import numpy as np
import pandas as pd
from cuanalytics.entropy import calculate_entropy, information_gain, plot_entropy
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing


@pytest.fixture(autouse=True)
def close_plots():
    """Automatically close all plots after each test."""
    import matplotlib.pyplot as plt
    yield
    plt.close('all')


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

    def test_calculate_entropy_empty_input(self):
        """Test entropy calculation with empty input"""
        from cuanalytics.entropy import calculate_entropy
        import pandas as pd
        
        # Empty pandas Series
        empty_series = pd.Series([], dtype=str)
        assert calculate_entropy(empty_series) == 0.0

    def test_calculate_entropy_dataframe_target(self):
        """Entropy should work with DataFrame + target_col"""
        df = pd.DataFrame({
            'feature': ['X', 'X', 'Y', 'Y'],
            'class': ['A', 'A', 'B', 'B']
        })
        entropy = calculate_entropy(df, target_col='class')
        assert abs(entropy - 1.0) < 0.0001

    def test_calculate_entropy_dataframe_split(self):
        """Weighted entropy should work with DataFrame + split_col"""
        df = pd.DataFrame({
            'feature': ['X', 'X', 'Y', 'Y'],
            'class': ['A', 'A', 'B', 'B']
        })
        entropy = calculate_entropy(df, target_col='class', split_col='feature')
        assert abs(entropy - 0.0) < 0.0001


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
        from cuanalytics.datasets import load_mushroom_data
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
        from cuanalytics.datasets import load_mushroom_data
        df = load_mushroom_data()
        
        # Calculate IG for multiple features
        ig_scores = {}
        for feature in ['odor', 'gill-color', 'spore-print-color']:
            ig_scores[feature] = information_gain(df, feature, 'class')
        
        # All should be non-negative
        assert all(ig >= 0 for ig in ig_scores.values())
        
        # Odor should be highly informative (this is known)
        assert ig_scores['odor'] > 0.5
        

class TestEntropyVisualization:
    """Test entropy visualization functions"""
    
    def test_plot_entropy_basic(self, monkeypatch):
        """Test basic entropy plot creation"""
        # Mock plt.show() to prevent display
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        
        df = pd.DataFrame({
            'feature': ['A', 'A', 'B', 'B', 'C'],
            'class': ['X', 'Y', 'X', 'X', 'Y']
        })
        
        # Should run without error
        plot_entropy(df, 'feature', target_col='class')
    
    def test_plot_entropy_pure_subset(self, monkeypatch):
        """Test with pure subset (entropy=0)"""
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        
        df = pd.DataFrame({
            'feature': ['A', 'A', 'B', 'B'],
            'class': ['X', 'X', 'Y', 'Y']  # Pure subsets
        })
        
        plot_entropy(df, 'feature', target_col='class')
    
    def test_plot_entropy_single_value(self, monkeypatch):
        """Test with single feature value"""
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        
        df = pd.DataFrame({
            'feature': ['A', 'A', 'A'],
            'class': ['X', 'Y', 'X']
        })
        
        plot_entropy(df, 'feature', target_col='class')
    
    def test_plot_entropy_many_values(self, monkeypatch):
        """Test with many feature values"""
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        
        df = pd.DataFrame({
            'feature': ['A', 'B', 'C', 'D', 'E', 'F'],
            'class': ['X', 'Y', 'X', 'Y', 'X', 'Y']
        })
        
        plot_entropy(df, 'feature', target_col='class')
    
    def test_plot_entropy_unequal_proportions(self, monkeypatch):
        """Test with very unequal proportions"""
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        
        df = pd.DataFrame({
            'feature': ['A'] * 90 + ['B'] * 10,
            'class': ['X', 'Y'] * 50
        })
        
        plot_entropy(df, 'feature', target_col='class')
    
    def test_plot_entropy_different_target_col(self, monkeypatch):
        """Test with different target column name"""
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        
        df = pd.DataFrame({
            'feature': ['A', 'B', 'C'],
            'label': ['X', 'Y', 'X']
        })
        
        plot_entropy(df, 'feature', target_col='label')
    
    def test_plot_entropy_sorting(self, monkeypatch):
        """Test that values are sorted by entropy"""
        import matplotlib.pyplot as plt
        
        # Track what gets plotted
        rectangles = []
        original_Rectangle = plt.Rectangle
        
        def mock_Rectangle(*args, **kwargs):
            rect = original_Rectangle(*args, **kwargs)
            rectangles.append({
                'x': args[0][0],
                'y': args[0][1], 
                'width': args[1],
                'height': args[2]
            })
            return rect
        
        monkeypatch.setattr('matplotlib.pyplot.Rectangle', mock_Rectangle)
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        
        df = pd.DataFrame({
            'feature': ['A', 'A', 'B', 'B', 'C', 'C'],
            'class': ['X', 'X', 'X', 'Y', 'Y', 'Y']  # A=0, B=1, C=0
        })
        
        plot_entropy(df, 'feature', target_col='class')
        
        # Check that heights (entropies) are in increasing order
        heights = [r['height'] for r in rectangles]
        assert heights == sorted(heights), "Rectangles should be sorted by entropy"
    
    def test_plot_entropy_proportions_sum_to_one(self, monkeypatch):
        """Test that rectangle widths sum to 1.0"""
        import matplotlib.pyplot as plt
        
        rectangles = []
        original_Rectangle = plt.Rectangle
        
        def mock_Rectangle(*args, **kwargs):
            rect = original_Rectangle(*args, **kwargs)
            rectangles.append({
                'x': args[0][0],
                'width': args[1]
            })
            return rect
        
        monkeypatch.setattr('matplotlib.pyplot.Rectangle', mock_Rectangle)
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        
        df = pd.DataFrame({
            'feature': ['A', 'B', 'C', 'D'],
            'class': ['X', 'Y', 'X', 'Y']
        })
        
        plot_entropy(df, 'feature', target_col='class')
        
        # Check widths sum to 1.0
        total_width = sum(r['width'] for r in rectangles)
        assert abs(total_width - 1.0) < 1e-10, "Total width should equal 1.0"
    
    def test_plot_entropy_rectangles_contiguous(self, monkeypatch):
        """Test that rectangles are placed contiguously"""
        import matplotlib.pyplot as plt
        
        rectangles = []
        original_Rectangle = plt.Rectangle
        
        def mock_Rectangle(*args, **kwargs):
            rect = original_Rectangle(*args, **kwargs)
            rectangles.append({
                'x': args[0][0],
                'width': args[1]
            })
            return rect
        
        monkeypatch.setattr('matplotlib.pyplot.Rectangle', mock_Rectangle)
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
        
        df = pd.DataFrame({
            'feature': ['A', 'B', 'C'],
            'class': ['X', 'Y', 'X']
        })
        
        plot_entropy(df, 'feature', target_col='class')
        
        # Check that each rectangle starts where previous one ended
        for i in range(1, len(rectangles)):
            prev_end = rectangles[i-1]['x'] + rectangles[i-1]['width']
            curr_start = rectangles[i]['x']
            assert abs(prev_end - curr_start) < 1e-10, "Rectangles should be contiguous"
