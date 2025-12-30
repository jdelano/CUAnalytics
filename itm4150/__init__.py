"""
ITM 4150: Advanced Business Analytics and Visualization
Python toolkit for course materials at Cedarville University.
"""

__version__ = "0.1.0"
__author__ = "John Delano"

# Import commonly used functions for convenient access
from itm4150.entropy.metrics import calculate_entropy, information_gain
from itm4150.entropy.visualization import plot_entropy
from itm4150.datasets.loaders import load_mushroom_data

# Define what gets imported with "from itm4150 import *"
__all__ = [
    'calculate_entropy',
    'information_gain',
    'plot_entropy',
    'load_mushroom_data',
]