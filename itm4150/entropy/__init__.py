"""
Entropy and information theory tools.
"""

from .metrics import calculate_entropy, information_gain
from .visualization import plot_entropy

__all__ = [
    'calculate_entropy',
    'information_gain',
    'plot_entropy',
]