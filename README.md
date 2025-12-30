# ITM 4150: Advanced Business Analytics and Visualization Toolkit

Python companion package for **ITM 4150: Advanced Business Analytics and Visualization** at Cedarville University.

This package provides tools for:
- Information theory and entropy analysis
- Machine learning fundamentals
- Data visualization
- Statistical analysis
- Business analytics workflows

## Installation
```bash
pip install itm4150
```

## Modules

### Entropy

Information theory concepts including entropy and information gain calculations.
```python
from itm4150.entropy import calculate_entropy, information_gain
from itm4150.entropy.visualization import plot_entropy_rectangles
from itm4150.datasets import load_mushroom_data

# Load sample data
df = load_mushroom_data()

# Calculate entropy
entropy = calculate_entropy(df['class'])
print(f"Dataset entropy: {entropy:.4f}")

# Calculate information gain for a feature
children = [df[df['odor'] == val]['class'] for val in df['odor'].unique()]
ig = information_gain(df['class'], *children)
print(f"Information gain: {ig:.4f}")

# Visualize entropy distribution
plot_entropy(df, 'odor')
```

## For Students

This package provides helper functions for concepts covered in ITM 4150. Each module corresponds to a major topic area in the course.

## License

MIT License - Free for educational use