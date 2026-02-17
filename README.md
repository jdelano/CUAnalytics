# CUAnalytics: Business Analytics Toolkit for Cedarville University

A Python package designed for Cedarville University students studying business analytics and data science. Provides intuitive, educational implementations of machine learning algorithms, statistical analysis tools, and data visualization capabilities.

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Purpose

CUAnalytics focuses on **understanding over complexity** - providing student-friendly interfaces to essential analytics techniques with clear, interpretable output that matches what you'd see in statistical software like R, SPSS, or Stata.

## 📦 Installation

```bash
pip install cuanalytics
```

For development:
```bash
pip install cuanalytics[dev]
```

## 🚀 Quick Start

```python
import cuanalytics as ca

# Load sample data
df = ca.load_real_estate_data()

# Split into train/test
train, test = ca.split_data(df, test_size=0.2)

# Fit a linear regression model
model = ca.fit_lm(train, formula='price_per_unit ~ .')

# View comprehensive statistical output
model.summary()

# Visualize results
model.visualize()
model.visualize_all_features()

# Evaluate on test set
test_r2 = model.score(test)['r2']
print(f"Test R²: {test_r2:.4f}")
```

## 📚 Modules

### 🌳 Decision Trees

Build and visualize decision trees for classification tasks.

```python
import cuanalytics as ca

# Load data
df = ca.load_mushroom_data()
train, test = ca.split_data(df, test_size=0.2)

# Build decision tree
tree = ca.fit_tree(train, formula='class ~ .', max_depth=3, criterion='entropy')

# Visualize tree structure
tree.visualize()

# Visualize decision regions
tree.visualize_features('odor', 'spore-print-color')

# Get feature importance
importance = tree.get_feature_importance()

# View decision rules
print(tree.get_rules())

# Evaluate
train_acc = tree.score(train)['accuracy']
test_acc = tree.score(test)['accuracy']
```

### 📊 Linear Discriminant Analysis (LDA)

Perform classification with dimensionality reduction.

```python
import cuanalytics as ca

# Load data
df = ca.load_iris_data()
train, test = ca.split_data(df, test_size=0.2)

# Fit LDA model
lda = ca.fit_lda(train, formula='species ~ .')

# Comprehensive summary
lda.summary()

# Visualize in discriminant space
lda.visualize()

# Visualize decision boundaries for specific features
lda.visualize_features('petal_length', 'petal_width')

# Get discriminant scores
scores = lda.transform(test)

# Predictions
predictions = lda.predict(test)
test_accuracy = lda.score(test)['accuracy']
```

### 🎯 Support Vector Machines (SVM)

Linear SVM for binary classification with margin visualization.

```python
import cuanalytics as ca

# Load data
df = ca.load_breast_cancer_data()
train, test = ca.split_data(df, test_size=0.2)

# Fit SVM (C parameter controls margin strictness)
svm = ca.fit_svm(train, formula='diagnosis ~ .', C=1.0)

# View model details including support vectors
svm.summary()

# Visualize support vectors and margin
svm.visualize()

# Visualize decision boundary
svm.visualize_features('radius_mean', 'texture_mean')

# Get support vectors
support_vectors = svm.get_support_vectors()

# Evaluate
test_accuracy = svm.score(test)['accuracy']
```

### 📈 Linear Regression

Comprehensive linear regression with formula support for interactions and transformations.

```python
import cuanalytics as ca

# Load data
df = ca.load_real_estate_data()
train, test = ca.split_data(df, test_size=0.2)

# Method 1: Use all features
model = ca.fit_lm(train, formula='price_per_unit ~ .')

# Method 2: Select specific features
model = ca.fit_lm(train, formula='price_per_unit ~ house_age + distance_to_MRT')

# Method 3: Use R-style formulas for interactions
model = ca.fit_lm(train, 
               formula='price_per_unit ~ house_age * num_convenience_stores')

# Statistical summary (like R/SPSS output)
summary = model.summary()
# Shows: coefficients, t-statistics, p-values, ANOVA table, R², F-statistic

# Visualizations
model.visualize()  # Predicted vs actual, residuals, coefficients
model.visualize_feature('house_age')  # Single feature relationship
model.visualize_all_features()  # Grid of all features

# Get metrics
metrics = model.get_metrics()
# Returns: {'metrics': {'r2': ..., 'rmse': ..., 'mae': ...}, ...}

# Predictions
predictions = model.predict(test)
```

### 🧭 Logistic Regression

Logistic regression for binary and multiclass classification.

```python
import cuanalytics as ca

# Load data
df = ca.load_breast_cancer_data()
train, test = ca.split_data(df, test_size=0.2)

# Fit logistic regression
logit = ca.fit_logit(train, formula='diagnosis ~ .', C=1.0, penalty='l2', solver='lbfgs')

# Summary and visualization
logit.summary()
logit.visualize()
logit.visualize_features('radius_mean', 'texture_mean')

# Evaluate
test_report = logit.score(test)
print(f"Accuracy: {test_report['accuracy']:.2%}")
```

Penalty and solver notes:
- `penalty`: regularization type. `l2` shrinks coefficients smoothly; `l1` can drop features; `elasticnet` mixes both.
- `solver`: optimization algorithm. `lbfgs` is a solid default; `liblinear` works well for small/binary data; `saga` supports `l1`/`elasticnet` and large datasets.

### 🧠 Neural Networks

Feedforward neural networks for classification or regression using scikit-learn MLP.

```python
import cuanalytics as ca

df = ca.load_breast_cancer_data()
train, test = ca.split_data(df, test_size=0.2, random_state=42)
train, scaler = ca.scale_data(train, exclude_cols=['diagnosis'])
test, _ = ca.scale_data(test, exclude_cols=['diagnosis'], scaler=scaler)

nn = ca.fit_nn(
    train,
    formula='diagnosis ~ .',
    hidden_layers=[3, 5, 2],
    max_iter=10000
)

nn.summary()
nn.visualize()

report = nn.score(test)
print(f"Accuracy: {report['accuracy']:.2%}")
```

#### Formula Syntax

```python
# Main effects only
ca.fit_lm(df, formula='y ~ x1 + x2')

# Interaction effects (includes main effects + interaction)
ca.fit_lm(df, formula='y ~ x1 * x2')
# Equivalent to: y ~ x1 + x2 + x1:x2

# Interaction only
ca.fit_lm(df, formula='y ~ x1:x2')

# All features
ca.fit_lm(df, formula='y ~ .')

# All except some
ca.fit_lm(df, formula='y ~ . - unwanted_feature')

# Polynomial terms
ca.fit_lm(df, formula='y ~ x + I(x**2)')

# Transformations
ca.fit_lm(df, formula='y ~ np.log(x)')
```

### 📉 Information Theory & Entropy

Calculate entropy and information gain for decision trees and data analysis.

```python
import cuanalytics as ca

# Calculate entropy of a variable
entropy = ca.calculate_entropy(df['class'])
print(f"Entropy: {entropy:.4f}")

# Calculate entropy from a DataFrame column
entropy = ca.calculate_entropy(df, target_col='class')
print(f"Entropy: {entropy:.4f}")

# Calculate information gain from a split
ig = ca.information_gain(df, feature='feature', target_col='class')
print(f"Information gain: {ig:.4f}")

# Visualize entropy with rectangles
ca.plot_entropy_rectangles(df, feature='odor', target='class')
```

### 📐 Similarity & Distance

```python
import cuanalytics as ca

ca.euclidean([1, 2], [4, 6])
ca.manhattan([1, 2], [4, 6])
ca.cosine([1, 0], [0, 1])
ca.jaccard([1, 0, 1], [1, 1, 0])
```

### 🤝 k-Nearest Neighbors (KNN)

Classification:

```python
import cuanalytics as ca

df = ca.load_breast_cancer_data()
train, test = ca.split_data(df, test_size=0.2, random_state=42)

knn = ca.fit_knn_classifier(train, formula='diagnosis ~ .', k=5)
knn.summary()

report = knn.score(test)
print(f"Accuracy: {report['accuracy']:.2%}")
```

Regression:

```python
import cuanalytics as ca

df = ca.load_real_estate_data()
train, test = ca.split_data(df, test_size=0.2, random_state=42)

knn = ca.fit_knn_regressor(train, formula='price_per_unit ~ .', k=5)
metrics = knn.score(test)
print(f"Test R²: {metrics['r2']:.4f}")
```

### 🧩 Clustering

K-Means:

```python
import cuanalytics as ca

df = ca.load_iris_data()
kmeans = ca.fit_kmeans(df, formula='~ sepal_length + sepal_width + petal_length + petal_width', n_clusters=3)
kmeans.summary()
kmeans.visualize()
metrics = kmeans.get_metrics()
print(metrics['silhouette'])

# Optional: one-vs-rest rule descriptions for each cluster
cluster_descriptions = kmeans.describe_clusters(max_depth=3)
cluster_descriptions[['cluster', 'cluster_rule']].drop_duplicates().sort_values('cluster')
```

Hierarchical:

```python
import cuanalytics as ca

df = ca.load_iris_data()
hier = ca.fit_hierarchical(df, formula='~ sepal_length + sepal_width + petal_length + petal_width', n_clusters=3)
hier.summary()
hier.visualize()  # Full dendrogram
hier.visualize(cutoff=10, truncate_mode='lastp')  # Last 10 groupings
hier.visualize(cutoff=2, truncate_mode='level')  # Top 2 hierarchy levels
hier.visualize_all_features()  # PCA projection of all features
```

### 📊 Dataset Loaders

Built-in datasets for practice and examples.

```python
import cuanalytics as ca

# Available dataset loaders
ca.load_iris_data            # Classification (3 classes, 4 features)
ca.load_mushroom_data        # Classification (binary, categorical features)
ca.load_breast_cancer_data   # Classification (binary, 30 features)
ca.load_real_estate_data     # Regression (real-world housing data)

# All loaders return pandas DataFrames
df = ca.load_iris_data()
print(df.head())
print(df.shape)
```

### 🛠️ Utilities

```python
import cuanalytics as ca

# Train/test split with optional random seed
train, test = ca.split_data(df, test_size=0.2, random_state=42)

# Stratified split (useful for categorical targets)
train, test = ca.split_data(df, test_size=0.3, stratify_on='class')

# Train/validation/test split
train, val, test = ca.split_data(df, test_size=0.2, val_size=0.1, random_state=42)

# Scale numeric features (fit on train, apply to test)
# By default, binary (0/1) columns are left unchanged.
train_scaled, scaler = ca.scale_data(train, exclude_cols=['class'])
test_scaled, _ = ca.scale_data(test, exclude_cols=['class'], scaler=scaler)

# Scale binary columns too (if desired)
train_scaled, scaler = ca.scale_data(train, exclude_cols=['class'], skip_binary=False)
```

## 🧪 Model Selection

Cross-validation for supervised models:

```python
import cuanalytics as ca

# Classification
cv_results = ca.cross_validate(
    ca.fit_logit,
    df,
    formula='class ~ .',
    k=5,
    stratify_on='class',
)
print(cv_results['summary']['mean'])

# Regression
cv_results = ca.cross_validate(
    ca.fit_lm,
    df,
    formula='price_per_unit ~ .',
    k=5,
)
print(cv_results['summary']['mean'])
```

Grid search (example with logistic regression):

```python
import cuanalytics as ca

df = ca.load_breast_cancer_data()
train, test = ca.split_data(df, test_size=0.2, random_state=42)

param_grid = {
    "C": [0.1, 1.0, 10.0],
}

results = ca.grid_search_cv(
    ca.fit_logit,
    train,
    formula='diagnosis ~ .',
    param_grid=param_grid,
    k=5,
    stratify_on='diagnosis',
    refit='accuracy',
)

best_model = results['best_model']
test_report = best_model.score(test)
print(f"Test Accuracy: {test_report['accuracy']:.2%}")
```

Notes:
- `ca.cross_validate` uses each model's `predict` output and computes metrics without printing.
- You can call `model.get_score(df)` for metrics without printing, or `model.score(df)` to print a report.
- Example notebook: `examples/14_grid_search_models.ipynb` (logistic regression, SVM, and neural net grids).

Learning curves (validation performance vs. training size):

```python
import cuanalytics as ca

ca.plot_learning_curves(
    [ca.fit_logit, ca.fit_svm, ca.fit_knn_classifier],
    df,
    formula='class ~ .',
    train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0],
    k=5,
    stratify_on='class',
    metric='accuracy',
    verbose=False,
)
```

## 🔄 API Pattern

Most models follow this pattern:

```python
# Fit model
model = fit_*(train_data, formula='target_column ~ .')

# Or with options
model = fit_*(train_data, formula='target_column ~ .', param1=value1, param2=value2)

# Make predictions
predictions = model.predict(test_data)

# Evaluate performance
score_report = model.score(test_data)
some_metric = list(score_report.values())[0]

# View detailed summary
model.summary()

# Visualize (availability varies by model)
model.visualize()
```

Common optional methods on many models:

```python
model.get_metrics()
model.visualize_features('feature1', 'feature2')   # Some supervised models
model.visualize_all_features()                     # Some regression/clustering models
```

## 📖 Documentation

For detailed documentation on each module:

```python
# Get help on any function
help(ca.fit_lm)
help(ca.fit_tree)

# View docstrings
import cuanalytics as ca
print(ca.fit_lda.__doc__)
```

## 🤝 Contributing

This package is developed for educational purposes. Suggestions and improvements welcome!

## 📝 License

MIT License - Free for educational and commercial use.

## 🎓 Educational Focus

This package is designed for **learning**, not production use. Key features:

- **Clear Output**: Statistical summaries match formats from R, SPSS, Stata
- **Visualizations**: Built-in plotting for every algorithm
- **Interpretability**: Methods to explain model decisions
- **Consistency**: Uniform API across all models (`fit_*`, `predict`, `score`, `summary`, `visualize`)
- **Ease of Use**: Simple, readable code that students can understand

## 👨‍🏫 Author

**Dr. John D. Delano**  
Professor of IT Management, Cedarville University  
jdelano@cedarville.edu

## 🔗 Links

- [GitHub Repository](https://github.com/jdelano/CUAnalytics)
- [PyPI Package](https://pypi.org/project/cuanalytics/)
- [Report Issues](https://github.com/jdelano/cuanalytics/issues)
