"""
Simple decision tree interface for students
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


class SimpleDecisionTree:
    """
    Wrapper around scikit-learn DecisionTreeClassifier.
    Handles categorical encoding automatically.
    """
    
    def __init__(self, df, target_col, max_depth=None, criterion='entropy'):
        self.df = df
        self.target_col = target_col
        self.max_depth = max_depth
        self.criterion = criterion
        
        # Store original data info
        self.feature_names = [col for col in df.columns if col != target_col]
        self.class_names = sorted(df[target_col].unique())
        
        # Prepare encoders
        self.encoders = {}
        self.target_encoder = LabelEncoder()
        
        # Encode and train
        self._prepare_data()
        self._train()
    
    def _prepare_data(self):
        """Encode categorical variables"""
        X = self.df[self.feature_names]
        y = self.df[self.target_col]
        
        # Encode features
        self.X_encoded = pd.DataFrame()
        for col in self.feature_names:
            encoder = LabelEncoder()
            self.X_encoded[col] = encoder.fit_transform(X[col])
            self.encoders[col] = encoder
        
        # Encode target
        self.y_encoded = self.target_encoder.fit_transform(y)
    
    def _train(self):
        """Train the decision tree"""
        self.clf = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            random_state=42
        )
        self.clf.fit(self.X_encoded, self.y_encoded)
    
    def predict(self, df):
        """Make predictions on new data"""
        X_encoded = pd.DataFrame()
        for col in self.feature_names:
            X_encoded[col] = self.encoders[col].transform(df[col])
        
        predictions_encoded = self.clf.predict(X_encoded)
        return self.target_encoder.inverse_transform(predictions_encoded)
    
    def score(self, df):
        """Calculate accuracy on a dataset"""
        predictions = self.predict(df[self.feature_names])
        accuracy = (predictions == df[self.target_col]).sum() / len(df)
        return accuracy


def build_tree(df, target='class', max_depth=None, criterion='entropy'):
    """
    Build a decision tree for classification.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Training data with features and target
    target : str
        Name of the target column to predict
    max_depth : int, optional
        Maximum depth of the tree (None = unlimited)
    criterion : str
        Split criterion: 'entropy' (information gain) or 'gini'
    
    Returns:
    --------
    tree : SimpleDecisionTree
        Trained decision tree
    
    Examples:
    ---------
    >>> from itm4150.datasets import load_mushroom_data
    >>> from itm4150.trees import build_tree
    >>> df = load_mushroom_data()
    >>> tree = build_tree(df, target='class', max_depth=3)
    >>> accuracy = tree.score(df)
    >>> print(f"Accuracy: {accuracy:.2%}")
    """
    return SimpleDecisionTree(df, target, max_depth, criterion)


def visualize_tree(tree, df=None, figsize=(20, 10), fontsize=10):
    """
    Visualize a decision tree.
    
    Parameters:
    -----------
    tree : SimpleDecisionTree
        Trained decision tree
    df : pd.DataFrame, optional
        Original data (for context, not required)
    figsize : tuple
        Figure size (width, height)
    fontsize : int
        Font size for tree labels
    
    Examples:
    ---------
    >>> tree = build_tree(df, target='class', max_depth=3)
    >>> visualize_tree(tree)
    """
    plt.figure(figsize=figsize)
    plot_tree(
        tree.clf,
        feature_names=tree.feature_names,
        class_names=tree.class_names,
        filled=True,
        rounded=True,
        fontsize=fontsize
    )
    plt.tight_layout()
    plt.show()


def get_feature_importance(tree, df=None):
    """
    Get feature importance from a decision tree.
    
    Parameters:
    -----------
    tree : SimpleDecisionTree
        Trained decision tree
    df : pd.DataFrame, optional
        Original data (for context, not required)
    
    Returns:
    --------
    importance : pd.DataFrame
        Features ranked by importance
    
    Examples:
    ---------
    >>> tree = build_tree(df, target='class')
    >>> importance = get_feature_importance(tree)
    >>> print(importance.head())
    """
    importance = pd.DataFrame({
        'feature': tree.feature_names,
        'importance': tree.clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance


def get_tree_rules(tree):
    """
    Get text representation of decision tree rules.
    
    Parameters:
    -----------
    tree : SimpleDecisionTree
        Trained decision tree
    
    Returns:
    --------
    rules : str
        Text-based decision rules
    
    Examples:
    ---------
    >>> tree = build_tree(df, target='class', max_depth=2)
    >>> print(get_tree_rules(tree))
    """
    return export_text(tree.clf, feature_names=tree.feature_names)