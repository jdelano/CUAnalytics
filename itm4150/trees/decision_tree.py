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
    Handles both numeric and categorical features automatically.
    """
    
    def __init__(self, df, target_col, max_depth=None, criterion='entropy'):
        self.df = df
        self.target_col = target_col
        self.max_depth = max_depth
        self.criterion = criterion
        
        # Store original data info
        self.feature_names_original = [col for col in df.columns if col != target_col]
        self.class_names = sorted(df[target_col].unique())
        
        # Identify numeric vs categorical features
        self.numeric_features = []
        self.categorical_features = []
        
        for col in self.feature_names_original:
            if pd.api.types.is_numeric_dtype(df[col]):
                self.numeric_features.append(col)
            else:
                self.categorical_features.append(col)
        
        print(f"Numeric features: {self.numeric_features}")
        print(f"Categorical features: {self.categorical_features}")
        
        # Prepare encoders
        self.target_encoder = LabelEncoder()
        
        # Encode and train
        self._prepare_data()
        self._train()
    
    def _prepare_data(self):
        """Encode categorical variables, keep numeric as-is"""
        X = self.df[self.feature_names_original]
        y = self.df[self.target_col]
        
        # Start with numeric features (no encoding needed)
        if self.numeric_features:
            self.X_encoded = X[self.numeric_features].copy()
        else:
            self.X_encoded = pd.DataFrame()
        
        # One-hot encode categorical features
        if self.categorical_features:
            X_categorical = X[self.categorical_features]
            X_onehot = pd.get_dummies(X_categorical, prefix=self.categorical_features)
            self.X_encoded = pd.concat([self.X_encoded, X_onehot], axis=1)
        
        self.feature_names = list(self.X_encoded.columns)
        
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
        # Process numeric features
        if self.numeric_features:
            X_encoded = df[self.numeric_features].copy()
        else:
            X_encoded = pd.DataFrame()
        
        # One-hot encode categorical features
        if self.categorical_features:
            X_categorical = df[self.categorical_features]
            X_onehot = pd.get_dummies(X_categorical, prefix=self.categorical_features)
            
            # Ensure all columns from training are present
            missing_cols = [col for col in self.feature_names 
                            if col not in X_onehot.columns and col not in self.numeric_features]

            if missing_cols:
                # Add all missing columns at once
                missing_df = pd.DataFrame(0, index=X_onehot.index, columns=missing_cols)
                X_onehot = pd.concat([X_onehot, missing_df], axis=1)
            
            # Reorder columns to match training
            if self.numeric_features:
                X_encoded = pd.concat([X_encoded, X_onehot[self.feature_names[len(self.numeric_features):]]], axis=1)
            else:
                X_encoded = X_onehot[self.feature_names]
        
        # Predict and decode
        predictions_encoded = self.clf.predict(X_encoded)
        predictions = self.target_encoder.inverse_transform(predictions_encoded)
        
        return predictions
    
    def score(self, df):
        """Calculate accuracy on a dataset"""
        X = df[self.feature_names_original]
        y_true = df[self.target_col]
        y_pred = self.predict(X)
        
        accuracy = (y_pred == y_true).sum() / len(y_true)
        return accuracy
    
    def plot_decision_regions(self, feature1=None, feature2=None, figsize=(10, 8)):
        """
        Plot decision regions for two features.
        
        Parameters:
        -----------
        feature1 : str, optional
            First feature to plot (original feature name, not one-hot encoded)
            If None, uses tree's first split
        feature2 : str, optional
            Second feature to plot (original feature name, not one-hot encoded)
            If None, uses tree's second split
        figsize : tuple
            Figure size
        
        Examples:
        ---------
        >>> tree = build_tree(df, target='class', max_depth=3)
        >>> tree.plot_decision_regions()
        >>> tree.plot_decision_regions(feature1='odor', feature2='gill-color')
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import ListedColormap
        
        # If features not specified, try to get them from original features
        if feature1 is None or feature2 is None:
            # Use first two original features
            if len(self.feature_names_original) >= 2:
                feature1 = self.feature_names_original[0]
                feature2 = self.feature_names_original[1]
            else:
                raise ValueError("Need at least 2 features to plot decision regions")
        
        print(f"Plotting decision regions for: {feature1} vs {feature2}")
        
        # Check if features are numeric or categorical
        is_numeric1 = feature1 in self.numeric_features
        is_numeric2 = feature2 in self.numeric_features
        
        # Get the actual values to plot
        if is_numeric1:
            X1 = self.df[feature1].values
            feature1_values = None  # Continuous
        else:
            # For categorical, we'll use encoded values but label with original
            feature1_encoder = LabelEncoder()
            X1 = feature1_encoder.fit_transform(self.df[feature1])
            feature1_values = sorted(self.df[feature1].unique())
        
        if is_numeric2:
            X2 = self.df[feature2].values
            feature2_values = None  # Continuous
        else:
            # For categorical, we'll use encoded values but label with original
            feature2_encoder = LabelEncoder()
            X2 = feature2_encoder.fit_transform(self.df[feature2])
            feature2_values = sorted(self.df[feature2].unique())
        
        y = self.y_encoded
        
        # Create mesh grid
        x_min, x_max = X1.min() - 0.5, X1.max() + 0.5
        y_min, y_max = X2.min() - 0.5, X2.max() + 0.5
        h = 0.02 if (is_numeric1 or is_numeric2) else 0.1  # Coarser grid for categorical
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Build prediction data for mesh
        n_mesh = len(mesh_points)
        
        # Start with a base DataFrame using median/mode values
        base_data = {}
        for feat in self.feature_names_original:
            if feat in self.numeric_features:
                base_data[feat] = [self.df[feat].median()] * n_mesh
            else:
                base_data[feat] = [self.df[feat].mode()[0]] * n_mesh
        
        mesh_df = pd.DataFrame(base_data)
        
        # Replace with our two features of interest
        if is_numeric1:
            mesh_df[feature1] = mesh_points[:, 0]
        else:
            # Map back to categorical values
            mesh_df[feature1] = [feature1_values[int(val)] if 0 <= int(val) < len(feature1_values) 
                                else feature1_values[0] for val in mesh_points[:, 0]]
        
        if is_numeric2:
            mesh_df[feature2] = mesh_points[:, 1]
        else:
            # Map back to categorical values
            mesh_df[feature2] = [feature2_values[int(val)] if 0 <= int(val) < len(feature2_values) 
                                else feature2_values[0] for val in mesh_points[:, 1]]
        
        # Encode the mesh data the same way as training data
        # Numeric features
        if self.numeric_features:
            X_mesh_encoded = mesh_df[self.numeric_features].copy()
        else:
            X_mesh_encoded = pd.DataFrame()
        
        # One-hot encode categorical features
        if self.categorical_features:
            X_categorical = mesh_df[self.categorical_features]
            X_onehot = pd.get_dummies(X_categorical, prefix=self.categorical_features)
            
            # Ensure all columns from training are present
            missing_cols = [col for col in self.feature_names 
                            if col not in X_onehot.columns and col not in self.numeric_features]

            if missing_cols:
                # Add all missing columns at once
                missing_df = pd.DataFrame(0, index=X_onehot.index, columns=missing_cols)
                X_onehot = pd.concat([X_onehot, missing_df], axis=1)
            
            # Combine and reorder
            if self.numeric_features:
                X_mesh_encoded = pd.concat([X_mesh_encoded, X_onehot[self.feature_names[len(self.numeric_features):]]], axis=1)
            else:
                X_mesh_encoded = X_onehot[self.feature_names]
        
        # Predict on mesh
        Z = self.clf.predict(X_mesh_encoded)
        Z = Z.reshape(xx.shape)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot decision regions
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
        cmap_bold = ['red', 'green']
        
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
        
        # Plot decision boundaries
        ax.contour(xx, yy, Z, colors='black', linewidths=2, levels=[0.5])
        
        # Plot data points
        for idx, class_label in enumerate(self.class_names):
            mask = y == idx
            ax.scatter(X1[mask], X2[mask], 
                    c=cmap_bold[idx], 
                    label=f'{self.target_col}={class_label}',
                    alpha=0.6,
                    edgecolors='black',
                    s=50)
        
        # Set labels
        ax.set_xlabel(feature1, fontsize=12, fontweight='bold')
        ax.set_ylabel(feature2, fontsize=12, fontweight='bold')
        ax.set_title(f'Decision Regions: {feature1} vs {feature2}', 
                    fontsize=14, fontweight='bold')
        
        # Add tick labels
        if not is_numeric1 and feature1_values and len(feature1_values) <= 10:
            ax.set_xticks(range(len(feature1_values)))
            ax.set_xticklabels(feature1_values, rotation=45, ha='right')
        
        if not is_numeric2 and feature2_values and len(feature2_values) <= 10:
            ax.set_yticks(range(len(feature2_values)))
            ax.set_yticklabels(feature2_values)
        
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

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


def visualize_tree(tree, df=None, figsize=(20, 10), fontsize=10, show_probabilities=False):
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
    if show_probabilities:
        plot_tree(
            tree.clf,
            feature_names=tree.feature_names,
            class_names=tree.class_names,
            filled=True,
            rounded=True,
            fontsize=fontsize,
            impurity=False,
            proportion=True,
            precision=3
        )    
    else:
        plot_tree(
            tree.clf,
            feature_names=tree.feature_names,
            class_names=tree.class_names,
            filled=True,
            rounded=True,
            fontsize=fontsize,
            impurity=False,
            precision=2
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