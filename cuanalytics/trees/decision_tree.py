# cuanalytics/trees/decision_tree.py
"""
Simple decision tree interface for students
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


class SimpleDecisionTree:
    """
    Wrapper around scikit-learn DecisionTreeClassifier.
    Handles both numeric and categorical features automatically.
    """
    
    def __init__(self, df, max_depth=None, criterion='entropy', formula=None):
        """
        Create and fit a decision tree classifier.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training data with features and target
        max_depth : int, optional
            Maximum depth of the tree (None = unlimited)
        criterion : str
            Split criterion: 'entropy' (information gain) or 'gini'
        formula : str
            Formula for specifying target and features
        """
        self.df = df
        self.original_df = df
        self.max_depth = max_depth
        self.criterion = criterion
        self.formula = formula
        self.model_spec = None

        if formula is None:
            raise ValueError("Must provide 'formula' for model specification")

        try:
            from formulaic import model_matrix
        except ImportError:
            raise ImportError(
                "Formula support requires the 'formulaic' library.\n"
                "Install it with: pip install formulaic"
            )

        if '~' not in formula:
            raise ValueError("Formula must include a target (e.g., 'class ~ x1 + x2').")

        lhs, rhs = formula.split('~', 1)
        lhs = lhs.strip()
        rhs = rhs.strip()

        if lhs and lhs not in df.columns and lhs.isidentifier():
            raise ValueError(f"Target '{lhs}' not found in data")

        y = df[lhs]
        df_rhs = df.drop(columns=[lhs])
        model_matrices = model_matrix(rhs, df_rhs, output='pandas')

        if hasattr(model_matrices, 'rhs'):
            X = model_matrices.rhs
        else:
            X = model_matrices

        self.model_spec = getattr(model_matrices, 'model_spec', None)
        self.target = lhs

        if 'Intercept' in X.columns:
            X = X.drop('Intercept', axis=1)

        self.df = X.copy()
        self.df[self.target] = y
        self.X_encoded = X.copy()
        self.feature_names = list(X.columns)
        self.feature_names_original = [col for col in df.columns if col != self.target]
        self.numeric_features = [col for col in self.feature_names_original
                                 if pd.api.types.is_numeric_dtype(df[col])]
        self.categorical_features = [col for col in self.feature_names_original
                                     if col not in self.numeric_features]
        self.classes = sorted(self.df[self.target].unique())

        self.target_encoder = LabelEncoder()
        self.y_encoded = self.target_encoder.fit_transform(self.df[self.target])

        self._train()
        self._print_fit_summary()
    
    def _train(self):
        """Train the decision tree"""
        self.tree = DecisionTreeClassifier(  # Changed from clf to tree
            criterion=self.criterion,
            max_depth=self.max_depth,
            random_state=42
        )
        self.tree.fit(self.X_encoded, self.y_encoded)
    
    def _print_fit_summary(self):
        """Print a brief summary after fitting."""
        print(f"\nDecision Tree fitted successfully!")
        print(f"  Classes: {self.classes}")
        print(f"  Features: {len(self.feature_names_original)} original ({len(self.feature_names)} after encoding)")
        print(f"  Numeric features: {self.numeric_features}")
        print(f"  Categorical features: {self.categorical_features}")
        print(f"  Training samples: {len(self.df)}")
        print(f"  Max depth: {self.max_depth if self.max_depth else 'unlimited'}")
        print(f"  Criterion: {self.criterion}")
    
    def _check_fitted(self):
        """Check if the model has been fitted."""
        if not hasattr(self, 'tree') or self.tree is None:
            raise RuntimeError(
                "Model has not been fitted yet. "
                "Create model with: tree = fit_tree(df, formula='class ~ x1 + x2')"
            )

    def _compute_classification_metrics(self, y_true, y_pred):
        """Compute confusion-matrix based metrics."""
        cm = confusion_matrix(y_true, y_pred, labels=self.classes)
        total = cm.sum()
        accuracy = np.trace(cm) / total if total else 0.0
        row_marginals = cm.sum(axis=1)
        col_marginals = cm.sum(axis=0)
        expected = (row_marginals * col_marginals).sum() / (total ** 2) if total else 0.0
        kappa = (accuracy - expected) / (1 - expected) if (1 - expected) else 0.0

        per_class = {}
        for idx, label in enumerate(self.classes):
            tp = cm[idx, idx]
            fn = cm[idx, :].sum() - tp
            fp = cm[:, idx].sum() - tp
            tn = total - tp - fn - fp

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            specificity = tn / (tn + fp) if (tn + fp) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

            per_class[label] = {
                'precision': precision,
                'recall': recall,
                'sensitivity': recall,
                'specificity': specificity,
                'f1': f1
            }

        return {
            'accuracy': accuracy,
            'kappa': kappa,
            'confusion_matrix': cm,
            'per_class': per_class,
        }

    def _print_score_report(self, report):
        """Print a summary of classification performance."""
        print("\nSCORE REPORT")
        print("=" * 60)
        print(f"Accuracy: {report['accuracy']:.2%}")
        print(f"Kappa: {report['kappa']:.4f}")

        conf_df = pd.DataFrame(
            report['confusion_matrix'],
            index=[f"Actual {c}" for c in self.classes],
            columns=[f"Pred {c}" for c in self.classes]
        )
        print("\nConfusion Matrix:")
        print(conf_df.to_string())

        per_class_df = pd.DataFrame(report['per_class']).T
        print("\nPer-Class Metrics:")
        print(per_class_df.to_string(float_format=lambda x: f"{x:.4f}"))


    def _transform_data_with_formula(self, df):
        """
        Transform data using the stored formula specification.
        """
        if getattr(self, 'model_spec', None) is None:
            raise RuntimeError("Formula metadata missing; cannot transform new data.")

        rhs_spec = getattr(self.model_spec, 'rhs', self.model_spec)
        df_rhs = df.drop(columns=[self.target], errors='ignore')
        model_matrices = rhs_spec.get_model_matrix(df_rhs, output='pandas')

        if hasattr(model_matrices, 'rhs'):
            X_new = model_matrices.rhs
        else:
            X_new = model_matrices

        if 'Intercept' in X_new.columns:
            X_new = X_new.drop('Intercept', axis=1)

        return X_new
    
    def predict(self, df):
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data to predict (with or without target column)
        
        Returns:
        --------
        predictions : array
            Predicted class labels
        """
        self._check_fitted()
        
        X_encoded = self._transform_data_with_formula(df)
        
        # Predict and decode
        predictions_encoded = self.tree.predict(X_encoded)
        predictions = self.target_encoder.inverse_transform(predictions_encoded)

        return pd.Series(predictions, index=df.index, name=self.target)
    
    def score(self, df):
        """
        Calculate and print classification metrics on a dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with true labels
        
        Returns:
        --------
        metrics : dict
            Dictionary of accuracy, confusion matrix, and derived metrics
        """
        self._check_fitted()

        y_true = df[self.target]
        y_pred = self.predict(df)

        report = self._compute_classification_metrics(y_true, y_pred)
        self._print_score_report(report)
        return report
    
    def get_feature_importance(self):
        """
        Get feature importance from the decision tree.
        
        Returns:
        --------
        importance : pd.DataFrame
            Features ranked by importance
        
        Examples:
        ---------
        >>> tree = fit_tree(df, formula='class ~ .')
        >>> importance = tree.get_feature_importance()
        >>> print(importance.head())
        """
        self._check_fitted()
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.tree.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance

    def get_metrics(self):
        """
        Calculate classification metrics on training data.

        Returns:
        --------
        metrics : dict
            Dictionary with model metrics and statistics
        """
        self._check_fitted()
        y_pred_encoded = self.tree.predict(self.X_encoded)
        y_pred = self.target_encoder.inverse_transform(y_pred_encoded)
        y_true = self.df[self.target]
        report = self._compute_classification_metrics(y_true, y_pred)

        metrics = {
            'model_type': 'decision_tree',
            'target': self.target,
            'n_samples': len(self.df),
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'classes': self.classes,
            'score': report,
            'feature_importance': self.get_feature_importance()
        }

        return metrics
    
    def get_rules(self):
        """
        Get text representation of decision tree rules in sentence format.

        Each rule traces a path from root to leaf, showing all conditions
        that must be true to reach that leaf node's prediction.

        Returns:
        --------
        rules : str
            Text-based decision rules in IF-THEN format

        Examples:
        ---------
        >>> tree = fit_tree(df, formula='class ~ .', max_depth=2)
        >>> print(tree.get_rules())
        IF (Balance < 50K) AND (Age >= 50) THEN Class=No Write Off
        """
        self._check_fitted()

        tree_model = self.tree.tree_
        feature_names = self.feature_names
        class_names = self.classes

        def get_leaf_paths(node=0, path_conditions=None):
            """Recursively traverse tree and collect all paths to leaf nodes"""
            if path_conditions is None:
                path_conditions = []

            # Check if this is a leaf node
            if tree_model.feature[node] == -2:  # -2 indicates leaf node
                # Get the predicted class
                class_idx = np.argmax(tree_model.value[node][0])
                predicted_class = class_names[class_idx]

                # Format the rule
                if len(path_conditions) == 0:
                    return [f"IF (Always) THEN Class={predicted_class}"]
                else:
                    conditions_str = " AND ".join(path_conditions)
                    return [f"IF {conditions_str} THEN Class={predicted_class}"]

            # Internal node - recurse on children
            feature_name = feature_names[tree_model.feature[node]]
            threshold = tree_model.threshold[node]

            rules = []

            # Left child (feature <= threshold)
            left_condition = f"({feature_name} <= {threshold:.2f})"
            left_path = path_conditions + [left_condition]
            rules.extend(get_leaf_paths(tree_model.children_left[node], left_path))

            # Right child (feature > threshold)
            right_condition = f"({feature_name} > {threshold:.2f})"
            right_path = path_conditions + [right_condition]
            rules.extend(get_leaf_paths(tree_model.children_right[node], right_path))

            return rules

        # Get all rules
        all_rules = get_leaf_paths()

        # Format as numbered list
        formatted_rules = []
        formatted_rules.append("Decision Tree Rules:")
        formatted_rules.append("=" * 80)
        for i, rule in enumerate(all_rules, 1):
            formatted_rules.append(f"Rule {i}: {rule}")

        return "\n".join(formatted_rules)
    
    def visualize(self, figsize=(20, 10), fontsize=10, show_probabilities=False):
        """
        Visualize the decision tree structure.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        fontsize : int
            Font size for tree labels
        show_probabilities : bool
            If True, show class probabilities instead of counts
        
        Examples:
        ---------
        >>> tree = fit_tree(df, formula='class ~ .', max_depth=3)
        >>> tree.visualize()
        >>> tree.visualize(show_probabilities=True)
        """
        self._check_fitted()
        
        plt.figure(figsize=figsize)
        if show_probabilities:
            plot_tree(
                self.tree,
                feature_names=self.feature_names,
                class_names=self.classes,
                filled=True,
                rounded=True,
                fontsize=fontsize,
                impurity=False,
                proportion=True,
                precision=3
            )    
        else:
            plot_tree(
                self.tree,
                feature_names=self.feature_names,
                class_names=self.classes,
                filled=True,
                rounded=True,
                fontsize=fontsize,
                impurity=False,
                precision=2
            )
        plt.tight_layout()
        plt.show()
        
        print("\nDecision Tree Visualization:")
        print(f"  • Node color = dominant class")
        print(f"  • Darker color = more confident (purer node)")
        print(f"  • Each node shows: split condition, samples, and class distribution")
    
    def visualize_features(self, feature1=None, feature2=None, figsize=(10, 8)):
        """
        Visualize decision regions for two features.
        
        Parameters:
        -----------
        feature1 : str, optional
            First feature to plot (original feature name)
            If None, uses first feature
        feature2 : str, optional
            Second feature to plot (original feature name)
            If None, uses second feature
        figsize : tuple
            Figure size
        
        Examples:
        ---------
        >>> tree = fit_tree(df, formula='class ~ .', max_depth=3)
        >>> tree.visualize_features()
        >>> tree.visualize_features('odor', 'gill-color')
        """
        self._check_fitted()

        from matplotlib.colors import ListedColormap

        data_df = self.original_df if getattr(self, 'original_df', None) is not None else self.df
        
        # If features not specified, use first two original features
        if feature1 is None or feature2 is None:
            if len(self.feature_names_original) >= 2:
                feature1 = self.feature_names_original[0]
                feature2 = self.feature_names_original[1]
            else:
                raise ValueError("Need at least 2 features to plot decision regions")
        
        if feature1 not in self.feature_names_original or feature2 not in self.feature_names_original:
            raise ValueError(
                f"Features must be from: {self.feature_names_original}\n"
                f"You provided: {feature1}, {feature2}"
            )
        
        # Check if features are numeric or categorical
        is_numeric1 = feature1 in self.numeric_features
        is_numeric2 = feature2 in self.numeric_features
        
        # Get the actual values to plot
        if is_numeric1:
            X1 = data_df[feature1].values
            feature1_values = None  # Continuous
        else:
            # For categorical, we'll use encoded values but label with original
            feature1_encoder = LabelEncoder()
            X1 = feature1_encoder.fit_transform(data_df[feature1])
            feature1_values = sorted(data_df[feature1].unique())
        
        if is_numeric2:
            X2 = data_df[feature2].values
            feature2_values = None  # Continuous
        else:
            # For categorical, we'll use encoded values but label with original
            feature2_encoder = LabelEncoder()
            X2 = feature2_encoder.fit_transform(data_df[feature2])
            feature2_values = sorted(data_df[feature2].unique())
        
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
                base_data[feat] = [data_df[feat].median()] * n_mesh
            else:
                base_data[feat] = [data_df[feat].mode()[0]] * n_mesh
        
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
        
        if self.formula is not None:
            X_mesh_encoded = self._transform_data_with_formula(mesh_df)
        else:
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
                
                # Combine and reorder to match training feature order
                if self.numeric_features:
                    X_mesh_encoded = pd.concat([X_mesh_encoded, X_onehot[self.feature_names[len(self.numeric_features):]]], axis=1)
                else:
                    X_mesh_encoded = X_onehot[self.feature_names]
        
        # Predict on mesh
        Z = self.tree.predict(X_mesh_encoded)
        Z = Z.reshape(xx.shape)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot decision regions
        n_classes = len(self.classes)
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
        cmap_light = ListedColormap(colors[:n_classes])
        
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
        
        # Plot decision boundaries
        ax.contour(xx, yy, Z, colors='black', linewidths=2, 
                  levels=np.arange(0.5, n_classes, 1))
        
        # Plot data points
        for idx, class_label in enumerate(self.classes):
            mask = y == idx
            ax.scatter(X1[mask], X2[mask], 
                    c=[colors[idx]], 
                    label=f'{self.target}={class_label}',
                    alpha=0.6,
                    edgecolors='black',
                    s=50)
        
        # Set labels
        ax.set_xlabel(feature1, fontsize=12, fontweight='bold')
        ax.set_ylabel(feature2, fontsize=12, fontweight='bold')
        ax.set_title(f'Decision Tree Regions: {feature1} vs {feature2}', 
                    fontsize=14, fontweight='bold')
        
        # Add tick labels for categorical features
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
        
        print(f"\nDecision Regions (2D visualization):")
        print(f"  • Shows how tree classifies based on {feature1} and {feature2}")
        print(f"  • Other features held at median (numeric) or mode (categorical)")
        print(f"  • Black lines = decision boundaries")

    def summary(self):
        """
        Print a detailed summary of the decision tree model.

        Shows model configuration, training fit, and feature importance.
        """
        self._check_fitted()

        y_pred_encoded = self.tree.predict(self.X_encoded)
        y_pred = self.target_encoder.inverse_transform(y_pred_encoded)
        y_true = self.df[self.target]
        train_metrics = self._compute_classification_metrics(y_true, y_pred)

        print("\n" + "=" * 70)
        print("DECISION TREE MODEL SUMMARY")
        print("=" * 70)

        print("\nMODEL INFORMATION:")
        print("-" * 70)
        print(f"Target: {self.target}")
        print(f"Number of features: {len(self.feature_names_original)} original ({len(self.feature_names)} after encoding)")
        print(f"Classes: {self.classes}")
        print(f"Training samples: {len(self.df)}")
        print(f"Max depth: {self.max_depth if self.max_depth else 'unlimited'}")
        print(f"Criterion: {self.criterion}")

        print("\nTRAINING FIT:")
        print("-" * 70)
        print(f"Training Accuracy: {train_metrics['accuracy']:.2%}")

        conf_df = pd.DataFrame(
            train_metrics['confusion_matrix'],
            index=[f"Actual {c}" for c in self.classes],
            columns=[f"Pred {c}" for c in self.classes]
        )
        print("\nTraining Confusion Matrix:")
        print(conf_df.to_string())

        per_class_df = pd.DataFrame(train_metrics['per_class']).T
        print("\nPer-Class Metrics:")
        print(per_class_df.to_string(float_format=lambda x: f"{x:.4f}"))

        print(f"\nKappa: {train_metrics['kappa']:.4f}")

        print("\nFEATURE IMPORTANCE:")
        print("-" * 70)
        importance_df = self.get_feature_importance()
        print(importance_df.to_string(index=False))

        print("\n" + "=" * 70)


def fit_tree(df, max_depth=None, criterion='entropy', formula=None):
    """
    Fit a decision tree for classification.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Training data with features and target
    max_depth : int, optional
        Maximum depth of the tree (None = unlimited)
    criterion : str
        Split criterion: 'entropy' (information gain) or 'gini'
    formula : str
        Formula for specifying target and features
    
    Returns:
    --------
    tree : SimpleDecisionTree
        Fitted decision tree
    
    Examples:
    ---------
    >>> from cuanalytics import load_mushroom_data, fit_tree, split_data
    >>> df = load_mushroom_data()
    >>> train, test = split_data(df, test_size=0.2)
    >>> tree = fit_tree(train, formula='class ~ .', max_depth=3)
    >>> tree.visualize()
    >>> tree.visualize_features('odor', 'spore-print-color')
    >>> train_acc = tree.score(train)['accuracy']
    >>> test_acc = tree.score(test)['accuracy']
    """
    if formula is None:
        raise ValueError("Must provide 'formula' for model specification")
    return SimpleDecisionTree(df, max_depth, criterion, formula=formula)
