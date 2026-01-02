# cuanalytics/svm/classifier.py
"""
Support Vector Machine classifier implementation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

class SVMModel:
    """
    Support Vector Machine model for binary classification.
    
    This class wraps scikit-learn's SVC to provide a student-friendly interface
    for learning about Support Vector Machines.
    
    Attributes:
    -----------
    df : pd.DataFrame
        Training data
    target : str
        Name of target column
    C : float
        Regularization parameter (soft margin control)
    svm : SVC
        Fitted scikit-learn SVM model
    feature_names : list
        Names of feature columns
    classes : list
        Unique class labels
    """
    
    def __init__(self, df, C=1.0, formula=None):
        """
        Create and fit a Support Vector Machine model.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features and target
        C : float, default=1.0
            Regularization parameter. Higher values mean stricter margin
            (fewer misclassifications allowed during training).
            Lower values allow more flexibility (wider margin, more errors ok).
        formula : str
            Formula for specifying target and features
        """
        self.df = df
        self.C = C
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
        self.X = X
        self.y = y
        
        # Store feature and class info
        self.feature_names = list(self.X.columns)
        self.classes = sorted(self.y.unique())
        
        # Validate
        if len(self.classes) != 2:
            raise ValueError(
                f"SVM currently supports only binary classification (2 classes).\n"
                f"Found {len(self.classes)} classes: {self.classes}\n"
                f"Please filter your data to include only 2 classes."
            )
        
        # Check if source features are numeric
        used_vars = set()
        if self.model_spec is not None:
            used_vars = {var for var in getattr(self.model_spec, 'variables', set())
                         if var in df_rhs.columns}
        if not used_vars:
            used_vars = set(df_rhs.columns)
        non_numeric = [col for col in used_vars
                       if not pd.api.types.is_numeric_dtype(df_rhs[col])]
        if non_numeric:
            raise ValueError(
                f"All features must be numeric. Non-numeric features found: {non_numeric}\n"
                "Hint: Use pd.get_dummies() or LabelEncoder to convert categorical features."
            )
        
        # Create and fit SVM
        self._fit()
        
        # Print summary
        self._print_fit_summary()
    
    def _fit(self):
        """Fit the SVM model."""
        # Use linear kernel with specified C parameter
        self.svm = SVC(kernel='linear', C=self.C)
        self.svm.fit(self.X, self.y)
    
    def _print_fit_summary(self):
        """Print a brief summary after fitting."""
        n_support = self.svm.n_support_.sum()
        support_pct = (n_support / len(self.df)) * 100
        
        print(f"\nSVM Model fitted successfully!")
        print(f"  Classes: {self.classes}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Training samples: {len(self.df)}")
        print(f"  Support vectors: {n_support} ({support_pct:.1f}% of training data)")
        print(f"  C parameter: {self.C}")
    
    def _check_fitted(self):
        """Check if the model has been fitted."""
        if not hasattr(self, 'svm') or self.svm is None:
            raise RuntimeError(
                "Model has not been fitted yet. "
                "Create model with: svm = fit_svm(df, formula='class ~ x1 + x2')"
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
        Predict classes for new data.
        
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
        X = self._transform_data_with_formula(df)
        predictions = self.svm.predict(X)
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
        X = self._transform_data_with_formula(df)
        y_true = df[self.target]

        y_pred = self.svm.predict(X)
        report = self._compute_classification_metrics(y_true, y_pred)
        self._print_score_report(report)
        return report
    
    def get_support_vectors(self):
        """
        Get the support vectors.
        
        Returns:
        --------
        support_vectors_df : pd.DataFrame
            DataFrame containing the support vectors with their features
        """
        self._check_fitted()
        
        # Get support vector indices
        support_indices = self.svm.support_
        
        # Get the actual support vectors as DataFrame
        support_vectors_df = self.df.iloc[support_indices].copy()
        
        return support_vectors_df
    
    def get_coefficients(self):
        """
        Get the coefficients of the linear decision function.
        
        Returns:
        --------
        coef_df : pd.DataFrame
            DataFrame with feature names and their coefficients
        """
        self._check_fitted()
        
        coef_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.svm.coef_[0]
        })
        
        return coef_df
    
    def get_margin_width(self):
        """
        Calculate the margin width.
        
        The margin is the distance between the decision boundary and the 
        nearest support vectors. Larger margin = more confident separation.
        
        Returns:
        --------
        margin : float
            Width of the margin
        """
        self._check_fitted()
        
        # Margin width = 2 / ||w|| where w is the coefficient vector
        w = self.svm.coef_[0]
        margin = 2.0 / np.linalg.norm(w)
        
        return margin

    def get_metrics(self):
        """
        Calculate classification metrics on training data.

        Returns:
        --------
        metrics : dict
            Dictionary with model metrics and statistics
        """
        self._check_fitted()

        metrics = {
            'model_type': 'svm',
            'target': self.target,
            'n_samples': len(self.y),
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'classes': self.classes
        }

        y_pred = self.svm.predict(self.X)
        y_true = self.y

        metrics['score'] = self._compute_classification_metrics(y_true, y_pred)
        metrics['coefficients'] = self.get_coefficients()
        metrics['margin_width'] = self.get_margin_width()
        metrics['n_support_vectors'] = int(self.svm.n_support_.sum())

        return metrics
    
    def visualize(self, figsize=(12, 5)):
        """
        Visualize SVM with support vectors highlighted.
        
        Shows a scatter plot of the first two principal components with
        support vectors marked distinctly.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        self._check_fitted()
        
        from sklearn.decomposition import PCA
        
        # Project to 2D using PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X)
        
        # Get support vector indices
        support_indices = self.svm.support_
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Data with support vectors highlighted
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.classes)))
        
        for i, class_name in enumerate(self.classes):
            mask = self.y == class_name
            
            # Regular points
            regular_mask = mask & ~np.isin(range(len(self.X)), support_indices)
            ax1.scatter(X_pca[regular_mask, 0], X_pca[regular_mask, 1],
                       c=[colors[i]], label=class_name, alpha=0.6, 
                       s=50, edgecolors='black', linewidth=0.5)
            
            # Support vectors
            support_mask = mask & np.isin(range(len(self.X)), support_indices)
            if support_mask.any():
                ax1.scatter(X_pca[support_mask, 0], X_pca[support_mask, 1],
                           c=[colors[i]], marker='s', s=200, 
                           edgecolors='black', linewidth=3,
                           label=f'{class_name} (support vectors)')
        
        ax1.set_xlabel('First Principal Component', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Second Principal Component', fontsize=12, fontweight='bold')
        ax1.set_title('SVM with Support Vectors', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Support vector counts
        support_counts = []
        for class_name in self.classes:
            mask = self.y == class_name
            support_mask = mask & np.isin(range(len(self.X)), support_indices)
            support_counts.append(support_mask.sum())
        
        ax2.bar(range(len(self.classes)), support_counts, color=colors, 
               edgecolor='black', linewidth=2)
        ax2.set_xticks(range(len(self.classes)))
        ax2.set_xticklabels(self.classes)
        ax2.set_ylabel('Number of Support Vectors', fontsize=12, fontweight='bold')
        ax2.set_title('Support Vectors per Class', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("\nVisualization Notes:")
        print("  • Square markers (□) = support vectors (the critical points)")
        print("  • Circle markers (●) = regular data points")
        print("  • Support vectors define the decision boundary")
        print("  • All other points could be removed without changing the model")
    
    def visualize_features(self, feature1, feature2, figsize=(10, 8)):
        """
        Visualize SVM decision boundary for two specific features.
        
        Shows the decision boundary, margin, and support vectors in 2D feature space.
        
        Parameters:
        -----------
        feature1 : str
            First feature for x-axis
        feature2 : str
            Second feature for y-axis
        figsize : tuple
            Figure size
        """
        self._check_fitted()
        
        if feature1 not in self.feature_names or feature2 not in self.feature_names:
            raise ValueError(
                f"Features must be from: {self.feature_names}\n"
                f"You provided: {feature1}, {feature2}"
            )
        
        # Get the data for these two features
        X_2d = self.df[[feature1, feature2]].values
        y = self.y.values
        
        # Retrain SVM on just these two features for visualization
        svm_2d = SVC(kernel='linear', C=self.C)
        svm_2d.fit(X_2d, y)
        
        # Create mesh for decision boundary
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        h = 0.02
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # Get decision function values
        Z = svm_2d.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.classes)))
        
        # Plot decision boundary and margins
        ax.contour(xx, yy, Z, colors='black', levels=[-1, 0, 1], 
                  alpha=0.5, linestyles=['--', '-', '--'], linewidths=[2, 3, 2])
        
        # Plot decision regions
        ax.contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf], 
                   colors=colors, alpha=0.2)
        
        # Get support vector indices for 2D model
        support_indices_2d = svm_2d.support_
        
        # Plot data points
        for i, class_name in enumerate(self.classes):
            mask = y == class_name
            
            # Regular points
            regular_mask = mask & ~np.isin(range(len(X_2d)), support_indices_2d)
            ax.scatter(X_2d[regular_mask, 0], X_2d[regular_mask, 1],
                      c=[colors[i]], label=class_name, alpha=0.7,
                      s=80, edgecolors='black', linewidth=1.5)
            
            # Support vectors
            support_mask = mask & np.isin(range(len(X_2d)), support_indices_2d)
            if support_mask.any():
                ax.scatter(X_2d[support_mask, 0], X_2d[support_mask, 1],
                          c=[colors[i]], marker='s', s=250,
                          edgecolors='black', linewidth=3,
                          label=f'{class_name} (support vectors)')
        
        ax.set_xlabel(feature1, fontsize=13, fontweight='bold')
        ax.set_ylabel(feature2, fontsize=13, fontweight='bold')
        ax.set_title(f'SVM Decision Boundary: {feature1} vs {feature2}',
                    fontsize=15, fontweight='bold')
        
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Calculate margin for 2D case
        w_2d = svm_2d.coef_[0]
        margin_2d = 2.0 / np.linalg.norm(w_2d)
        
        print(f"\nSVM Decision Boundary (2D visualization):")
        print(f"  • Solid line = decision boundary")
        print(f"  • Dashed lines = margin boundaries")
        print(f"  • Square markers (□) = support vectors")
        print(f"  • The region between dashed lines is the 'margin'")
        print(f"  • Margin width: {margin_2d:.4f}")
        print(f"  • Only support vectors affect the boundary position")
    
    def summary(self):
        """
        Print detailed summary of SVM model.
        
        Shows decision function, margin width, support vectors, and training fit.
        
        Returns:
        --------
        None
        """
        self._check_fitted()
        
        # Get predictions and metrics
        y_pred_train = self.svm.predict(self.X)
        train_metrics = self._compute_classification_metrics(self.y, y_pred_train)
        train_fit = train_metrics['accuracy']
        conf_matrix_train = confusion_matrix(self.y, y_pred_train)
        
        # Get support vector info
        support_indices = self.svm.support_
        n_support = len(support_indices)
        support_pct = (n_support / len(self.df)) * 100
        
        # Get margin
        margin = self.get_margin_width()
        
        # Print summary
        print("\n" + "="*70)
        print("SUPPORT VECTOR MACHINE MODEL SUMMARY")
        print("="*70)
        
        # Model information
        print("\nMODEL INFORMATION:")
        print("-" * 70)
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Number of classes: {len(self.classes)}")
        print(f"Classes: {self.classes}")
        print(f"C parameter: {self.C}")
        print(f"  (Higher C = stricter margin, fewer training errors allowed)")
        print(f"  (Lower C = more flexible, wider margin, more errors ok)")
        print(f"Training samples: {len(self.df)}")
        print(f"Number of support vectors: {n_support} ({support_pct:.1f}% of training data)")
        
        # Decision function
        print("\nDECISION FUNCTION:")
        print("-" * 70)
        
        # Build equation string
        coef = self.svm.coef_[0]
        intercept = self.svm.intercept_[0]
        
        terms = []
        for j, feature in enumerate(self.feature_names):
            if j == 0:
                terms.append(f"{coef[j]:.4f}×{feature}")
            else:
                sign = "+" if coef[j] >= 0 else "-"
                terms.append(f"{sign} {abs(coef[j]):.4f}×{feature}")
        
        equation = " ".join(terms)
        sign = "+" if intercept >= 0 else "-"
        equation += f" {sign} {abs(intercept):.4f}"
        
        print(f"\nf(x) = {equation}")
        
        print(f"\nDecision Rule:")
        print(f"  If f(x) > 0: predict '{self.classes[1]}'")
        print(f"  If f(x) ≤ 0: predict '{self.classes[0]}'")
        print(f"  If f(x) = 0: on the decision boundary")
        
        print(f"\nNote: Use raw feature values directly (no centering needed).")
        print(f"      This is the decision function used by predict().")
        
        # Margin width
        print("\nMARGIN WIDTH:")
        print("-" * 70)
        print(f"Margin: {margin:.4f} units")
        print(f"\nThe margin is the 'safety zone' between classes.")
        print(f"  • Larger margin = more confident separation")
        print(f"  • Points outside the margin are easy to classify")
        print(f"  • Support vectors lie on or within the margin boundaries")
        
        # Support vectors
        print("\nSUPPORT VECTORS:")
        print("-" * 70)
        
        for i, class_name in enumerate(self.classes):
            mask = self.y == class_name
            support_mask = mask & np.isin(range(len(self.X)), support_indices)
            n_class_support = support_mask.sum()
            print(f"  {class_name}: {n_class_support} support vectors")
        
        print(f"\nSupport vectors are the critical data points that define the boundary.")
        print(f"Only these {n_support} points affect the decision boundary.")
        print(f"All other {len(self.df) - n_support} points could be removed without changing the model.")
        
        # Training fit
        print("\nTRAINING FIT:")
        print("-" * 70)
        print(f"Training Set Fit: {train_fit:.2%}")
        print("(This measures how well the model fits the training data)")
        
        print("\nTraining Confusion Matrix:")
        conf_df_train = pd.DataFrame(
            conf_matrix_train,
            index=[f"Actual {c}" for c in self.classes],
            columns=[f"Pred {c}" for c in self.classes]
        )
        print(conf_df_train.to_string())
        print(f"\nKappa: {train_metrics['kappa']:.4f}")
        
        print("\n" + "="*70)
        
def fit_svm(df, C=1.0, formula=None):
    """
    Fit a Support Vector Machine model for binary classification.
    
    This function creates and fits an SVM model using a linear kernel.
    The model finds the decision boundary that maximally separates two classes
    while maintaining the largest possible margin.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Training data with features and target column
    C : float, default=1.0
        Regularization parameter (soft margin control).
        - Higher values (e.g., 10, 100) = stricter margin, fewer errors allowed
        - Lower values (e.g., 0.1, 0.01) = more flexible, wider margin
        - Default of 1.0 is usually a good starting point
    formula : str
        Formula for specifying target and features
    
    Returns:
    --------
    model : SVMModel
        Fitted SVM model
    
    Examples:
    ---------
    >>> from cuanalytics import fit_svm, split_data, load_iris_data
    >>> df = load_iris_data()
    >>> df_binary = df[df['species'].isin(['Iris-setosa', 'Iris-versicolor'])]
    >>> train, test = split_data(df_binary, test_size=0.2)
    >>> svm = fit_svm(train, formula='species ~ .', C=1.0)
    >>> svm.summary()
    >>> train_acc = svm.score(train)['accuracy']
    >>> test_acc = svm.score(test)['accuracy']
    """
    if formula is None:
        raise ValueError("Must provide 'formula' for model specification")
    return SVMModel(df, C=C, formula=formula)
