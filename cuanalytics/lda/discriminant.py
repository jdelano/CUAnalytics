"""
Linear Discriminant Analysis helper functions for classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix


class LDAModel:
    """
    Wrapper around scikit-learn's LinearDiscriminantAnalysis.
    Provides a simple interface for students.
    """
    
    def __init__(self, df, n_components=None, solver='svd', formula=None):
        """
        Create and fit a Linear Discriminant Analysis model.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features and target
        n_components : int, optional
            Number of discriminant components to keep
            If None, keeps min(n_features, n_classes - 1)
        solver : str
            Solver to use: 'svd' (default), 'lsqr', or 'eigen'
        formula : str
            Formula for specifying target and features
        """
        self.df = df
        self.original_df = df
        self.n_components = n_components
        self.solver = solver
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
            raise ValueError("All features must be numeric. Encode categorical variables first.")
        
        # Create and fit LDA
        self._fit()
        
        # Print summary
        self._print_fit_summary()
    
    def _fit(self):
        """Fit the LDA model"""
        self.lda = LinearDiscriminantAnalysis(
            n_components=self.n_components,
            solver=self.solver
        )
        self.lda.fit(self.X, self.y)
    
    def _print_fit_summary(self):
        """Print fit summary"""
        print(f"Linear Discriminant Analysis Model")
        print("=" * 60)
        print(f"Number of classes: {len(self.classes)}")
        print(f"Classes: {self.classes}")
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Number of components: {self.lda.n_components if hasattr(self.lda, 'n_components') else 'N/A'}")
        y_pred = self.lda.predict(self.X)
        train_metrics = self._compute_classification_metrics(self.y, y_pred)
        print(f"Training accuracy: {train_metrics['accuracy']:.2%}")
    
    def _check_fitted(self):
        """Check if the model has been fitted"""
        if not hasattr(self, 'lda') or self.lda is None:
            raise RuntimeError(
                "Model has not been fitted yet. "
                "Create model with: lda = fit_lda(df, formula='class ~ x1 + x2')"
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
            Data to predict (without target column)
        
        Returns:
        --------
        predictions : array
            Predicted class labels
        """
        self._check_fitted()
        X = self._transform_data_with_formula(df)
        predictions = self.lda.predict(X)
        return pd.Series(predictions, index=df.index, name=self.target)
    
    def predict_proba(self, df):
        """
        Predict class probabilities for new data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data to predict (without target column)
        
        Returns:
        --------
        probabilities : array
            Predicted probabilities for each class
        """
        self._check_fitted()
        X = self._transform_data_with_formula(df)
        return self.lda.predict_proba(X)
    
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

        y_pred = self.lda.predict(X)
        report = self._compute_classification_metrics(y_true, y_pred)
        self._print_score_report(report)
        return report
    
    def transform(self, df):
        """
        Transform data to LDA space (dimensionality reduction).
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data to transform
        
        Returns:
        --------
        transformed : array
            Data in LDA space
        """
        X = self._transform_data_with_formula(df)
        return self.lda.transform(X)
    
    def get_feature_importance(self):
        """
        Get feature importance based on LDA coefficients.
        
        Returns:
        --------
        importance_df : pd.DataFrame
            Features ranked by importance
        """
        self._check_fitted()  
        
        # Average absolute coefficients across classes
        feature_importance = np.abs(self.lda.coef_).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df

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
            'model_type': 'lda',
            'target': self.target,
            'n_samples': len(self.y),
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'classes': self.classes
        }

        y_pred = self.lda.predict(self.X)
        y_true = self.y

        metrics['score'] = self._compute_classification_metrics(y_true, y_pred)
        metrics['feature_importance'] = self.get_feature_importance()

        return metrics

    def visualize(self, figsize=(12, 5)):
        """
        Visualize LDA projection in discriminant space.
        
        Shows how all features combine to separate classes when projected
        onto the discriminant axes (LD1, LD2, etc.).
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        
        Examples:
        ---------
        >>> lda = fit_lda(df, formula='species ~ .')
        >>> lda.visualize()  # Shows projection onto LD1/LD2
        """
        self._check_fitted()
        
        # Use the FITTED model to transform data
        X_lda = self.lda.transform(self.X)
        
        # Determine if we have 1D or 2D projection
        n_components = X_lda.shape[1]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Projection (1D or 2D)
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.classes)))
        
        if n_components >= 2:
            # 2D plot
            for i, class_name in enumerate(self.classes):
                mask = self.y == class_name
                ax1.scatter(X_lda[mask, 0], X_lda[mask, 1], 
                        c=[colors[i]], label=class_name, 
                        alpha=0.6, s=50, edgecolors='black')
            
            ax1.set_xlabel('LD1 (First Discriminant)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('LD2 (Second Discriminant)', fontsize=12, fontweight='bold')
            ax1.set_title('LDA Projection (2D)', fontsize=14, fontweight='bold')
        else:
            # 1D plot (with jitter for visibility)
            for i, class_name in enumerate(self.classes):
                mask = self.y == class_name
                y_jitter = np.random.normal(0, 0.02, mask.sum())
                ax1.scatter(X_lda[mask, 0], y_jitter, 
                        c=[colors[i]], label=class_name, 
                        alpha=0.6, s=50, edgecolors='black')
            
            ax1.set_xlabel('LD1 (First Discriminant)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('(Random jitter for visibility)', fontsize=10, style='italic', color='gray')
            ax1.set_title('LDA Projection (1D)', fontsize=14, fontweight='bold')
            ax1.set_yticks([])
        
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Feature importance
        if hasattr(self.lda, 'coef_'):
            importance_df = self.get_feature_importance()
            
            ax2.barh(range(len(importance_df)), importance_df['importance'], color='steelblue')
            ax2.set_yticks(range(len(importance_df)))
            ax2.set_yticklabels(importance_df['feature'])
            ax2.set_xlabel('Absolute Coefficient (Importance)', fontsize=12, fontweight='bold')
            ax2.set_title('Feature Importance', fontsize=14, fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("\nLDA Projection:")
        print(f"  • Shows all {len(self.feature_names)} features combined via discriminants")
        print(f"  • Points are plotted in {n_components}D discriminant space")
        print(f"  • Separation between classes shows discriminant effectiveness")
        
    def visualize_features(self, feature1, feature2, figsize=(10, 8)):
        """
        Visualize LDA decision boundaries for two features.
        
        Note: This retrains LDA on ONLY the two selected features to show
        how well they separate the classes in 2D space.
        
        Parameters:
        -----------
        feature1 : str
            First feature for x-axis (must be in original feature names)
        feature2 : str
            Second feature for y-axis (must be in original feature names)
        figsize : tuple
            Figure size
        
        Examples:
        ---------
        >>> lda = fit_lda(df, formula='species ~ .')
        >>> lda.visualize('petal_length', 'petal_width')
        """
        self._check_fitted()

        data_df = self.original_df if getattr(self, 'original_df', None) is not None else self.df
        
        # Check that features are in the ORIGINAL feature set
        if feature1 not in self.feature_names or feature2 not in self.feature_names:
            raise ValueError(
                f"Features must be from original features: {self.feature_names}\n"
                f"You provided: {feature1}, {feature2}"
            )
        
        # Get the actual data for these two features from original dataframe
        X_2d = data_df[[feature1, feature2]].values
        y = data_df[self.target].values
        
        # Retrain LDA on just these two features
        lda_2d = LinearDiscriminantAnalysis()
        lda_2d.fit(X_2d, y)
        
        # Create mesh for decision boundary
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        h = 0.02
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Predict using the 2D model
        Z = lda_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        
        # Convert to numeric for plotting
        class_to_num = {cls: i for i, cls in enumerate(lda_2d.classes_)}
        Z_numeric = np.array([class_to_num[z] for z in Z]).reshape(xx.shape)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(lda_2d.classes_)))
        
        # Decision regions
        ax.contourf(xx, yy, Z_numeric, alpha=0.2, cmap='Set1')
        
        # Decision boundaries (thick lines)
        ax.contour(xx, yy, Z_numeric, colors='black', linewidths=3, 
                levels=list(range(len(lda_2d.classes_))))
        
        # Plot actual data points
        for i, class_name in enumerate(lda_2d.classes_):
            mask = y == class_name
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                    c=[colors[i]], label=class_name,
                    alpha=0.7, s=80, edgecolors='black', linewidth=1.5)
        
        # Plot class means
        for i, class_name in enumerate(lda_2d.classes_):
            mask = y == class_name
            mean_x = X_2d[mask, 0].mean()
            mean_y = X_2d[mask, 1].mean()
            ax.scatter(mean_x, mean_y, c=[colors[i]], 
                    marker='*', s=500, edgecolors='black', 
                    linewidth=2, label=f'{class_name} mean')
        
        ax.set_xlabel(feature1, fontsize=13, fontweight='bold')
        ax.set_ylabel(feature2, fontsize=13, fontweight='bold')
        ax.set_title(f'LDA Decision Boundaries: {feature1} vs {feature2}', 
                    fontsize=15, fontweight='bold')
        
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print explanation
        print(f"\nLDA Decision Boundaries (2D visualization):")
        print(f"  • Trained LDA on ONLY {feature1} and {feature2}")
        print(f"  • Shows how well these 2 features separate the classes")
        print(f"  • Black lines = decision boundaries")
        print(f"  • Stars = class means")
        print(f"  • Colored regions = predicted class")
        
        # Show 2D model performance
        accuracy_2d = lda_2d.score(X_2d, y)
        print(f"\n  2D Model Accuracy: {accuracy_2d:.2%}")
    
    def summary(self):
        """
        Print detailed summary of LDA model information.
        
        Shows discriminant functions, class means, feature importance, and training fit.
        For test set evaluation, use the score() method separately.
        
        Returns:
        --------
        None
        
        Examples:
        ---------
        >>> lda = fit_lda(train, formula='species ~ .')
        >>> lda.summary()  # Show model details
        >>> train_fit = lda.score(train)['accuracy']  # Training fit
        >>> test_acc = lda.score(test)['accuracy']    # Test accuracy
        """
        self._check_fitted()
        
        # Training metrics
        y_pred_train = self.lda.predict(self.X)
        train_metrics = self._compute_classification_metrics(self.y, y_pred_train)
        train_fit = train_metrics['accuracy']
        conf_matrix_train = train_metrics['confusion_matrix']
        
        # Print summary
        print("\n" + "="*70)
        print("LINEAR DISCRIMINANT ANALYSIS MODEL SUMMARY")
        print("="*70)
        
        # Model information
        print("\nMODEL INFORMATION:")
        print("-" * 70)
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Number of classes: {len(self.classes)}")
        print(f"Classes: {self.classes}")
        print(f"Number of discriminant components: {len(self.classes) - 1}")
        print(f"Solver: {self.solver}")
        print(f"Training samples: {len(self.df)}")
        
        # Discriminant functions (using coef_)
        if hasattr(self.lda, 'coef_'):
            print("\nDISCRIMINANT FUNCTIONS (for classification):")
            print("-" * 70)
            
            # Number of discriminants
            n_discriminants = len(self.classes) - 1
            
            if n_discriminants == 1:
                # Binary classification - show decision function
                print(f"\nDecision Function:")
                
                # Build equation string using coef_
                terms = []
                for j, feature in enumerate(self.feature_names):
                    coef = self.lda.coef_[0, j]
                    if j == 0:
                        terms.append(f"{coef:.4f}×{feature}")
                    else:
                        sign = "+" if coef >= 0 else "-"
                        terms.append(f"{sign} {abs(coef):.4f}×{feature}")
                
                equation = " ".join(terms)
                
                # Add intercept
                if hasattr(self.lda, 'intercept_'):
                    intercept = self.lda.intercept_[0]
                    sign = "+" if intercept >= 0 else "-"
                    equation += f" {sign} {abs(intercept):.4f}"
                
                print(f"  f(x) = {equation}")
                
                # Show decision rule
                print(f"\nDecision Rule:")
                print(f"  If f(x) > 0: predict '{self.classes[1]}'")
                print(f"  If f(x) ≤ 0: predict '{self.classes[0]}'")
                print(f"  If f(x) = 0: on the decision boundary (maximum uncertainty)")
                
                print(f"\nNote: This decision function is used by predict().")
                print(f"      Larger |f(x)| means higher confidence in the prediction.")
            
            else:
                # Multi-class - show discriminant functions
                # For multi-class, we show the discriminant score for each class
                print(f"\nWith {len(self.classes)} classes, LDA creates discriminant scores for each class.")
                print(f"Classification: Predict the class with the highest discriminant score.\n")
                
                for i, class_name in enumerate(self.classes):
                    print(f"Discriminant for '{class_name}':")
                    
                    # Build equation string using coef_
                    terms = []
                    for j, feature in enumerate(self.feature_names):
                        coef = self.lda.coef_[i, j]
                        if j == 0:
                            terms.append(f"{coef:.4f}×{feature}")
                        else:
                            sign = "+" if coef >= 0 else "-"
                            terms.append(f"{sign} {abs(coef):.4f}×{feature}")
                    
                    equation = " ".join(terms)
                    
                    # Add intercept for this class
                    if hasattr(self.lda, 'intercept_'):
                        intercept = self.lda.intercept_[i]
                        sign = "+" if intercept >= 0 else "-"
                        equation += f" {sign} {abs(intercept):.4f}"
                    
                    print(f"  f_{class_name}(x) = {equation}\n")
                
                print("Decision Rule: Predict class with maximum f_class(x)")
        
        # Projection functions (using scalings_) - for transformation to LD space
        if hasattr(self.lda, 'scalings_'):
            print("\nPROJECTION FUNCTIONS (for dimensionality reduction):")
            print("-" * 70)
            
            n_components = min(self.lda.scalings_.shape[1], len(self.classes) - 1)
            
            print(f"These functions project data from {len(self.feature_names)}D to {n_components}D space.")
            print(f"Used by transform() method for visualization.\n")
            
            for i in range(n_components):
                print(f"Projection {i+1} (LD{i+1}):")
                
                # Build equation string using scalings_
                terms = []
                for j, feature in enumerate(self.feature_names):
                    coef = self.lda.scalings_[j, i]
                    if j == 0:
                        terms.append(f"{coef:.4f}×(centered_{feature})")
                    else:
                        sign = "+" if coef >= 0 else "-"
                        terms.append(f"{sign} {abs(coef):.4f}×(centered_{feature})")
                
                equation = " ".join(terms)
                print(f"  LD{i+1} = {equation}\n")
            
            print("Note: 'centered' means subtract the overall feature mean (see below).")
        
        # Overall feature means (used for centering in projections)
        if hasattr(self.lda, 'xbar_'):
            overall_means = self.lda.xbar_
            
            print("\nOVERALL FEATURE MEANS (for centering in projections):")
            print("-" * 70)
            for i, feature in enumerate(self.feature_names):
                print(f"  {feature}: {overall_means[i]:.4f}")
        
        # Class means in original space
        if hasattr(self.lda, 'means_'):
            print("\nCLASS MEANS (in original feature space):")
            print("-" * 70)
            means_df = pd.DataFrame(
                self.lda.means_,
                index=self.classes,
                columns=self.feature_names
            )
            print(means_df.to_string())
            # Add class centroids in LD space
            if hasattr(self.lda, 'scalings_'):
                print("\nCLASS CENTROIDS (in LD discriminant space):")
                print("-" * 70)
                
                # Transform class means to LD space
                class_means_LD = self.lda.transform(
                    pd.DataFrame(self.lda.means_, columns=self.feature_names)
                )
                
                n_components = class_means_LD.shape[1]
                
                # Create column names based on number of components
                ld_columns = [f'LD{i+1}' for i in range(n_components)]
                
                centroids_df = pd.DataFrame(
                    class_means_LD,
                    index=self.classes,
                    columns=ld_columns
                )
                print(centroids_df.to_string())
                
                print(f"\nNote: To classify a flower, calculate its (LD1, LD2) coordinates,")
                print(f"      then find which centroid is closest (minimum Euclidean distance).")

        # Explained variance ratio (if available)
        if hasattr(self.lda, 'explained_variance_ratio_'):
            print("\nEXPLAINED VARIANCE BY COMPONENT:")
            print("-" * 70)
            for i, var_ratio in enumerate(self.lda.explained_variance_ratio_):
                print(f"  LD{i+1}: {var_ratio*100:.2f}%")
            print(f"  Total: {sum(self.lda.explained_variance_ratio_)*100:.2f}%")
        
        # Feature importance
        if hasattr(self.lda, 'coef_'):
            print("\nFEATURE IMPORTANCE:")
            print("-" * 70)
            importance_df = self.get_feature_importance()
            print(importance_df.to_string(index=False))
        
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
        
        # Prior probabilities
        if hasattr(self.lda, 'priors_'):
            print("\nCLASS PRIOR PROBABILITIES:")
            print("-" * 70)
            for i, (class_name, prior) in enumerate(zip(self.classes, self.lda.priors_)):
                print(f"  {class_name}: {prior:.4f} ({prior*100:.2f}%)")
        
        print("\n" + "="*70)
        
        

def fit_lda(df, n_components=None, solver='svd', formula=None):
    """
    Fit a Linear Discriminant Analysis model.
    
    This is a convenience wrapper that creates an LDAModel.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and target
    n_components : int, optional
        Number of discriminant components to keep
    solver : str
        Solver to use: 'svd' (default), 'lsqr', or 'eigen'
    formula : str
        Formula for specifying target and features
    
    Returns:
    --------
    model : LDAModel
        Fitted LDA model
    
    Examples:
    ---------
    >>> from cuanalytics.datasets import load_iris_data
    >>> from cuanalytics.lda import fit_lda
    >>> df = load_iris_data()
    >>> lda = fit_lda(df, formula='species ~ .')
    >>> lda.visualize()
    >>> print(f"Accuracy: {lda.score(df)['accuracy']:.2%}")
    """
    if formula is None:
        raise ValueError("Must provide 'formula' for model specification")
    return LDAModel(df, n_components, solver, formula=formula)
