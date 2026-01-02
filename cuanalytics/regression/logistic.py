# cuanalytics/regression/logistic.py
"""
Logistic regression implementation for classification tasks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


class LogisticRegressionModel:
    """
    Logistic Regression model for classification outcomes.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and target
    formula : str
        R-style formula for specifying the model
    C : float
        Inverse regularization strength
    penalty : str
        Regularization type ('l2', 'l1', or 'elasticnet')
    solver : str
        Solver to use for optimization
        - 'lbfgs': good default, supports 'l2' and multiclass
        - 'liblinear': supports 'l1'/'l2', best for small datasets (binary/ovr)
        - 'saga': supports 'l1', 'l2', and 'elasticnet' with large datasets
    max_iter : int
        Maximum number of iterations
    """

    def __init__(self, df, formula, C=1.0, penalty='l2', solver='lbfgs', max_iter=1000):
        self.original_df = df
        self.formula = formula
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
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

        self.feature_names = list(self.X.columns)
        self.feature_names_original = [col for col in df.columns if col != self.target]
        self.classes = sorted(self.y.unique())

        used_vars = set()
        if self.model_spec is not None:
            rhs_spec = getattr(self.model_spec, 'rhs', self.model_spec)
            used_vars = {var for var in getattr(rhs_spec, 'variables', set())
                         if var in df_rhs.columns}
        if not used_vars:
            used_vars = set(df_rhs.columns)
        non_numeric = [col for col in used_vars
                       if not pd.api.types.is_numeric_dtype(df_rhs[col])]
        if non_numeric:
            raise ValueError(
                "All features must be numeric. Encode categorical variables first."
            )

        self._fit()
        self._print_fit_summary()

    def _fit(self):
        """Fit the logistic regression model."""
        kwargs = {
            'C': self.C,
            'solver': self.solver,
            'max_iter': self.max_iter,
        }
        if self.penalty != 'l2':
            kwargs['penalty'] = self.penalty
        self.model = LogisticRegression(**kwargs)
        self.model.fit(self.X, self.y)

    def _check_fitted(self):
        """Check if the model has been fitted."""
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError(
                "Model has not been fitted yet. "
                "Create model with: model = fit_logit(df, formula='class ~ x1 + x2')"
            )

    def _print_fit_summary(self):
        """Print a brief summary after fitting."""
        print("\nLogistic Regression fitted successfully!")
        print(f"  Classes: {self.classes}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Training samples: {len(self.df)}")
        print(f"  C parameter: {self.C}")
        print(f"  Solver: {self.solver}")

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
        """
        self._check_fitted()
        X = self._transform_data_with_formula(df)
        predictions = self.model.predict(X)
        return pd.Series(predictions, index=df.index, name=self.target)

    def predict_proba(self, df):
        """
        Predict class probabilities for new data.
        """
        self._check_fitted()
        X = self._transform_data_with_formula(df)
        return self.model.predict_proba(X)

    def score(self, df):
        """
        Calculate and print classification metrics on a dataset.
        """
        self._check_fitted()
        X = self._transform_data_with_formula(df)
        y_true = df[self.target]

        y_pred = self.model.predict(X)
        report = self._compute_classification_metrics(y_true, y_pred)
        self._print_score_report(report)
        return report

    def get_coefficients(self):
        """
        Get a coefficient table with z-scores and p-values.
        """
        self._check_fitted()
        return self._build_coefficient_table()

    def get_metrics(self):
        """
        Calculate classification metrics on training data.
        """
        self._check_fitted()

        y_pred = self.model.predict(self.X)
        y_true = self.y

        metrics = {
            'model_type': 'logistic_regression',
            'target': self.target,
            'n_samples': len(self.y),
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'classes': self.classes,
            'score': self._compute_classification_metrics(y_true, y_pred),
            'coefficients': self.get_coefficients(),
        }

        return metrics

    def visualize(self, figsize=(10, 6)):
        """
        Visualize logistic regression coefficients.
        """
        self._check_fitted()

        coefs = self.model.coef_
        if coefs.shape[0] > 1:
            values = np.abs(coefs).mean(axis=0)
            title = "Logistic Regression Feature Importance (avg |coef|)"
        else:
            values = coefs[0]
            title = "Logistic Regression Coefficients"

        order = np.argsort(np.abs(values))
        features = [self.feature_names[i] for i in order]
        values_sorted = values[order]
        colors = ['red' if v < 0 else 'green' for v in values_sorted]

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(range(len(features)), values_sorted, color=colors, edgecolor='black')
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Coefficient' if coefs.shape[0] == 1 else 'Avg |Coefficient|')
        ax.set_title(title)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def visualize_all_features(self, figsize=None, cols=3):
        """
        Visualize sigmoid curves for each feature with predicted probabilities.

        For binary classification, plots each feature against the model's
        predicted probability for the positive class.
        """
        self._check_fitted()

        if len(self.classes) != 2:
            raise ValueError("visualize_all_features is only supported for binary classification.")

        n_features = len(self.feature_names)
        rows = int(np.ceil(n_features / cols))
        if figsize is None:
            figsize = (5 * cols, 4 * rows)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

        positive_class = self.classes[1]
        class_index = list(self.model.classes_).index(positive_class)
        base_data = self.X.mean()

        for i, feature in enumerate(self.feature_names):
            ax = axes[i]

            feature_values = np.linspace(self.X[feature].min(), self.X[feature].max(), 200)
            X_pred = pd.DataFrame({col: base_data[col] for col in self.feature_names}, index=range(200))
            X_pred[feature] = feature_values
            probs_line = self.model.predict_proba(X_pred)[:, class_index]

            probs_points = self.model.predict_proba(self.X)[:, class_index]
            ax.scatter(self.X[feature], probs_points, alpha=0.6, s=30,
                       edgecolors='black', linewidth=0.5)
            ax.plot(feature_values, probs_line, 'r-', linewidth=2)

            ax.set_xlabel(feature, fontsize=10, fontweight='bold')
            ax.set_ylabel(f"P({positive_class})", fontsize=10, fontweight='bold')
            ax.set_title(f'{feature} vs P({positive_class})', fontsize=11)
            ax.grid(True, alpha=0.3)

        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle('Logistic Regression Sigmoid Curves', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

    def visualize_features(self, feature1, feature2, figsize=(10, 8)):
        """
        Visualize logistic regression decision boundary for two features.
        """
        self._check_fitted()

        if feature1 not in self.feature_names_original or feature2 not in self.feature_names_original:
            raise ValueError(
                f"Features must be from: {self.feature_names_original}\n"
                f"You provided: {feature1}, {feature2}"
            )

        X_2d = self.original_df[[feature1, feature2]].values
        y = self.original_df[self.target].values

        kwargs = {
            'C': self.C,
            'solver': self.solver,
            'max_iter': self.max_iter,
        }
        if self.penalty != 'l2':
            kwargs['penalty'] = self.penalty
        model_2d = LogisticRegression(**kwargs)
        model_2d.fit(X_2d, y)

        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        h = 0.02
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        class_to_num = {cls: i for i, cls in enumerate(self.classes)}
        Z = np.array([class_to_num[z] for z in Z]).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.classes)))
        cmap = plt.cm.Set1

        ax.contourf(xx, yy, Z, alpha=0.2, cmap=cmap)
        ax.contour(xx, yy, Z, colors='black', linewidths=2,
                   levels=np.arange(0.5, len(self.classes), 1))

        for i, class_name in enumerate(self.classes):
            mask = y == class_name
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=[colors[i]], label=class_name,
                       alpha=0.7, s=80, edgecolors='black', linewidth=1.5)

        ax.set_xlabel(feature1, fontsize=12, fontweight='bold')
        ax.set_ylabel(feature2, fontsize=12, fontweight='bold')
        ax.set_title(f'Logistic Regression Boundary: {feature1} vs {feature2}',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def summary(self):
        """
        Print detailed summary of logistic regression model.
        """
        self._check_fitted()

        y_pred = self.model.predict(self.X)
        train_metrics = self._compute_classification_metrics(self.y, y_pred)

        print("\n" + "=" * 70)
        print("LOGISTIC REGRESSION MODEL SUMMARY")
        print("=" * 70)

        print("\nMODEL INFORMATION:")
        print("-" * 70)
        print(f"Target: {self.target}")
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Classes: {self.classes}")
        print(f"Training samples: {len(self.df)}")
        print(f"C parameter: {self.C}")
        print(f"Solver: {self.solver}")

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

        print("\nCOEFFICIENTS:")
        print("-" * 70)
        coef_table = self.get_coefficients()
        print(coef_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(
            "\nNote: z-scores and p-values use a normal approximation;\n"
            "      multiclass values are computed one-vs-rest."
        )

        print("\n" + "=" * 70)

    def _build_coefficient_table(self):
        """Build coefficient table with z-scores and p-values."""
        from scipy import stats

        X = self.X.values
        n = X.shape[0]
        X_design = np.column_stack([np.ones(n), X])
        probs = self.model.predict_proba(self.X)
        coefs = self.model.coef_
        intercepts = self.model.intercept_
        rows = []

        if coefs.shape[0] == 1 and len(self.classes) == 2:
            class_label = self.classes[1]
            p = probs[:, 1]
            W = p * (1 - p)
            try:
                cov = np.linalg.inv(X_design.T @ (X_design * W[:, None]))
                std_err = np.sqrt(np.diag(cov))
            except np.linalg.LinAlgError:
                std_err = np.full(X_design.shape[1], np.nan)
            coef_all = np.concatenate([[intercepts[0]], coefs[0]])
            z_vals = coef_all / std_err
            p_vals = 2 * (1 - stats.norm.cdf(np.abs(z_vals)))
            for feature_name, coef_val, z_val, p_val in zip(
                ['(Intercept)'] + self.feature_names,
                coef_all,
                z_vals,
                p_vals
            ):
                rows.append({
                    'class': class_label,
                    'feature': feature_name,
                    'coefficient': coef_val,
                    'z_score': z_val,
                    'p_value': p_val,
                    'significance': self._get_significance_stars(p_val),
                })
        else:
            for class_idx, class_label in enumerate(self.classes):
                p = probs[:, class_idx]
                W = p * (1 - p)
                try:
                    cov = np.linalg.inv(X_design.T @ (X_design * W[:, None]))
                    std_err = np.sqrt(np.diag(cov))
                except np.linalg.LinAlgError:
                    std_err = np.full(X_design.shape[1], np.nan)
                coef_all = np.concatenate([[intercepts[class_idx]], coefs[class_idx]])
                z_vals = coef_all / std_err
                p_vals = 2 * (1 - stats.norm.cdf(np.abs(z_vals)))
                for feature_name, coef_val, z_val, p_val in zip(
                    ['(Intercept)'] + self.feature_names,
                    coef_all,
                    z_vals,
                    p_vals
                ):
                    rows.append({
                        'class': class_label,
                        'feature': feature_name,
                        'coefficient': coef_val,
                        'z_score': z_val,
                        'p_value': p_val,
                        'significance': self._get_significance_stars(p_val),
                    })

        return pd.DataFrame(rows)

    def _get_significance_stars(self, p_value):
        """Return significance stars based on p-value."""
        if p_value < 0.001:
            return '***'
        if p_value < 0.01:
            return '**'
        if p_value < 0.05:
            return '*'
        if p_value < 0.1:
            return '.'
        return ''


def fit_logit(df, formula, C=1.0, penalty='l2', solver='lbfgs', max_iter=1000):
    """
    Fit a logistic regression model for classification.

    Parameters:
    -----------
    df : pd.DataFrame
        Training data with features and target column
    formula : str
        R-style formula
    C : float
        Inverse regularization strength
    penalty : str
        Regularization type
    solver : str
        Solver to use
    max_iter : int
        Maximum number of iterations
    """
    if formula is None:
        raise ValueError("Must provide 'formula' for model specification")
    return LogisticRegressionModel(
        df,
        formula=formula,
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
    )
