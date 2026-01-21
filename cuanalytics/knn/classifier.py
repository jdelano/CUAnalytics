# cuanalytics/knn/classifier.py
"""
k-Nearest Neighbors classifier.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


class KNNClassifierModel:
    """
    K-Nearest Neighbors model for classification.
    """

    def __init__(self, df, formula, k=5, weights='uniform', metric='minkowski'):
        self.original_df = df
        self.formula = formula
        self.k = k
        self.weights = weights
        self.metric = metric
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
            raise ValueError("All features must be numeric. Encode categorical variables first.")

        self._fit()
        self._print_fit_summary()

    def _fit(self):
        self.model = KNeighborsClassifier(n_neighbors=self.k, weights=self.weights, metric=self.metric)
        self.model.fit(self.X, self.y)

    def _check_fitted(self):
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError(
                "Model has not been fitted yet. "
                "Create model with: model = fit_knn_classifier(df, formula='class ~ x1 + x2')"
            )

    def _print_fit_summary(self):
        print("\nKNN Classifier fitted successfully!")
        print(f"  Classes: {self.classes}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Training samples: {len(self.df)}")
        print(f"  k: {self.k}")

    def _transform_data_with_formula(self, df):
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

    def _compute_classification_metrics(self, y_true, y_pred):
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

    def predict(self, df):
        self._check_fitted()
        X = self._transform_data_with_formula(df)
        preds = self.model.predict(X)
        return pd.Series(preds, index=df.index, name=self.target)

    def score(self, df):
        self._check_fitted()
        X = self._transform_data_with_formula(df)
        y_true = df[self.target]
        y_pred = self.model.predict(X)
        report = self._compute_classification_metrics(y_true, y_pred)
        self._print_score_report(report)
        return report

    def get_metrics(self):
        self._check_fitted()
        y_pred = self.model.predict(self.X)
        report = self._compute_classification_metrics(self.y, y_pred)
        return {
            'model_type': 'knn_classifier',
            'target': self.target,
            'n_samples': len(self.y),
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'classes': self.classes,
            'k': self.k,
            'weights': self.weights,
            'metric': self.metric,
            'score': report,
        }

    def visualize(self, figsize=(8, 6)):
        self._check_fitted()
        if len(self.feature_names) != 2:
            raise ValueError("visualize requires exactly 2 numeric features.")

        X = self.X.values
        y = self.y.values

        fig, ax = plt.subplots(figsize=figsize)
        for label in self.classes:
            mask = y == label
            ax.scatter(X[mask, 0], X[mask, 1], label=str(label), alpha=0.7, edgecolors='black')
        ax.set_xlabel(self.feature_names[0])
        ax.set_ylabel(self.feature_names[1])
        ax.set_title('KNN Classification (2D)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def summary(self):
        self._check_fitted()
        y_pred = self.model.predict(self.X)
        report = self._compute_classification_metrics(self.y, y_pred)

        print("\n" + "=" * 70)
        print("KNN CLASSIFIER SUMMARY")
        print("=" * 70)
        print(f"Target: {self.target}")
        print(f"Classes: {self.classes}")
        print(f"Features: {len(self.feature_names)}")
        print(f"k: {self.k}")
        print(f"Weights: {self.weights}")
        print(f"Metric: {self.metric}")
        print(f"Training samples: {len(self.df)}")

        print("\nTRAINING FIT:")
        print("-" * 70)
        print(f"Training Accuracy: {report['accuracy']:.2%}")

        conf_df = pd.DataFrame(
            report['confusion_matrix'],
            index=[f"Actual {c}" for c in self.classes],
            columns=[f"Pred {c}" for c in self.classes]
        )
        print("\nTraining Confusion Matrix:")
        print(conf_df.to_string())
        print(f"\nKappa: {report['kappa']:.4f}")

        print("\n" + "=" * 70)


def fit_knn_classifier(df, formula, k=5, weights='uniform', metric='minkowski'):
    if formula is None:
        raise ValueError("Must provide 'formula' for model specification")
    return KNNClassifierModel(df, formula=formula, k=k, weights=weights, metric=metric)
