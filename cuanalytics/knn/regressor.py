# cuanalytics/knn/regressor.py
"""
k-Nearest Neighbors regressor.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class KNNRegressorModel:
    """
    K-Nearest Neighbors model for regression.
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
            raise ValueError("Formula must include a target (e.g., 'y ~ x1 + x2').")

        lhs, rhs = formula.split('~', 1)
        lhs = lhs.strip()
        rhs = rhs.strip()

        if lhs and lhs not in df.columns and lhs.isidentifier():
            raise ValueError(f"Target '{lhs}' not found in data")

        y = df[lhs]
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError("Target must be numeric for regression.")

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
        self.model = KNeighborsRegressor(n_neighbors=self.k, weights=self.weights, metric=self.metric)
        self.model.fit(self.X, self.y)

    def _check_fitted(self):
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError(
                "Model has not been fitted yet. "
                "Create model with: model = fit_knn_regressor(df, formula='y ~ x1 + x2')"
            )

    def _print_fit_summary(self):
        print("\nKNN Regressor fitted successfully!")
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
        return {
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
        }

    def get_metrics(self):
        self._check_fitted()
        y_pred = self.model.predict(self.X)
        return {
            'model_type': 'knn_regressor',
            'target': self.target,
            'n_samples': len(self.y),
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'k': self.k,
            'weights': self.weights,
            'metric': self.metric,
            'score': {
                'r2': r2_score(self.y, y_pred),
                'rmse': np.sqrt(mean_squared_error(self.y, y_pred)),
                'mae': mean_absolute_error(self.y, y_pred),
            },
        }

    def visualize(self, figsize=(8, 6)):
        self._check_fitted()
        y_pred = self.model.predict(self.X)
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(self.y, y_pred, alpha=0.6, edgecolors='black')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('KNN Regression: Predicted vs Actual')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def summary(self):
        self._check_fitted()
        report = self.score(self.df)
        print("\n" + "=" * 70)
        print("KNN REGRESSOR SUMMARY")
        print("=" * 70)
        print(f"Target: {self.target}")
        print(f"Features: {len(self.feature_names)}")
        print(f"k: {self.k}")
        print(f"Weights: {self.weights}")
        print(f"Metric: {self.metric}")
        print(f"Training samples: {len(self.df)}")
        print("\nTRAINING FIT:")
        print("-" * 70)
        print(f"Training R²: {report['r2']:.4f}")
        print(f"Training RMSE: {report['rmse']:.4f}")
        print(f"Training MAE: {report['mae']:.4f}")
        print("\n" + "=" * 70)


def fit_knn_regressor(df, formula, k=5, weights='uniform', metric='minkowski'):
    if formula is None:
        raise ValueError("Must provide 'formula' for model specification")
    return KNNRegressorModel(df, formula=formula, k=k, weights=weights, metric=metric)
