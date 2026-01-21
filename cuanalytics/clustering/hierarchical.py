# cuanalytics/clustering/hierarchical.py
"""
Hierarchical clustering.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


class HierarchicalClusteringModel:
    """
    Hierarchical clustering model.
    """

    def __init__(self, df, formula, n_clusters=3, linkage='ward'):
        self.original_df = df
        self.formula = formula
        self.n_clusters = n_clusters
        self.linkage = linkage
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

        rhs = formula
        if '~' in formula:
            rhs = formula.split('~', 1)[1].strip()

        df_rhs = df.copy()
        model_matrices = model_matrix(rhs, df_rhs, output='pandas')

        if hasattr(model_matrices, 'rhs'):
            X = model_matrices.rhs
        else:
            X = model_matrices

        self.model_spec = getattr(model_matrices, 'model_spec', None)

        if 'Intercept' in X.columns:
            X = X.drop('Intercept', axis=1)

        self.X = X
        self.feature_names = list(self.X.columns)

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
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)
        self.labels_ = self.model.fit_predict(self.X)

    def _check_fitted(self):
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError(
                "Model has not been fitted yet. "
                "Create model with: model = fit_hierarchical(df, formula='x1 + x2')"
            )

    def _print_fit_summary(self):
        print("\nHierarchical clustering fitted successfully!")
        print(f"  Clusters: {self.n_clusters}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Samples: {len(self.X)}")
        print(f"  Linkage: {self.linkage}")

    def _transform_data_with_formula(self, df):
        if getattr(self, 'model_spec', None) is None:
            raise RuntimeError("Formula metadata missing; cannot transform new data.")

        rhs_spec = getattr(self.model_spec, 'rhs', self.model_spec)
        model_matrices = rhs_spec.get_model_matrix(df, output='pandas')

        if hasattr(model_matrices, 'rhs'):
            X_new = model_matrices.rhs
        else:
            X_new = model_matrices

        if 'Intercept' in X_new.columns:
            X_new = X_new.drop('Intercept', axis=1)

        return X_new

    def predict(self, df=None):
        self._check_fitted()
        if df is not None:
            raise ValueError("Hierarchical clustering does not support predicting new data.")
        return pd.Series(self.labels_, index=self.X.index, name='cluster')

    def score(self, df=None):
        self._check_fitted()
        X = self.X if df is None else self._transform_data_with_formula(df)
        labels = self.labels_ if df is None else AgglomerativeClustering(
            n_clusters=self.n_clusters, linkage=self.linkage
        ).fit_predict(X)
        metrics = {}
        if self.n_clusters > 1 and len(X) > self.n_clusters:
            metrics['silhouette'] = float(silhouette_score(X, labels))
        return metrics

    def get_metrics(self):
        self._check_fitted()
        metrics = {
            'model_type': 'hierarchical',
            'n_clusters': self.n_clusters,
            'n_samples': len(self.X),
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'linkage': self.linkage,
            'cluster_counts': dict(pd.Series(self.labels_).value_counts().sort_index()),
        }
        if self.n_clusters > 1 and len(self.X) > self.n_clusters:
            metrics['silhouette'] = float(silhouette_score(self.X, self.labels_))
        return metrics

    def visualize(self, figsize=(8, 6)):
        self._check_fitted()
        if len(self.feature_names) > 2:
            pca = PCA(n_components=2)
            X_plot = pca.fit_transform(self.X)
            xlabel = 'PC1'
            ylabel = 'PC2'
        else:
            X_plot = self.X.values
            xlabel, ylabel = self.feature_names

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(X_plot[:, 0], X_plot[:, 1], c=self.labels_, cmap='tab10', alpha=0.7, edgecolors='black')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title('Hierarchical Clusters')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def summary(self):
        self._check_fitted()
        metrics = self.get_metrics()
        print("\n" + "=" * 70)
        print("HIERARCHICAL CLUSTERING SUMMARY")
        print("=" * 70)
        print(f"Clusters: {self.n_clusters}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Samples: {len(self.X)}")
        print(f"Linkage: {self.linkage}")
        if 'silhouette' in metrics:
            print(f"Silhouette: {metrics['silhouette']:.4f}")
        print("\nCluster Counts:")
        print(metrics['cluster_counts'])
        print("\n" + "=" * 70)


def fit_hierarchical(df, formula, n_clusters=3, linkage='ward'):
    if formula is None:
        raise ValueError("Must provide 'formula' for model specification")
    return HierarchicalClusteringModel(
        df,
        formula=formula,
        n_clusters=n_clusters,
        linkage=linkage,
    )
