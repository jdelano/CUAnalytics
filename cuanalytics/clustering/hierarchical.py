# cuanalytics/clustering/hierarchical.py
"""
Hierarchical clustering.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
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
            'cluster_counts': {
                int(cluster): int(count)
                for cluster, count in pd.Series(self.labels_).value_counts().sort_index().items()
            },
        }
        if self.n_clusters > 1 and len(self.X) > self.n_clusters:
            metrics['silhouette'] = float(silhouette_score(self.X, self.labels_))
        return metrics

    def visualize(
        self,
        figsize=None,
        leaf_rotation=90,
        leaf_font_size=8,
        cutoff=None,
        truncate_mode='lastp',
    ):
        self._check_fitted()
        Z = linkage(self.X, method=self.linkage)
        n_samples = len(self.X)
        valid_truncate_modes = {'lastp', 'level', None}
        if truncate_mode not in valid_truncate_modes:
            raise ValueError("truncate_mode must be one of: 'lastp', 'level', or None.")
        if cutoff is not None:
            if not isinstance(cutoff, int):
                raise ValueError("cutoff must be an integer.")
            min_cutoff = 0 if truncate_mode == 'level' else 2
            if cutoff < min_cutoff:
                raise ValueError(f"cutoff must be >= {min_cutoff} for truncate_mode='{truncate_mode}'.")
            if cutoff > n_samples:
                raise ValueError("cutoff cannot exceed the number of samples.")

        if figsize is None:
            width = max(10, min(40, n_samples * 0.18 if cutoff is None else cutoff * 0.6))
            figsize = (width, 6)

        fig, ax = plt.subplots(figsize=figsize)
        dendrogram_kwargs = {
            'ax': ax,
            'leaf_rotation': leaf_rotation,
            'leaf_font_size': leaf_font_size,
        }
        if cutoff is None:
            dendrogram_kwargs['labels'] = self.X.index.astype(str).tolist()
            title = 'Hierarchical Clustering Dendrogram'
        else:
            mode = 'lastp' if truncate_mode is None else truncate_mode
            if mode == 'level' and cutoff == 0:
                # Show only the root split as two top-level buckets.
                dendrogram_kwargs['truncate_mode'] = 'lastp'
                dendrogram_kwargs['p'] = 2
                dendrogram_kwargs['show_contracted'] = True
                title = 'Hierarchical Clustering Dendrogram (level=0, root split)'
            else:
                dendrogram_kwargs['truncate_mode'] = mode
                dendrogram_kwargs['p'] = cutoff
                dendrogram_kwargs['show_contracted'] = True
                title = f'Hierarchical Clustering Dendrogram ({mode}, cutoff={cutoff})'

        dendrogram(Z, **dendrogram_kwargs)
        ax.set_title(title)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Distance')
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def visualize_all_features(self, figsize=(8, 6)):
        self._check_fitted()
        if len(self.feature_names) > 2:
            pca = PCA(n_components=2)
            X_plot = pca.fit_transform(self.X)
            xlabel = 'PC1'
            ylabel = 'PC2'
        else:
            X_plot = self.X.values
            xlabel, ylabel = self.feature_names

        unique_labels = np.sort(np.unique(self.labels_))
        base_colors = [
            '#ff7f00', '#377eb8', '#4daf4a', '#984ea3', '#b8860b',
            '#a65628', '#f781bf', '#999999', '#66c2a5', '#1b9e77',
        ]
        marker_cycle = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>', '*']
        colors = {label: base_colors[i % len(base_colors)] for i, label in enumerate(unique_labels)}
        markers = {label: marker_cycle[i % len(marker_cycle)] for i, label in enumerate(unique_labels)}

        fig, ax = plt.subplots(figsize=figsize)
        for label in unique_labels:
            mask = self.labels_ == label
            ax.scatter(
                X_plot[mask, 0],
                X_plot[mask, 1],
                color=colors[label],
                marker=markers[label],
                alpha=0.9,
                edgecolors='white',
                linewidth=0.7,
                label=f'Cluster {int(label)}',
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title('Hierarchical Clusters (All Features)')
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
