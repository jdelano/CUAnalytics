# cuanalytics/clustering/kmeans.py
"""
K-Means clustering.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


class KMeansModel:
    """
    K-Means clustering model.
    """

    def __init__(self, df, formula, n_clusters=3, random_state=None, n_init='auto'):
        self.original_df = df
        self.formula = formula
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
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
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init
        )
        self.model.fit(self.X)
        self.labels_ = self.model.labels_

    def _check_fitted(self):
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError(
                "Model has not been fitted yet. "
                "Create model with: model = fit_kmeans(df, formula='x1 + x2')"
            )

    def _print_fit_summary(self):
        print("\nK-Means fitted successfully!")
        print(f"  Clusters: {self.n_clusters}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Samples: {len(self.X)}")

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

    def predict(self, df):
        self._check_fitted()
        X = self._transform_data_with_formula(df)
        labels = self.model.predict(X)
        return pd.Series(labels, index=df.index, name='cluster')

    def score(self, df):
        self._check_fitted()
        X = self._transform_data_with_formula(df)
        labels = self.model.predict(X)
        metrics = {'inertia': float(self.model.inertia_)}
        if self.n_clusters > 1 and len(X) > self.n_clusters:
            metrics['silhouette'] = float(silhouette_score(X, labels))
        return metrics

    def get_metrics(self):
        self._check_fitted()
        metrics = {
            'model_type': 'kmeans',
            'n_clusters': self.n_clusters,
            'n_samples': len(self.X),
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'inertia': float(self.model.inertia_),
            'cluster_counts': {
                int(cluster): int(count)
                for cluster, count in pd.Series(self.labels_).value_counts().sort_index().items()
            },
        }
        if self.n_clusters > 1 and len(self.X) > self.n_clusters:
            metrics['silhouette'] = float(silhouette_score(self.X, self.labels_))
        return metrics

    def _extract_condensed_positive_rule(self, tree, feature_names):
        tree_ = tree.tree_

        def recurse(node, path, candidates):
            feature_idx = tree_.feature[node]
            if feature_idx != -2:
                threshold = tree_.threshold[node]
                feature = feature_names[feature_idx]
                left_path = path + [f"{feature} <= {threshold:.3f}"]
                right_path = path + [f"{feature} > {threshold:.3f}"]
                recurse(tree_.children_left[node], left_path, candidates)
                recurse(tree_.children_right[node], right_path, candidates)
                return

            counts = tree_.value[node][0]
            predicted = int(np.argmax(counts))
            positive_support = int(counts[1]) if len(counts) > 1 else 0
            if predicted == 1 and positive_support > 0:
                candidates.append((positive_support, path))

        candidates = []
        recurse(0, [], candidates)
        if not candidates:
            return "No concise rule found at this tree depth."

        best_path = max(candidates, key=lambda x: x[0])[1]
        if not best_path:
            return "All observations (root rule)."
        return " AND ".join(best_path)

    def describe_clusters(self, max_depth=3, criterion='entropy', random_state=42):
        """
        Return original training data with cluster labels and condensed rule descriptions.

        For each cluster, this fits a one-vs-rest decision tree on the model features and
        extracts a concise positive rule. Each row receives the rule for its assigned cluster.
        """
        self._check_fitted()

        allowed_criteria = {'entropy', 'gini'}
        if criterion not in allowed_criteria:
            raise ValueError("criterion must be 'entropy' or 'gini'.")

        unique_clusters = np.sort(np.unique(self.labels_))
        cluster_rules = {}
        X_values = self.X.values

        for cluster in unique_clusters:
            y_binary = (self.labels_ == cluster).astype(int)
            tree = DecisionTreeClassifier(
                max_depth=max_depth,
                criterion=criterion,
                random_state=random_state,
            )
            tree.fit(X_values, y_binary)
            cluster_rules[int(cluster)] = self._extract_condensed_positive_rule(tree, self.feature_names)

        labeled = self.original_df.copy()
        labels_series = pd.Series(self.labels_, index=self.X.index, name='cluster').astype(int)
        labeled['cluster'] = labels_series.reindex(labeled.index)
        labeled['cluster_rule'] = labeled['cluster'].map(cluster_rules)
        return labeled

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
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title('K-Means Clusters')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def summary(self):
        self._check_fitted()
        metrics = self.get_metrics()
        print("\n" + "=" * 70)
        print("K-MEANS SUMMARY")
        print("=" * 70)
        print(f"Clusters: {self.n_clusters}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Samples: {len(self.X)}")
        print(f"Inertia: {metrics['inertia']:.4f}")
        if 'silhouette' in metrics:
            print(f"Silhouette: {metrics['silhouette']:.4f}")
        print("\nCluster Counts:")
        print(metrics['cluster_counts'])
        print("\n" + "=" * 70)


def fit_kmeans(df, formula, n_clusters=3, random_state=None, n_init='auto'):
    if formula is None:
        raise ValueError("Must provide 'formula' for model specification")
    return KMeansModel(
        df,
        formula=formula,
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )
