# cuanalytics/neuralnet.py
"""
Neural network models using scikit-learn MLP.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error, r2_score


class NeuralNetModel:
    """
    Feedforward neural network for classification or regression.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and target
    formula : str
        R-style formula for specifying the model
    hidden_layers : list[int] | tuple[int]
        Nodes per hidden layer (e.g., [3, 5, 2])
    activation : str
        Activation function for hidden layers
    solver : str
        Optimizer ('adam', 'lbfgs', 'sgd')
    alpha : float
        L2 regularization strength
    max_iter : int
        Maximum number of training iterations
    random_state : int | None
        Random seed for reproducibility
    """

    def __init__(
        self,
        df,
        formula,
        hidden_layers=None,
        activation='relu',
        solver='adam',
        alpha=0.0001,
        max_iter=1000,
        random_state=None,
    ):
        self.original_df = df
        self.formula = formula
        self.model_spec = None
        self.hidden_layers = hidden_layers if hidden_layers is not None else [5, 5, 5]
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state

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

        self.task_type = self._infer_task_type(self.y)
        self.classes = sorted(self.y.unique()) if self.task_type == 'classification' else None

        self._fit()
        self._print_fit_summary()

    def _infer_task_type(self, y):
        """Infer whether the target indicates classification or regression."""
        if not pd.api.types.is_numeric_dtype(y):
            return 'classification'
        if y.nunique() <= 10:
            return 'classification'
        return 'regression'

    def _fit(self):
        """Fit the neural network model."""
        hidden_layers = tuple(self.hidden_layers) if isinstance(self.hidden_layers, list) else self.hidden_layers
        if self.task_type == 'classification':
            self.model = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation=self.activation,
                solver=self.solver,
                alpha=self.alpha,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
        else:
            self.model = MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                activation=self.activation,
                solver=self.solver,
                alpha=self.alpha,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
        self.model.fit(self.X, self.y)

    def _check_fitted(self):
        """Check if the model has been fitted."""
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError(
                "Model has not been fitted yet. "
                "Create model with: model = fit_nn(df, formula='y ~ x1 + x2')"
            )

    def _print_fit_summary(self):
        """Print a brief summary after fitting."""
        print("\nNeural Network fitted successfully!")
        print(f"  Task type: {self.task_type}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Training samples: {len(self.df)}")
        print(f"  Hidden layers: {self.hidden_layers}")
        print(f"  Activation: {self.activation}")
        print(f"  Solver: {self.solver}")

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

    def predict(self, df):
        """
        Predict target values or classes for new data.
        """
        self._check_fitted()
        X = self._transform_data_with_formula(df)
        preds = self.model.predict(X)
        return pd.Series(preds, index=df.index, name=self.target)

    def predict_proba(self, df):
        """
        Predict class probabilities for new data (classification only).
        """
        self._check_fitted()
        if self.task_type != 'classification':
            raise ValueError("predict_proba is only available for classification models.")
        X = self._transform_data_with_formula(df)
        return self.model.predict_proba(X)

    def score(self, df):
        """
        Calculate metrics on a dataset (classification or regression).
        """
        self._check_fitted()
        X = self._transform_data_with_formula(df)
        y_true = df[self.target]
        y_pred = self.model.predict(X)

        if self.task_type == 'classification':
            report = self._compute_classification_metrics(y_true, y_pred)
            self._print_score_report(report)
            return report

        return {
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
        }

    def get_metrics(self):
        """
        Calculate metrics on training data.
        """
        self._check_fitted()
        y_pred = self.model.predict(self.X)

        metrics = {
            'model_type': 'neural_network',
            'task_type': self.task_type,
            'target': self.target,
            'n_samples': len(self.y),
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'hidden_layers': list(self.hidden_layers) if isinstance(self.hidden_layers, (list, tuple)) else self.hidden_layers,
            'activation': self.activation,
            'solver': self.solver,
        }

        if self.task_type == 'classification':
            metrics['classes'] = self.classes
            metrics['score'] = self._compute_classification_metrics(self.y, y_pred)
        else:
            metrics['score'] = {
                'r2': r2_score(self.y, y_pred),
                'rmse': np.sqrt(mean_squared_error(self.y, y_pred)),
                'mae': mean_absolute_error(self.y, y_pred),
            }

        return metrics

    def visualize(self, figsize=(12, 6)):
        """
        Visualize neural network structure with edge weights.
        """
        self._check_fitted()

        if self.task_type == 'classification' and hasattr(self.model, 'classes_'):
            if self.model.n_outputs_ == len(self.model.classes_):
                output_labels = list(self.model.classes_)
            else:
                output_labels = [self.model.classes_[1]]
        else:
            output_labels = [self.target]

        layer_sizes = [len(self.feature_names)] + list(self.model.hidden_layer_sizes) + [len(output_labels)]
        n_layers = len(layer_sizes)

        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')

        x_positions = np.linspace(0.1, 0.9, n_layers)
        node_positions = []
        for layer_idx, size in enumerate(layer_sizes):
            y_positions = np.linspace(0.1, 0.9, size)
            offset = (1 - (y_positions.max() - y_positions.min())) / 2 - y_positions.min()
            y_positions = y_positions + offset
            node_positions.append([(x_positions[layer_idx], y) for y in y_positions])

        for layer_idx, layer in enumerate(node_positions):
            for node_idx, (x, y) in enumerate(layer):
                ax.scatter(x, y, s=200, color='white', edgecolor='black', zorder=3)
                if layer_idx == 0 and node_idx < len(self.feature_names):
                    ax.text(x - 0.03, y, self.feature_names[node_idx], fontsize=8, ha='right', va='center')
                elif layer_idx == n_layers - 1 and node_idx < len(output_labels):
                    ax.text(x + 0.03, y, str(output_labels[node_idx]), fontsize=8, ha='left', va='center')

        max_edges_for_labels = 200
        edge_count = sum(w.size for w in self.model.coefs_)
        show_edge_labels = edge_count <= max_edges_for_labels

        for layer_idx, weights in enumerate(self.model.coefs_):
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    x0, y0 = node_positions[layer_idx][i]
                    x1, y1 = node_positions[layer_idx + 1][j]
                    weight = weights[i, j]
                    color = 'green' if weight >= 0 else 'red'
                    width = max(0.5, min(3.0, abs(weight)))
                    ax.plot([x0, x1], [y0, y1], color=color, linewidth=width, alpha=0.6, zorder=1)
                    if show_edge_labels:
                        mid_x = (x0 + x1) / 2
                        mid_y = (y0 + y1) / 2
                        dx = x1 - x0
                        dy = y1 - y0
                        length = max(np.hypot(dx, dy), 1e-6)
                        nx, ny = -dy / length, dx / length
                        sign = -1 if (i + j + layer_idx) % 2 == 0 else 1
                        offset = 0.015 * sign
                        ax.text(mid_x + nx * offset, mid_y + ny * offset,
                                f"{weight:.2f}", fontsize=6, ha='center', va='center')

        ax.set_title('Neural Network Structure', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def summary(self):
        """
        Print a summary of the neural network model.
        """
        self._check_fitted()

        print("\n" + "=" * 70)
        print("NEURAL NETWORK MODEL SUMMARY")
        print("=" * 70)
        print(f"Task type: {self.task_type}")
        print(f"Target: {self.target}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Hidden layers: {self.hidden_layers}")
        print(f"Activation: {self.activation}")
        print(f"Solver: {self.solver}")
        print(f"Training samples: {len(self.df)}")

        if self.task_type == 'classification':
            y_pred = self.model.predict(self.X)
            report = self._compute_classification_metrics(self.y, y_pred)
            print(f"\nTraining Accuracy: {report['accuracy']:.2%}")

            conf_df = pd.DataFrame(
                report['confusion_matrix'],
                index=[f"Actual {c}" for c in self.classes],
                columns=[f"Pred {c}" for c in self.classes]
            )
            print("\nTraining Confusion Matrix:")
            print(conf_df.to_string())
            print(f"\nKappa: {report['kappa']:.4f}")
        else:
            y_pred = self.model.predict(self.X)
            r2 = r2_score(self.y, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y, y_pred))
            mae = mean_absolute_error(self.y, y_pred)
            print(f"\nTraining R²: {r2:.4f}")
            print(f"Training RMSE: {rmse:.4f}")
            print(f"Training MAE: {mae:.4f}")

        print("\n" + "=" * 70)


def fit_nn(
    df,
    formula,
    hidden_layers=None,
    activation='relu',
    solver='adam',
    alpha=0.0001,
    max_iter=1000,
    random_state=None,
):
    """
    Fit a neural network model for classification or regression.

    Parameters:
    -----------
    df : pd.DataFrame
        Training data with features and target column
    formula : str
        R-style formula
    hidden_layers : list[int] | tuple[int]
        Nodes per hidden layer (e.g., [3, 5, 2])
    """
    if formula is None:
        raise ValueError("Must provide 'formula' for model specification")
    return NeuralNetModel(
        df,
        formula=formula,
        hidden_layers=hidden_layers,
        activation=activation,
        solver=solver,
        alpha=alpha,
        max_iter=max_iter,
        random_state=random_state,
    )
