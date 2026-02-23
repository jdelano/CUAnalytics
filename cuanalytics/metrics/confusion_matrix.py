"""Confusion matrix utilities for classification models."""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


class ConfusionMatrix:
    """Represent a confusion matrix and derived classification metrics."""

    def __init__(self, y_true, y_pred, labels=None, positive_label=None):
        if labels is None:
            labels = list(pd.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)])))
        self.labels = list(labels)
        if not self.labels:
            raise ValueError("labels cannot be empty")

        self.matrix = confusion_matrix(y_true, y_pred, labels=self.labels)
        self.total = int(self.matrix.sum())

        if positive_label is None and len(self.labels) == 2:
            positive_label = self.labels[1]
        if positive_label is not None and positive_label not in self.labels:
            raise ValueError(f"positive_label '{positive_label}' not found in labels")
        self.positive_label = positive_label

    @classmethod
    def from_matrix(cls, matrix, labels, positive_label=None):
        """Create a ConfusionMatrix instance from an existing matrix."""
        matrix = np.asarray(matrix)
        labels = list(labels)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be a square 2D array")
        if matrix.shape[0] != len(labels):
            raise ValueError(
                f"labels length must match matrix size ({matrix.shape[0]}), got {len(labels)}"
            )

        obj = cls.__new__(cls)
        obj.labels = labels
        obj.matrix = matrix
        obj.total = int(matrix.sum())
        if positive_label is None and len(labels) == 2:
            positive_label = labels[1]
        if positive_label is not None and positive_label not in labels:
            raise ValueError(f"positive_label '{positive_label}' not found in labels")
        obj.positive_label = positive_label
        return obj

    def to_dataframe(self, display='inverted'):
        """Return a labeled confusion matrix table.

        display='inverted' (default): textbook binary orientation with
        rows=predicted, cols=actual.
        display='normal': rows=actual, cols=predicted.
        """
        if display not in {'normal', 'inverted'}:
            raise ValueError("display must be 'normal' or 'inverted'")

        ordered_matrix, ordered_labels = self._matrix_for_display(display)

        if display == 'normal':
            return pd.DataFrame(
                ordered_matrix,
                index=[f"Actual {label}" for label in ordered_labels],
                columns=[f"Pred {label}" for label in ordered_labels],
            )

        return pd.DataFrame(
            ordered_matrix.T,
            index=[f"Pred {label}" for label in ordered_labels],
            columns=[f"Actual {label}" for label in ordered_labels],
        )

    def summary(self, display='inverted'):
        """Print a readable confusion matrix report and return the table."""
        metrics = self.get_metrics()
        conf_df = self.to_dataframe(display=display)
        print("\nConfusion Matrix and Statistics")
        print("=" * 60)
        print("\nConfusion Matrix:")
        print(conf_df.to_string())

        overall = {
            'Accuracy': metrics['accuracy'],
            'Kappa': metrics['kappa'],
            'Macro Precision': metrics['macro_precision'],
            'Macro Recall': metrics['macro_recall'],
            'Macro Specificity': metrics['macro_specificity'],
            'Macro F1': metrics['macro_f1'],
        }

        if len(self.labels) == 2:
            overall.update({
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'Sensitivity': metrics['sensitivity'],
                'Specificity': metrics['specificity'],
                'FPR': metrics['fpr'],
                'FNR': metrics['fnr'],
                'F-measure': metrics['f_measure'],
            })

        print("\nOverall Statistics:")
        print(pd.Series(overall).to_string(float_format=lambda x: f"{x:.4f}"))

        if len(self.labels) == 2:
            counts = metrics['binary_counts']
            print(f"\nBinary Counts (Positive = '{metrics['positive_label']}'):")
            print(f"TP: {counts['tp']}  FP: {counts['fp']}  FN: {counts['fn']}  TN: {counts['tn']}")

        per_class_df = pd.DataFrame(metrics['per_class']).T
        print("\nStatistics by Class:")
        print(per_class_df.to_string(float_format=lambda x: f"{x:.4f}"))

        return conf_df

    def get_metrics(self):
        """Return confusion-matrix derived metrics as a dictionary."""
        cm = self.matrix
        total = self.total
        accuracy = np.trace(cm) / total if total else 0.0

        row_marginals = cm.sum(axis=1)
        col_marginals = cm.sum(axis=0)
        expected = (row_marginals * col_marginals).sum() / (total ** 2) if total else 0.0
        kappa = (accuracy - expected) / (1 - expected) if (1 - expected) else 0.0

        per_class = {}
        for idx, label in enumerate(self.labels):
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
                'f1': f1,
            }

        macro_precision = float(np.mean([m['precision'] for m in per_class.values()]))
        macro_recall = float(np.mean([m['recall'] for m in per_class.values()]))
        macro_specificity = float(np.mean([m['specificity'] for m in per_class.values()]))
        macro_f1 = float(np.mean([m['f1'] for m in per_class.values()]))

        metrics = {
            'accuracy': accuracy,
            'kappa': kappa,
            'confusion_matrix': cm,
            'classes': self.labels,
            'per_class': per_class,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_specificity': macro_specificity,
            'macro_f1': macro_f1,
        }

        if len(self.labels) == 2:
            positive_label = self.positive_label if self.positive_label is not None else self.labels[1]
            pos_idx = self.labels.index(positive_label)

            tp = cm[pos_idx, pos_idx]
            fn = cm[pos_idx, :].sum() - tp
            fp = cm[:, pos_idx].sum() - tp
            tn = total - tp - fn - fp

            sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
            specificity = tn / (tn + fp) if (tn + fp) else 0.0
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = sensitivity
            tpr = sensitivity
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            fnr = fn / (fn + tp) if (fn + tp) else 0.0
            tnr = specificity
            f_measure = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

            metrics.update({
                'positive_label': positive_label,
                'binary_counts': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn},
                'sensitivity': sensitivity,
                'specificity': specificity,
                'recall': recall,
                'precision': precision,
                'tpr': tpr,
                'fpr': fpr,
                'fnr': fnr,
                'tnr': tnr,
                'f_measure': f_measure,
                'f1': f_measure,
            })

        return metrics

    def get_expected_value(self, costs, display='inverted'):
        """Return expected value per observation for a given cost/benefit matrix.

        Costs are interpreted with the same layout as `to_dataframe(display=...)`.
        Default is `display='inverted'`.
        """
        costs = np.asarray(costs, dtype=float)
        matrix_for_costs = self.to_dataframe(display=display).to_numpy()
        if costs.shape != matrix_for_costs.shape:
            raise ValueError(
                f"costs must have shape {matrix_for_costs.shape}, got {costs.shape}"
            )

        total_value = float(np.sum(matrix_for_costs * costs))
        return total_value / self.total if self.total else 0.0

    def _matrix_for_display(self, display):
        """Return matrix and labels in the requested display order."""
        if display not in {'normal', 'inverted'}:
            raise ValueError("display must be 'normal' or 'inverted'")

        ordered_labels = self.labels
        ordered_matrix = self.matrix
        if display == 'inverted' and len(self.labels) == 2:
            # Textbook binary layout with positive class first:
            # [[TP, FP],
            #  [FN, TN]]
            positive_label = self.positive_label if self.positive_label is not None else self.labels[1]
            negative_label = next(label for label in self.labels if label != positive_label)
            ordered_labels = [positive_label, negative_label]
            idx = [self.labels.index(label) for label in ordered_labels]
            ordered_matrix = self.matrix[np.ix_(idx, idx)]
        return ordered_matrix, ordered_labels
