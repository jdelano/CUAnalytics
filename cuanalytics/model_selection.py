"""
Cross-validation utilities for cuanalytics models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def _parse_target(formula: str) -> str:
    if formula is None or "~" not in formula:
        raise ValueError("Formula must include a target (e.g., 'y ~ x1 + x2').")
    lhs, _rhs = formula.split("~", 1)
    target = lhs.strip()
    if not target:
        raise ValueError("Formula must include a target (e.g., 'y ~ x1 + x2').")
    return target


def _infer_task_type(y: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    if y.nunique() <= 10:
        return "classification"
    return "regression"


def _default_scoring(task: str) -> dict:
    if task == "classification":
        return {
            "accuracy": lambda y_true, y_pred: accuracy_score(y_true, y_pred),
            "kappa": lambda y_true, y_pred: cohen_kappa_score(y_true, y_pred),
            "macro_f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
        }
    return {
        "r2": lambda y_true, y_pred: r2_score(y_true, y_pred),
        "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": lambda y_true, y_pred: mean_absolute_error(y_true, y_pred),
    }


def cross_validate(
    fit_fn,
    df,
    formula,
    k=5,
    shuffle=True,
    random_state=None,
    stratify_on="auto",
    task="auto",
    scoring=None,
    return_models=False,
    return_oof=False,
    verbose=False,
    **model_kwargs,
):
    """
    Cross-validate a cuanalytics supervised model.

    Parameters:
    -----------
    fit_fn : callable
        Model fitting function (e.g., fit_lm, fit_logit).
    df : pd.DataFrame
        DataFrame containing features and target.
    formula : str
        R-style formula (e.g., 'y ~ x1 + x2').
    k : int
        Number of folds.
    shuffle : bool
        Whether to shuffle data before splitting.
    random_state : int | None
        Random seed for reproducibility (used if shuffle=True).
    stratify_on : str | "auto" | None
        Column to stratify on (classification only).
        - "auto": uses target column for classification tasks
        - None: no stratification
    task : str
        "auto", "classification", or "regression".
    scoring : dict[str, callable] | None
        Custom scoring functions. Each callable takes (y_true, y_pred).
    return_models : bool
        If True, returns list of fitted fold models.
    return_oof : bool
        If True, returns out-of-fold predictions as a pandas Series.
    verbose : bool
        If False, suppresses model fit output during cross-validation.
    **model_kwargs
        Additional arguments passed to fit_fn.

    Returns:
    --------
    results : dict
        Dictionary with per-fold metrics and summary statistics.
    """
    if k < 2:
        raise ValueError("k must be at least 2 for cross-validation.")

    target = _parse_target(formula)
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in data.")

    y = df[target]
    resolved_task = task
    if task == "auto":
        resolved_task = _infer_task_type(y)
    if resolved_task not in {"classification", "regression"}:
        raise ValueError("task must be 'auto', 'classification', or 'regression'.")

    if stratify_on == "auto":
        stratify_on = target if resolved_task == "classification" else None
    if stratify_on is not None and stratify_on not in df.columns:
        raise KeyError(f"Stratification column '{stratify_on}' not found in DataFrame")

    scoring_fns = scoring if scoring is not None else _default_scoring(resolved_task)
    if not isinstance(scoring_fns, dict) or not scoring_fns:
        raise ValueError("scoring must be a non-empty dict of name -> callable.")

    if resolved_task == "classification" and stratify_on is not None:
        splitter = StratifiedKFold(
            n_splits=k,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )
        split_iter = splitter.split(df, df[stratify_on])
    else:
        splitter = KFold(
            n_splits=k,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )
        split_iter = splitter.split(df)

    fold_metrics = []
    models = [] if return_models else None
    oof_predictions = pd.Series(index=df.index, dtype=object) if return_oof else None

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter, start=1):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        if verbose:
            model = fit_fn(train_df, formula=formula, **model_kwargs)
        else:
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                model = fit_fn(train_df, formula=formula, **model_kwargs)

        y_true = test_df[target]
        y_pred = model.predict(test_df)

        fold_result = {"fold": fold_idx}
        for name, fn in scoring_fns.items():
            fold_result[name] = fn(y_true, y_pred)
        fold_metrics.append(fold_result)

        if return_models:
            models.append(model)

        if return_oof:
            oof_predictions.iloc[test_idx] = y_pred

    metrics_df = pd.DataFrame(fold_metrics).set_index("fold")
    summary = {
        "mean": metrics_df.mean().to_dict(),
        "std": metrics_df.std(ddof=1).to_dict(),
    }

    results = {
        "task": resolved_task,
        "k": k,
        "stratify_on": stratify_on,
        "folds": fold_metrics,
        "summary": summary,
    }

    if return_models:
        results["models"] = models
    if return_oof:
        results["oof_predictions"] = oof_predictions

    return results
