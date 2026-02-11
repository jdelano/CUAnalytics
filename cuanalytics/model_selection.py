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


def grid_search_cv(
    fit_fn,
    df,
    formula,
    param_grid,
    k=5,
    shuffle=True,
    random_state=None,
    stratify_on="auto",
    task="auto",
    scoring=None,
    refit=None,
    verbose=False,
    **model_kwargs,
):
    """
    Grid-search hyperparameters using cross-validation and refit the best model.

    Parameters:
    -----------
    fit_fn : callable
        Model fitting function (e.g., fit_lm, fit_logit).
    df : pd.DataFrame
        DataFrame containing features and target.
    formula : str
        R-style formula (e.g., 'y ~ x1 + x2').
    param_grid : dict[str, list]
        Dictionary of hyperparameter names to candidate values.
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
    refit : str | None
        Metric name to choose the best params. Defaults to
        "accuracy" for classification and "rmse" for regression.
    verbose : bool
        If False, suppresses model fit output during CV runs.
    **model_kwargs
        Additional arguments passed to fit_fn.

    Returns:
    --------
    results : dict
        Dictionary with CV results and the refit best model.
    """
    import itertools

    if k < 2:
        raise ValueError("k must be at least 2 for cross-validation.")

    if not isinstance(param_grid, dict) or not param_grid:
        raise ValueError("param_grid must be a non-empty dict of name -> list of values.")
    for name, values in param_grid.items():
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            raise ValueError(f"param_grid values for '{name}' must be a non-empty list.")

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

    if refit is None:
        refit = "accuracy" if resolved_task == "classification" else "rmse"
    if refit not in scoring_fns:
        raise ValueError(f"refit '{refit}' not found in scoring functions.")

    if resolved_task == "classification" and stratify_on is not None:
        splitter = StratifiedKFold(
            n_splits=k,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )
        split_iter = list(splitter.split(df, df[stratify_on]))
    else:
        splitter = KFold(
            n_splits=k,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )
        split_iter = list(splitter.split(df))

    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    param_combinations = [
        dict(zip(param_names, values)) for values in itertools.product(*param_values)
    ]

    cv_results = []
    for params in param_combinations:
        fold_metrics = []
        for fold_idx, (train_idx, test_idx) in enumerate(split_iter, start=1):
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            if verbose:
                model = fit_fn(train_df, formula=formula, **model_kwargs, **params)
            else:
                import contextlib
                import io
                with contextlib.redirect_stdout(io.StringIO()):
                    model = fit_fn(train_df, formula=formula, **model_kwargs, **params)

            y_true = test_df[target]
            y_pred = model.predict(test_df)

            fold_result = {"fold": fold_idx}
            for name, fn in scoring_fns.items():
                fold_result[name] = fn(y_true, y_pred)
            fold_metrics.append(fold_result)

        metrics_df = pd.DataFrame(fold_metrics).set_index("fold")
        summary = {
            "mean": metrics_df.mean().to_dict(),
            "std": metrics_df.std(ddof=1).to_dict(),
        }

        cv_results.append(
            {
                "params": params,
                "folds": fold_metrics,
                "mean": summary["mean"],
                "std": summary["std"],
            }
        )

    lower_is_better = {"rmse", "mae", "mse"}
    best_idx = None
    best_score = None
    for idx, result in enumerate(cv_results):
        score = result["mean"][refit]
        if best_idx is None:
            best_idx = idx
            best_score = score
            continue
        if refit in lower_is_better:
            if score < best_score:
                best_idx = idx
                best_score = score
        else:
            if score > best_score:
                best_idx = idx
                best_score = score

    best_params = cv_results[best_idx]["params"]
    if verbose:
        best_model = fit_fn(df, formula=formula, **model_kwargs, **best_params)
    else:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            best_model = fit_fn(df, formula=formula, **model_kwargs, **best_params)

    return {
        "task": resolved_task,
        "k": k,
        "stratify_on": stratify_on,
        "refit": refit,
        "best_params": best_params,
        "best_score": best_score,
        "best_model": best_model,
        "cv_results": cv_results,
    }


def plot_learning_curves(
    fit_fns,
    df,
    formula,
    train_sizes=None,
    k=5,
    shuffle=True,
    random_state=None,
    stratify_on="auto",
    task="auto",
    scoring=None,
    metric=None,
    verbose=False,
    figsize=(10, 6),
    title=None,
    **model_kwargs,
):
    """
    Plot learning curves (validation performance vs training size) for one or more models.

    Parameters:
    -----------
    fit_fns : callable | list[callable]
        Model fitting function(s) (e.g., fit_lm, fit_logit).
    df : pd.DataFrame
        DataFrame containing features and target.
    formula : str
        R-style formula (e.g., 'y ~ x1 + x2').
    train_sizes : list[float|int] | None
        Fractions (0-1] or absolute sizes of training data to use per fold.
        If None, defaults to [0.1, 0.2, 0.4, 0.6, 0.8, 1.0].
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
    metric : str | None
        Metric name to plot (must be a key in scoring). Defaults to
        "accuracy" for classification and "rmse" for regression.
    verbose : bool
        If False, suppresses model fit output during learning curve runs.
    figsize : tuple
        Matplotlib figure size.
    title : str | None
        Plot title.
    **model_kwargs
        Additional arguments passed to fit_fn.

    Returns:
    --------
    results : dict
        Dictionary with learning curve results and matplotlib Axes.
    """
    import matplotlib.pyplot as plt

    if k < 2:
        raise ValueError("k must be at least 2 for learning curves.")

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

    if metric is None:
        metric = "accuracy" if resolved_task == "classification" else "rmse"
    if metric not in scoring_fns:
        raise ValueError(f"metric '{metric}' not found in scoring functions.")

    if train_sizes is None:
        train_sizes = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    if not isinstance(fit_fns, (list, tuple)):
        fit_fns = [fit_fns]
    if not fit_fns:
        raise ValueError("fit_fns must include at least one model function.")
    show_band = len(fit_fns) == 1

    if resolved_task == "classification" and stratify_on is not None:
        splitter = StratifiedKFold(
            n_splits=k,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )
        split_iter = list(splitter.split(df, df[stratify_on]))
    else:
        splitter = KFold(
            n_splits=k,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )
        split_iter = list(splitter.split(df))

    rng = np.random.default_rng(random_state) if shuffle else None
    base_train_len = len(split_iter[0][0]) if split_iter else 0
    plot_x = []
    for size in train_sizes:
        if isinstance(size, float):
            if size <= 0 or size > 1:
                raise ValueError("train_sizes fractions must be in (0, 1].")
            n_train = max(1, int(round(size * base_train_len)))
        else:
            n_train = int(size)
            if n_train <= 0:
                raise ValueError("train_sizes must be positive.")
        n_train = min(n_train, base_train_len)
        plot_x.append(n_train)

    results = {
        "task": resolved_task,
        "train_sizes": train_sizes,
        "train_sizes_abs": plot_x,
        "metric": metric,
        "models": {},
    }

    fig, ax = plt.subplots(figsize=figsize)
    band_label = None

    for fit_fn in fit_fns:
        model_name = getattr(fit_fn, "__name__", "model")
        val_means = []
        val_stds = []

        for size in train_sizes:
            fold_scores = []
            for train_idx, test_idx in split_iter:
                train_idx = np.array(train_idx)
                if shuffle and rng is not None:
                    rng.shuffle(train_idx)

                if isinstance(size, float):
                    if size <= 0 or size > 1:
                        raise ValueError("train_sizes fractions must be in (0, 1].")
                    n_train = max(1, int(round(size * len(train_idx))))
                else:
                    n_train = int(size)
                    if n_train <= 0:
                        raise ValueError("train_sizes must be positive.")
                n_train = min(n_train, len(train_idx))

                subset_idx = train_idx[:n_train]
                train_df = df.iloc[subset_idx]
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
                fold_scores.append(scoring_fns[metric](y_true, y_pred))

            val_means.append(float(np.mean(fold_scores)))
            val_stds.append(float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0)

        results["models"][model_name] = {
            "val_mean": val_means,
            "val_std": val_stds,
        }

        ax.plot(plot_x, val_means, marker="o", label=model_name)
        if show_band:
            band_label = band_label or "±1 std (validation)"
            ax.fill_between(
                plot_x,
                np.array(val_means) - np.array(val_stds),
                np.array(val_means) + np.array(val_stds),
                alpha=0.15,
                label=band_label,
            )
            band_label = None

    ax.set_xlabel("Training Set Size")
    ax.set_ylabel(metric)
    ax.set_title(title or f"Learning Curve ({metric})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    results["ax"] = ax
    return results
