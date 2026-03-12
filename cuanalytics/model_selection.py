"""
Cross-validation utilities for cuanalytics models.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    auc,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_curve,
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


def _as_model_list(models):
    if isinstance(models, (list, tuple)):
        model_list = list(models)
    else:
        model_list = [models]
    if not model_list:
        raise ValueError("models must include at least one fitted classifier.")
    return model_list


def _resolve_model_names(models, model_names=None):
    if model_names is None:
        names = []
        seen = {}
        for model in models:
            base = type(model).__name__
            seen[base] = seen.get(base, 0) + 1
            suffix = f" #{seen[base]}" if seen[base] > 1 else ""
            names.append(f"{base}{suffix}")
        return names
    if len(model_names) != len(models):
        raise ValueError("model_names length must match the number of models.")
    return list(model_names)


def _resolve_positive_class(y_true, positive_class=None):
    classes = list(pd.Series(y_true).dropna().unique())
    if len(classes) != 2:
        raise ValueError("ROC/Lift/Profit currently require binary classification targets.")
    if positive_class is not None:
        if positive_class not in classes:
            raise ValueError(f"positive_class '{positive_class}' not found in y_true classes {classes}.")
        return positive_class
    class_set = set(classes)
    if class_set == {0, 1}:
        return 1
    if class_set == {"0", "1"}:
        return "1"
    return sorted(classes)[-1]


def _resolve_sample_weight(sample_weight, test_df):
    if sample_weight is None:
        return np.ones(len(test_df), dtype=float)
    if isinstance(sample_weight, pd.Series):
        w = sample_weight.reindex(test_df.index).to_numpy(dtype=float)
    else:
        w = np.asarray(sample_weight, dtype=float)
    if len(w) != len(test_df):
        raise ValueError("sample_weight length must match test_df length.")
    return w


def _get_model_classes(model):
    if hasattr(model, "target_encoder") and hasattr(model.target_encoder, "classes_"):
        return list(model.target_encoder.classes_)
    if hasattr(model, "model") and hasattr(model.model, "classes_"):
        return list(model.model.classes_)
    if hasattr(model, "tree") and hasattr(model.tree, "classes_"):
        return list(model.tree.classes_)
    if hasattr(model, "lda") and hasattr(model.lda, "classes_"):
        return list(model.lda.classes_)
    if hasattr(model, "classes"):
        return list(model.classes)
    return None


def _extract_positive_scores(model, test_df, positive_class):
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            f"{type(model).__name__} does not implement predict_proba(df). "
            "ROC/Lift/Profit require probability scores."
        )

    proba = model.predict_proba(test_df)
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1:
        return proba
    if proba.ndim != 2:
        raise ValueError(f"{type(model).__name__}.predict_proba returned unexpected shape: {proba.shape}")
    if proba.shape[1] < 2:
        raise ValueError(f"{type(model).__name__}.predict_proba must return at least 2 columns for binary tasks.")

    classes = _get_model_classes(model)
    if classes is not None:
        classes = list(classes)
        if positive_class in classes:
            return proba[:, classes.index(positive_class)]

    if proba.shape[1] == 2:
        warnings.warn(
            f"Could not resolve classes for {type(model).__name__}; using predict_proba[:, 1] for positive class.",
            RuntimeWarning,
            stacklevel=2,
        )
        return proba[:, 1]
    raise ValueError(
        f"Could not map positive_class '{positive_class}' to predict_proba columns for {type(model).__name__}."
    )


class CurveAnalysisResult:
    """
    Holds model scores and cached curve calculations for ROC/Lift/Profit analysis.
    """

    def __init__(
        self,
        models,
        model_names,
        test_df,
        y_true,
        positive_class,
        sample_weight,
        profit_config,
        scores_by_model,
        target,
    ):
        self.models = models
        self.model_names = model_names
        self.test_df = test_df
        self.y_true = pd.Series(y_true, index=test_df.index)
        self.target = target
        self.positive_class = positive_class
        self.sample_weight = sample_weight
        self.profit_config = profit_config
        self.scores_by_model = scores_by_model
        self._curve_cache = {}

    def _calculate_curve(self, curve):
        curve = str(curve).lower()
        if curve in self._curve_cache:
            return self._curve_cache[curve]
        if curve not in {"roc", "lift", "profit", "cumulative_response"}:
            raise ValueError("curve must be one of {'roc', 'lift', 'profit', 'cumulative_response'}.")

        y_bin = (self.y_true.values == self.positive_class).astype(int)
        w = self.sample_weight
        curve_result = {}

        for model_name in self.model_names:
            score = self.scores_by_model[model_name]

            if curve == "roc":
                fpr, tpr, thresholds = roc_curve(y_bin, score, sample_weight=w)
                model_df = pd.DataFrame({
                    "threshold": thresholds,
                    "fpr": fpr,
                    "tpr": tpr,
                })
                finite_mask = np.isfinite(thresholds)
                if np.any(finite_mask):
                    dist = np.hypot(fpr[finite_mask], 1 - tpr[finite_mask])
                    best_local_idx = int(np.argmin(dist))
                    best_idx = np.where(finite_mask)[0][best_local_idx]
                else:
                    dist = np.hypot(fpr, 1 - tpr)
                    best_idx = int(np.argmin(dist))
                curve_result[model_name] = {
                    "data": model_df,
                    "auc": float(auc(fpr, tpr)),
                    "best_cutoff": float(thresholds[best_idx]),
                    "best_fpr": float(fpr[best_idx]),
                    "best_tpr": float(tpr[best_idx]),
                    "best_distance": float(np.hypot(fpr[best_idx], 1 - tpr[best_idx])),
                }
                continue

            if curve in {"lift", "cumulative_response"}:
                order = np.argsort(-score, kind="mergesort")
                y_sorted = y_bin[order]
                w_sorted = w[order]
                score_sorted = score[order]
                cum_w = np.cumsum(w_sorted)
                cum_pos = np.cumsum(w_sorted * y_sorted)
                total_w = float(cum_w[-1]) if len(cum_w) else 0.0
                total_pos = float(cum_pos[-1]) if len(cum_pos) else 0.0

                pop_frac = cum_w / total_w if total_w else np.zeros_like(cum_w)
                capture = cum_pos / total_pos if total_pos else np.zeros_like(cum_pos)
                with np.errstate(divide="ignore", invalid="ignore"):
                    lift = np.where(pop_frac > 0, capture / pop_frac, 1.0)

                model_df = pd.DataFrame({
                    "threshold": score_sorted,
                    "population_frac": pop_frac,
                    "capture_rate": capture,
                    "lift": lift,
                })
                model_df = pd.concat(
                    [pd.DataFrame([{
                        "threshold": 1.0,
                        "population_frac": 0.0,
                        "capture_rate": 0.0,
                        "lift": 1.0,
                    }]), model_df],
                    ignore_index=True,
                )
                top_decile = model_df.loc[model_df["population_frac"] >= 0.10, "lift"]
                top_decile_lift = float(top_decile.iloc[0]) if not top_decile.empty else float(model_df["lift"].iloc[-1])
                curve_result[model_name] = {
                    "data": model_df,
                    "top_decile_lift": top_decile_lift,
                }
                continue

            order = np.argsort(-score, kind="mergesort")
            y_sorted = y_bin[order]
            w_sorted = w[order]
            score_sorted = score[order]

            total_w = float(np.sum(w_sorted))
            total_pos_w = float(np.sum(w_sorted * y_sorted))
            total_neg_w = float(np.sum(w_sorted * (1 - y_sorted)))

            tp_cum = np.cumsum(w_sorted * y_sorted)
            fp_cum = np.cumsum(w_sorted * (1 - y_sorted))
            fn_cum = total_pos_w - tp_cum
            tn_cum = total_neg_w - fp_cum
            pop_frac = tp_cum.copy()
            if total_w > 0:
                pop_frac = np.cumsum(w_sorted) / total_w
            else:
                pop_frac = np.zeros_like(tp_cum)

            # Baseline at threshold=1.0 means no one is targeted.
            tp = np.concatenate(([0.0], tp_cum))
            fp = np.concatenate(([0.0], fp_cum))
            fn = np.concatenate(([total_pos_w], fn_cum))
            tn = np.concatenate(([total_neg_w], tn_cum))
            thresholds = np.concatenate(([1.0], score_sorted))
            population_frac = np.concatenate(([0.0], pop_frac))

            conf = self.profit_config
            profit = (
                tp * conf["tp_value"]
                + fp * conf["fp_value"]
                + tn * conf["tn_value"]
                + fn * conf["fn_value"]
                + conf["fixed_value"]
            )

            model_df = pd.DataFrame({
                "threshold": thresholds,
                "population_frac": population_frac,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "profit": profit,
            })
            best_idx = int(np.argmax(profit))
            curve_result[model_name] = {
                "data": model_df,
                "max_profit": float(profit[best_idx]),
                "max_profit_threshold": float(thresholds[best_idx]),
                "max_profit_population_frac": float(population_frac[best_idx]),
            }

        self._curve_cache[curve] = curve_result
        return curve_result

    def get_roc(self):
        return self._calculate_curve("roc")

    def get_lift(self):
        return self._calculate_curve("lift")

    def get_profit(self):
        return self._calculate_curve("profit")

    def get_cumulative_response(self):
        return self._calculate_curve("cumulative_response")

    def summary_table(self):
        roc = self.get_roc()
        lift = self.get_lift()
        profit = self.get_profit()
        rows = []
        for name in self.model_names:
            rows.append({
                "model": name,
                "auc": roc[name]["auc"],
                "top_decile_lift": lift[name]["top_decile_lift"],
                "max_profit": profit[name]["max_profit"],
                "max_profit_threshold": profit[name]["max_profit_threshold"],
            })
        return pd.DataFrame(rows).set_index("model")


def calculate_curve(
    models,
    test_df,
    positive_class=None,
    profit_config=None,
    sample_weight=None,
    model_names=None,
):
    """
    Build a curve-analysis result object for ROC/Lift/Profit comparisons.
    """
    models = _as_model_list(models)
    model_names = _resolve_model_names(models, model_names=model_names)

    first_model = models[0]
    target = getattr(first_model, "target", None)
    if target is None:
        raise ValueError("Could not infer target column from model. Expected model.target.")
    if target not in test_df.columns:
        raise ValueError(f"Target '{target}' not found in test_df.")
    for model in models[1:]:
        model_target = getattr(model, "target", None)
        if model_target != target:
            raise ValueError("All models must share the same target column.")

    y_true = test_df[target]
    positive_class = _resolve_positive_class(y_true, positive_class=positive_class)
    w = _resolve_sample_weight(sample_weight, test_df)

    default_profit = {
        "tp_value": 1.0,
        "fp_value": -1.0,
        "tn_value": 0.0,
        "fn_value": -1.0,
        "fixed_value": 0.0,
    }
    if profit_config is None:
        resolved_profit = default_profit
    else:
        resolved_profit = default_profit.copy()
        resolved_profit.update(profit_config)
    required = {"tp_value", "fp_value", "tn_value", "fn_value", "fixed_value"}
    missing = required - set(resolved_profit.keys())
    if missing:
        raise ValueError(f"profit_config missing required keys: {sorted(missing)}")

    scores_by_model = {}
    for model, name in zip(models, model_names):
        scores = _extract_positive_scores(model, test_df, positive_class)
        if len(scores) != len(test_df):
            raise ValueError(f"{name}.predict_proba returned {len(scores)} rows, expected {len(test_df)}.")
        scores_by_model[name] = np.asarray(scores, dtype=float)

    return CurveAnalysisResult(
        models=models,
        model_names=model_names,
        test_df=test_df,
        y_true=y_true,
        positive_class=positive_class,
        sample_weight=w,
        profit_config=resolved_profit,
        scores_by_model=scores_by_model,
        target=target,
    )


def plot_roc(
    models,
    test_df=None,
    positive_class=None,
    sample_weight=None,
    model_names=None,
    show_cutoffs=False,
    cutoff_step=0.1,
    cutoff_fontsize=8,
    show=True,
    ax=None,
    figsize=(8, 6),
    title="ROC Curve",
):
    """
    Plot ROC comparison and return ROC curve data.
    """
    import matplotlib.pyplot as plt

    if isinstance(models, CurveAnalysisResult):
        curve_result = models
    else:
        if test_df is None:
            raise ValueError("test_df is required when passing models to plot_roc.")
        curve_result = calculate_curve(
            models=models,
            test_df=test_df,
            positive_class=positive_class,
            sample_weight=sample_weight,
            model_names=model_names,
        )
    if ax is None:
        _fig, ax = plt.subplots(figsize=figsize)

    roc = curve_result.get_roc()
    for name in curve_result.model_names:
        data = roc[name]["data"]
        line, = ax.plot(data["fpr"], data["tpr"], label=f"{name} (AUC={roc[name]['auc']:.3f})")

        if show_cutoffs:
            if cutoff_step <= 0 or cutoff_step > 1:
                raise ValueError("cutoff_step must be in (0, 1].")

            finite_mask = np.isfinite(data["threshold"].to_numpy())
            finite_df = data.loc[finite_mask].copy()
            if not finite_df.empty:
                thresholds = finite_df["threshold"].to_numpy(dtype=float)
                fprs = finite_df["fpr"].to_numpy(dtype=float)
                tprs = finite_df["tpr"].to_numpy(dtype=float)

                desired = np.arange(0.0, 1.0 + 1e-9, cutoff_step)
                used_idx = set()
                for cutoff in desired:
                    idx = int(np.argmin(np.abs(thresholds - cutoff)))
                    if idx in used_idx:
                        continue
                    used_idx.add(idx)
                    x = float(fprs[idx])
                    y = float(tprs[idx])
                    ax.scatter([x], [y], s=14, color=line.get_color(), zorder=4)
                    ax.annotate(
                        f"{cutoff:.1f}",
                        (x, y),
                        textcoords="offset points",
                        xytext=(4, 4),
                        fontsize=cutoff_fontsize,
                        color=line.get_color(),
                        alpha=0.9,
                    )

            best_x = roc[name]["best_fpr"]
            best_y = roc[name]["best_tpr"]
            ax.scatter(
                [best_x],
                [best_y],
                marker="*",
                s=120,
                facecolor=line.get_color(),
                edgecolor="black",
                linewidth=0.7,
                zorder=5,
            )
            ax.annotate(
                f"best={roc[name]['best_cutoff']:.2f}",
                (best_x, best_y),
                textcoords="offset points",
                xytext=(6, -10),
                fontsize=max(7, cutoff_fontsize - 1),
                color=line.get_color(),
                alpha=0.95,
            )

    ax.plot([0, 1], [0, 1], linestyle="--", color="#6f665b", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    if show:
        plt.show()
    return {"ax": ax, "data": roc}


def plot_lift(
    models,
    test_df=None,
    positive_class=None,
    sample_weight=None,
    model_names=None,
    show=True,
    ax=None,
    figsize=(8, 6),
    title="Lift Curve",
):
    """
    Plot lift comparison and return lift curve data.
    """
    import matplotlib.pyplot as plt

    if isinstance(models, CurveAnalysisResult):
        curve_result = models
    else:
        if test_df is None:
            raise ValueError("test_df is required when passing models to plot_lift.")
        curve_result = calculate_curve(
            models=models,
            test_df=test_df,
            positive_class=positive_class,
            sample_weight=sample_weight,
            model_names=model_names,
        )
    if ax is None:
        _fig, ax = plt.subplots(figsize=figsize)

    lift = curve_result.get_lift()
    for name in curve_result.model_names:
        data = lift[name]["data"]
        ax.plot(data["population_frac"] * 100, data["lift"], label=f"{name} (Top 10%={lift[name]['top_decile_lift']:.2f}x)")

    ax.axhline(1.0, linestyle="--", color="#6f665b", label="Baseline")
    ax.set_xlabel("Population Contacted (%)")
    ax.set_ylabel("Lift")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    if show:
        plt.show()
    return {"ax": ax, "data": lift}


def plot_profit(
    models,
    test_df=None,
    positive_class=None,
    profit_config=None,
    sample_weight=None,
    model_names=None,
    show=True,
    ax=None,
    figsize=(8, 6),
    title="Profit Curve",
):
    """
    Plot profit vs targeted test population (%) and return profit curve data.
    """
    import matplotlib.pyplot as plt

    if isinstance(models, CurveAnalysisResult):
        curve_result = models
    else:
        if test_df is None:
            raise ValueError("test_df is required when passing models to plot_profit.")
        curve_result = calculate_curve(
            models=models,
            test_df=test_df,
            positive_class=positive_class,
            profit_config=profit_config,
            sample_weight=sample_weight,
            model_names=model_names,
        )
    if ax is None:
        _fig, ax = plt.subplots(figsize=figsize)

    profit = curve_result.get_profit()
    for name in curve_result.model_names:
        data = profit[name]["data"]
        ax.plot(data["population_frac"] * 100, data["profit"], label=name)
        max_x = profit[name].get("max_profit_population_frac")
        if max_x is None:
            max_x = float(data.loc[data["profit"].idxmax(), "population_frac"])
        ax.scatter(
            [max_x * 100],
            [profit[name]["max_profit"]],
            s=30,
            zorder=3,
        )

    ax.set_xlabel("Test Instances Targeted (%)")
    ax.set_ylabel("Profit")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    if show:
        plt.show()
    return {"ax": ax, "data": profit}


def plot_cumulative_response(
    models,
    test_df=None,
    positive_class=None,
    sample_weight=None,
    model_names=None,
    show=True,
    ax=None,
    figsize=(8, 6),
    title="Cumulative Response Curve",
):
    """
    Plot cumulative response (capture rate) vs population percentage.
    Returns the same underlying curve data used for lift.
    """
    import matplotlib.pyplot as plt

    if isinstance(models, CurveAnalysisResult):
        curve_result = models
    else:
        if test_df is None:
            raise ValueError("test_df is required when passing models to plot_cumulative_response.")
        curve_result = calculate_curve(
            models=models,
            test_df=test_df,
            positive_class=positive_class,
            sample_weight=sample_weight,
            model_names=model_names,
        )
    if ax is None:
        _fig, ax = plt.subplots(figsize=figsize)

    cumulative = curve_result.get_cumulative_response()
    for name in curve_result.model_names:
        data = cumulative[name]["data"]
        ax.plot(data["population_frac"] * 100, data["capture_rate"] * 100, label=name)

    ax.plot([0, 100], [0, 100], linestyle="--", color="#6f665b", label="Random baseline")
    ax.set_xlabel("Population Contacted (%)")
    ax.set_ylabel("Cumulative Response / Capture (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    if show:
        plt.show()
    return {"ax": ax, "data": cumulative}
