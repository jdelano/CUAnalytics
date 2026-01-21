"""
Formula handling utilities.
"""

import pandas as pd
import re


class ModelFormula:
    """
    Parse and store formula metadata and design matrices.
    """

    def __init__(self, df, formula, drop_intercept=True):
        self.formula = formula
        self.drop_intercept = drop_intercept
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

        if '~' in formula:
            lhs, rhs = formula.split('~', 1)
            self.target = lhs.strip() or None
            self.rhs = rhs.strip()
            if self.target and self.target not in df.columns and self.target.isidentifier():
                raise ValueError(f"Target '{self.target}' not found in data")
            model_matrices = model_matrix(formula, df, output='pandas')
        else:
            self.target = None
            self.rhs = formula.strip()
            model_matrices = model_matrix(self.rhs, df, output='pandas')

        if hasattr(model_matrices, 'lhs') and hasattr(model_matrices, 'rhs'):
            y = model_matrices.lhs
            X = model_matrices.rhs
        elif isinstance(model_matrices, tuple) and len(model_matrices) == 2:
            y, X = model_matrices
        else:
            if self.target is not None:
                raise RuntimeError("Unexpected formulaic output when building design matrices.")
            y = None
            X = model_matrices

        self.model_spec = getattr(model_matrices, 'model_spec', None)

        if self.drop_intercept and 'Intercept' in X.columns:
            X = X.drop('Intercept', axis=1)

        self.X = X
        self.feature_names = list(X.columns)

        if y is not None:
            if isinstance(y, pd.Series):
                y = y.to_frame()
            self.y = y.iloc[:, 0]
            if self.target is None:
                self.target = y.columns[0]
        else:
            self.y = None

        if self.target:
            self.feature_names_original = [col for col in df.columns if col != self.target]
        else:
            self.feature_names_original = list(df.columns)

    def transform(self, df):
        """
        Transform new data using the stored formula specification.
        """
        if getattr(self, 'model_spec', None) is None:
            raise RuntimeError("Formula metadata missing; cannot transform new data.")

        rhs_spec = getattr(self.model_spec, 'rhs', self.model_spec)
        model_matrices = rhs_spec.get_model_matrix(df, output='pandas')

        if hasattr(model_matrices, 'rhs'):
            X_new = model_matrices.rhs
        else:
            X_new = model_matrices

        if self.drop_intercept and 'Intercept' in X_new.columns:
            X_new = X_new.drop('Intercept', axis=1)

        return X_new

    def validate_numeric_features(self, df):
        """
        Validate that source features are numeric (unless explicitly encoded).
        """
        used_vars = set()
        if self.model_spec is not None:
            rhs_spec = getattr(self.model_spec, 'rhs', self.model_spec)
            used_vars = {var for var in getattr(rhs_spec, 'variables', set())
                         if var in df.columns and var != self.target}
        if not used_vars:
            used_vars = {col for col in df.columns if col != self.target}

        non_numeric = [col for col in used_vars
                       if not pd.api.types.is_numeric_dtype(df[col])]
        if non_numeric:
            allowed = set()
            for col in non_numeric:
                pattern = r'(?<![\w.])C\(\s*' + re.escape(col) + r'(\s*[,)])'
                if re.search(pattern, self.formula):
                    allowed.add(col)
            non_numeric = [col for col in non_numeric if col not in allowed]
        if non_numeric:
            raise ValueError(
                f"All features must be numeric. Non-numeric features found: {non_numeric}\n"
                "Hint: Use pd.get_dummies() to convert categorical features,\n"
                "      or use formula syntax with C() for categorical variables."
            )

    def validate_regression_target(self, df):
        """
        Validate that a regression formula includes a numeric target column.
        """
        if self.formula is None:
            raise ValueError("Must provide 'formula' for model specification")

        if '~' not in self.formula:
            raise ValueError("Formula must include a target (e.g., 'y ~ x1 + x2').")

        if self.target and self.target not in df.columns and self.target.isidentifier():
            raise ValueError(f"Target '{self.target}' not found in data")

        if self.target in df.columns and not pd.api.types.is_numeric_dtype(df[self.target]):
            raise ValueError(
                f"Target variable '{self.target}' must be numeric for regression.\n"
                f"Found type: {df[self.target].dtype}"
            )
