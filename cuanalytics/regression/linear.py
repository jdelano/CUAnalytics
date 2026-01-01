# cuanalytics/regression/linear.py
"""
Linear regression implementation for ITM 4150.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class LinearRegressionModel:
    """
    Linear Regression model for predicting continuous outcomes.
    
    This class wraps scikit-learn's LinearRegression to provide a student-friendly
    interface for learning about linear regression.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and target
    target : str, optional
        Name of target column (required unless using formula)
    features : list of str, optional
        List of feature columns to use. If None, uses all except target.
    formula : str, optional
        R-style formula for specifying the model
    """
    
    def __init__(self, df, target=None, features=None, formula=None):
        self.original_df = df  # Keep original for reference
        self.formula = formula
        self.design_info = None  # For formula transformations
        
        if formula is not None:
            # Use patsy to parse formula and create transformed dataframe
            try:
                from patsy import dmatrices
            except ImportError:
                raise ImportError(
                    "Formula support requires the 'patsy' library.\n"
                    "Install it with: pip install patsy"
                )
            
            # Expand '.' to all column names (patsy sometimes has issues with .)
            if '~' in formula:
                lhs, rhs = formula.split('~')
                lhs = lhs.strip()
                rhs = rhs.strip()
                
                # Handle the . notation
                if '.' in rhs:
                    # Get target from LHS
                    target_from_formula = lhs
                    if target_from_formula not in df.columns:
                        raise ValueError(f"Target '{target_from_formula}' not found in data")
                    
                    # Get all features except target
                    all_features = [col for col in df.columns if col != target_from_formula]
                    
                    # Handle exclusions (. - feature1 - feature2)
                    excluded_features = []
                    if '-' in rhs:
                        # Parse exclusions
                        parts = rhs.split('-')
                        # First part should contain the .
                        base = parts[0].strip()
                        # Remaining parts are exclusions
                        for part in parts[1:]:
                            excluded = part.strip().split('+')[0].strip()  # Handle '. - x + y'
                            excluded_features.append(excluded)
                        
                        # Remove excluded features
                        all_features = [f for f in all_features if f not in excluded_features]
                        
                        # Rebuild RHS
                        rhs = ' + '.join(all_features)
                        
                        # Add back any additions after the exclusions
                        if '+' in parts[-1]:
                            additions = ' + '.join(parts[-1].split('+')[1:])
                            rhs = rhs + ' + ' + additions
                    
                    elif rhs == '.' or rhs == '. ':
                        # Simple case: just all features
                        rhs = ' + '.join(all_features)
                    else:
                        # Case like '. + interaction'
                        rhs = rhs.replace('.', ' + '.join(all_features))
                    
                    formula = f"{lhs} ~ {rhs}"
            
            # Parse formula and create design matrices
            y, X = dmatrices(formula, df, return_type='dataframe')
            
            # Store design info for transforming new data later
            self.design_info = X.design_info
            
            # Extract target name
            self.target = y.columns[0]
            
            # Remove intercept (sklearn adds its own)
            if 'Intercept' in X.columns:
                X = X.drop('Intercept', axis=1)
            
            # Create working dataframe with target + transformed features
            self.df = X.copy()
            self.df[self.target] = y.iloc[:, 0]
            
            # Feature names are the transformed column names
            self.feature_names = list(X.columns)
            
        else:
            # Standard approach (no formula)
            if target is None:
                raise ValueError("Must provide 'target' when not using formula")
            
            self.target = target
            
            # Determine which features to use
            if features is None:
                # Use all columns except target
                self.df = df.copy()
                self.feature_names = [col for col in df.columns if col != target]
            else:
                # Use specified features only
                missing = [f for f in features if f not in df.columns]
                if missing:
                    raise ValueError(f"Features not found in DataFrame: {missing}")
                if target in features:
                    raise ValueError(f"Target '{target}' cannot be in features list")
                
                # Create working dataframe with only selected features + target
                self.df = df[features + [target]].copy()
                self.feature_names = features
        
        # Now extract X and y from the working dataframe
        self.X = self.df[self.feature_names]
        self.y = self.df[self.target]
        
        # Validate target is numeric
        if not pd.api.types.is_numeric_dtype(self.y):
            raise ValueError(
                f"Target variable '{self.target}' must be numeric for regression.\n"
                f"Found type: {self.y.dtype}"
            )
        
        # Check if features are numeric
        non_numeric = [col for col in self.X.columns 
                       if not pd.api.types.is_numeric_dtype(self.X[col])]
        if non_numeric:
            raise ValueError(
                f"All features must be numeric. Non-numeric features found: {non_numeric}\n"
                "Hint: Use pd.get_dummies() to convert categorical features,\n"
                "      or use formula syntax with C() for categorical variables."
            )
        
        # Create and fit model
        try:
            self._fit()
        except Exception as e:
            raise RuntimeError(f"Failed to fit linear regression model: {str(e)}")
        

    
    def _fit(self):
        """Fit the linear regression model."""
        self.model = LinearRegression()
        self.model.fit(self.X, self.y)
        
    def _check_fitted(self):
        """Check if the model has been fitted."""
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError(
                "Model has not been fitted yet. "
                "Create model with: model = fit_linear_regression(df, target='column_name')"
            )
    
    def _transform_data_with_formula(self, df):
        """
        Transform data using the stored formula design info.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with original column names
        
        Returns:
        --------
        X : pd.DataFrame
            Transformed feature matrix (without intercept)
        """
        from patsy import dmatrix
        
        X_new = dmatrix(self.design_info, df, return_type='dataframe')
        
        # Remove intercept if present
        if 'Intercept' in X_new.columns:
            X_new = X_new.drop('Intercept', axis=1)
        
        return X_new

    def predict(self, df):
        """
        Predict target values for new data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data to predict (with or without target column)
            If using formula, provide data with original column names
        
        Returns:
        --------
        predictions : array
            Predicted values
        """
        self._check_fitted()
        
        if self.formula is not None:
            # Transform new data using formula
            X = self._transform_data_with_formula(df)
            return self.model.predict(X)
        else:
            # Standard approach
            X = df[self.feature_names] if self.target in df.columns else df
            return self.model.predict(X)
    
    def score(self, df):
        """
        Calculate R² score on a dataset.
        
        R² (coefficient of determination) measures how well the model
        explains the variance in the target variable.
        - R² = 1.0: Perfect predictions
        - R² = 0.0: Model is no better than predicting the mean
        - R² < 0.0: Model is worse than predicting the mean
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with true target values
        
        Returns:
        --------
        r2 : float
            R² score
        """
        self._check_fitted()
        
        if self.formula is not None:
            X = self._transform_data_with_formula(df)
        else:
            X = df[self.feature_names]
        
        y_true = df[self.target]
        
        return self.model.score(X, y_true)
    
    def get_metrics(self, df):
        """
        Calculate multiple regression metrics on a dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with true target values
        
        Returns:
        --------
        metrics : dict
            Dictionary with R², RMSE, and MAE
        """
        self._check_fitted()
        
        y_true = df[self.target]
        y_pred = self.predict(df)
        
        metrics = {
            'R2': r2_score(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred)
        }
        
        return metrics
    
    def get_coefficients(self):
        """
        Get the regression coefficients.
        
        Returns:
        --------
        coef_df : pd.DataFrame
            DataFrame with feature names and their coefficients
        """
        self._check_fitted()
        
        readable_features = [f.replace(':', ' × ') for f in self.feature_names]
        
        coef_df = pd.DataFrame({
            'feature': readable_features,
            'coefficient': self.model.coef_
        })
        
        return coef_df
    
    def get_equation(self):
        """
        Get the regression equation as a string.
        
        Returns:
        --------
        equation : str
            The regression equation
        """
        terms = []
        for i, feature in enumerate(self.feature_names):
            # Make feature names more readable
            readable_feature = feature.replace(':', ' × ')
            
            coef = self.model.coef_[i]
            if i == 0:
                terms.append(f"{coef:.4f}×({readable_feature})")
            else:
                sign = "+" if coef >= 0 else "-"
                terms.append(f"{sign} {abs(coef):.4f}×({readable_feature})")
        
        equation = " ".join(terms)
        intercept = self.model.intercept_
        sign = "+" if intercept >= 0 else "-"
        equation += f" {sign} {abs(intercept):.4f}"
        
        return f"ŷ = {equation}"
    
    def visualize(self, figsize=(14, 5)):
        """
        Visualize regression results.
        
        Shows:
        1. Predicted vs Actual values
        2. Residual plot
        3. Feature coefficients
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        self._check_fitted()
        
        # Use the already-transformed X and y
        y_pred = self.model.predict(self.X)
        residuals = self.y - y_pred
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        
        # Plot 1: Predicted vs Actual
        ax1.scatter(self.y, y_pred, alpha=0.6, edgecolors='black', s=50)
        
        # Add perfect prediction line
        min_val = min(self.y.min(), y_pred.min())
        max_val = max(self.y.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        
        ax1.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
        ax1.set_title('Predicted vs Actual', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Residual plot
        ax2.scatter(y_pred, residuals, alpha=0.6, edgecolors='black', s=50)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        
        ax2.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
        ax2.set_title('Residual Plot', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Coefficient magnitudes
        coef_df = self.get_coefficients()
        
        # Limit to top features if there are too many (e.g., > 15)
        if len(coef_df) > 15:
            coef_df = coef_df.reindex(coef_df['coefficient'].abs().sort_values(ascending=False).index).head(15)
            title_suffix = ' (Top 15)'
        else:
            title_suffix = ''
        
        coef_df = coef_df.sort_values('coefficient', key=abs, ascending=True)
        
        colors = ['red' if c < 0 else 'green' for c in coef_df['coefficient']]
        ax3.barh(range(len(coef_df)), coef_df['coefficient'], color=colors, 
                edgecolor='black', linewidth=1.5)
        ax3.set_yticks(range(len(coef_df)))
        ax3.set_yticklabels(coef_df['feature'], fontsize=8)  # Smaller font for long names
        ax3.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
        ax3.set_title(f'Feature Coefficients{title_suffix}', fontsize=14, fontweight='bold')
        ax3.axvline(x=0, color='black', linestyle='-', lw=1)
        ax3.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("\nVisualization Notes:")
        print("  • Predicted vs Actual: Points closer to red line = better predictions")
        print("  • Residual Plot: Points closer to horizontal line = better fit")
        print("  • Coefficients: Green = positive effect, Red = negative effect")
        
        if self.formula:
            print(f"\nModel uses formula: {self.formula}")    
            
    def visualize_feature(self, feature, figsize=(10, 6)):
        """
        Visualize regression line for a single feature.
        
        Shows scatter plot with regression line when controlling for other features
        at their mean values.
        
        Parameters:
        -----------
        feature : str
            Feature to visualize
        figsize : tuple
            Figure size
        """
        self._check_fitted()
        
        if feature not in self.feature_names:
            raise ValueError(
                f"Feature must be from: {self.feature_names}\n"
                f"You provided: {feature}"
            )
        
        # Create range of values for the selected feature
        feature_values = np.linspace(self.X[feature].min(), self.X[feature].max(), 100)
        
        # Create prediction DataFrame with other features at mean
        X_pred = pd.DataFrame(index=range(100))
        
        for feat in self.feature_names:
            if feat == feature:
                X_pred[feat] = feature_values
            else:
                mean_val = self.X[feat].mean()
                # Check for NaN (shouldn't happen with valid data, but just in case)
                if pd.isna(mean_val):
                    mean_val = self.X[feat].median()  # Fall back to median
                X_pred[feat] = mean_val
        
        # Predict using the DataFrame
        y_pred = self.model.predict(X_pred)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Actual data points
        ax.scatter(self.X[feature], self.y, alpha=0.6, s=50, 
                edgecolors='black', label='Actual data')
        
        # Regression line (holding other features at mean)
        ax.plot(feature_values, y_pred, 'r-', lw=3, 
            label=f'Regression line\n(other features at mean)')
        
        ax.set_xlabel(feature, fontsize=12, fontweight='bold')
        ax.set_ylabel(self.target, fontsize=12, fontweight='bold')
        ax.set_title(f'Linear Regression: {self.target} vs {feature}', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Show coefficient for this feature
        feature_idx = self.feature_names.index(feature)
        coef = self.model.coef_[feature_idx]
        print(f"\n{feature} coefficient: {coef:.4f}")
        print(f"Interpretation: For each 1-unit increase in {feature},")
        print(f"                {self.target} {'increases' if coef > 0 else 'decreases'} by {abs(coef):.4f} units")
        print(f"                (holding other features constant)")

    def visualize_all_features(self, figsize=None, cols=3):
        """
        Create a grid of scatter plots showing each feature vs the target variable.
        
        Useful for exploratory data analysis to see which features have
        strong relationships with the target.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height). If None, automatically calculated.
        cols : int
            Number of columns in the grid (default: 3)
        
        Examples:
        ---------
        >>> model = fit_lm(train, target='monthly_sales')
        >>> model.visualize_all_features()
        >>> model.visualize_all_features(cols=2)  # 2 columns instead of 3
        """
        self._check_fitted()
        
        n_features = len(self.feature_names)
        
        # Calculate grid dimensions
        rows = int(np.ceil(n_features / cols))
        
        # Auto-calculate figure size if not provided
        if figsize is None:
            figsize = (5 * cols, 4 * rows)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # Flatten axes array for easier iteration
        # Handle different subplot return types
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            # Single subplot returns an Axes object, not an array
            axes = [axes]
                
        # Plot each feature
        for i, feature in enumerate(self.feature_names):
            ax = axes[i]
            
            # Scatter plot
            ax.scatter(self.X[feature], self.y, alpha=0.6, s=30, 
                    edgecolors='black', linewidth=0.5)
            
            # Add trend line
            z = np.polyfit(self.X[feature], self.y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(self.X[feature].min(), self.X[feature].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
            
            # Calculate and display correlation
            corr = self.X[feature].corr(self.y)
            
            ax.set_xlabel(feature, fontsize=10, fontweight='bold')
            ax.set_ylabel(self.target, fontsize=10, fontweight='bold')
            ax.set_title(f'{feature}\n(r = {corr:.3f})', fontsize=11)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Feature Relationships with {self.target}', 
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.show()
        
        # Print correlation summary
        print(f"\nCorrelations with {self.target}:")
        print("-" * 50)
        correlations = []
        for feature in self.feature_names:
            corr = self.X[feature].corr(self.y)
            correlations.append({'feature': feature, 'correlation': corr})
        
        corr_df = pd.DataFrame(correlations).sort_values('correlation', 
                                                        key=abs, 
                                                        ascending=False)
        print(corr_df.to_string(index=False))
        
        print("\nNote: Correlation (r) measures linear relationship strength:")
        print("  • |r| > 0.7: Strong relationship")
        print("  • 0.3 < |r| < 0.7: Moderate relationship")
        print("  • |r| < 0.3: Weak relationship")

    def summary(self):
        """
        Print detailed summary of regression model in traditional statistical format.
        
        Shows regression table with coefficients, t-statistics, p-values, significance stars,
        ANOVA table with F-statistic, and model fit metrics.
        
        Returns:
        --------
        summary_dict : dict
            Dictionary with model information and metrics
        
        Examples:
        ---------
        >>> model = fit_lm(train, target='price')
        >>> summary = model.summary()
        """
        self._check_fitted()
        
        from scipy import stats
        
        # Calculate metrics
        y_pred = self.model.predict(self.X)
        residuals = self.y - y_pred
        n = len(self.y)
        k = len(self.feature_names)  # Number of predictors
        
        # Calculate R², adjusted R², RMSE, MAE
        r2 = r2_score(self.y, y_pred)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        rmse = np.sqrt(mean_squared_error(self.y, y_pred))
        mae = mean_absolute_error(self.y, y_pred)
        
        # Calculate standard errors and t-statistics
        # Residual standard error
        rse = np.sqrt(np.sum(residuals**2) / (n - k - 1))
        
        # Variance-covariance matrix
        X_with_intercept = np.column_stack([np.ones(n), self.X])
        try:
            var_covar = rse**2 * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            std_errors = np.sqrt(np.diag(var_covar))
        except np.linalg.LinAlgError:
            # If matrix is singular, use approximate values
            std_errors = np.ones(k + 1) * rse / np.sqrt(n)
        
        # Coefficients with intercept
        all_coefs = np.concatenate([[self.model.intercept_], self.model.coef_])
        
        # t-statistics and p-values
        t_stats = all_coefs / std_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k - 1))
        
        # Confidence intervals (95%)
        t_critical = stats.t.ppf(0.975, df=n - k - 1)
        conf_lower = all_coefs - t_critical * std_errors
        conf_upper = all_coefs + t_critical * std_errors
        
        # ANOVA table calculations
        ss_total = np.sum((self.y - self.y.mean())**2)
        ss_regression = np.sum((y_pred - self.y.mean())**2)
        ss_residual = np.sum(residuals**2)
        
        ms_regression = ss_regression / k
        ms_residual = ss_residual / (n - k - 1)
        
        f_statistic = ms_regression / ms_residual
        f_pvalue = 1 - stats.f.cdf(f_statistic, k, n - k - 1)
        
        # Determine column width for variable names (dynamic based on longest name)
        readable_features = [f.replace(':', ' × ') for f in self.feature_names]
        var_width = max(len('(Intercept)'), max(len(f) for f in readable_features), 15)
        
        # Calculate total table width
        table_width = var_width + 16 + 16 + 10 + 12 + 8 + 5  # 5 spaces between 6 columns
        
        # Print summary
        print("\n" + "="*table_width)
        print("LINEAR REGRESSION SUMMARY")
        print("="*table_width)
        
        # Header information
        print(f"\nDependent Variable: {self.target}")
        print(f"Number of observations: {n}")
        print(f"Number of predictors: {k}")
        
        # Model fit statistics
        print("\n" + "-"*table_width)
        print("MODEL FIT:")
        print("-"*table_width)
        print(f"R-squared:           {r2:.4f}      Adjusted R-squared:  {adj_r2:.4f}")
        p_formatted = self._format_pvalue(f_pvalue)
        print(f"F-statistic:         {f_statistic:.4f}      Prob (F-statistic):  {p_formatted}")
        print(f"Residual Std Error:  {rse:.4f}      on {n - k - 1} degrees of freedom")
        print(f"RMSE:                {rmse:.4f}")
        print(f"MAE:                 {mae:.4f}")
        
        # ANOVA table
        print("\n" + "-"*table_width)
        print("ANALYSIS OF VARIANCE (ANOVA):")
        print("-"*table_width)
        print(f"{'Source':<15} {'df':>8} {'Sum of Sq':>15} {'Mean Sq':>15} {'F value':>12} {'Pr(>F)':>12}")
        print("-"*table_width)
        p_formatted = self._format_pvalue(f_pvalue)
        print(f"{'Regression':<15} {k:>8} {ss_regression:>15.4f} {ms_regression:>15.4f} {f_statistic:>12.4f} {p_formatted:>12}")
        print(f"{'Residual':<15} {n-k-1:>8} {ss_residual:>15.4f} {ms_residual:>15.4f}")
        print(f"{'Total':<15} {n-1:>8} {ss_total:>15.4f}")
        
        # Coefficients table
        print("\n" + "-"*table_width)
        print("COEFFICIENTS:")
        print("-"*table_width)
        print(f"{'Variable':<{var_width}} {'Coefficient (β)':>16} {'Std Error (ε)':>16} {'t-value':>10} {'Pr(>|t|)':>12} {'Sig.':>8}")
        print("-"*table_width)
        
        # Intercept row
        sig_stars = self._get_significance_stars(p_values[0])
        p_formatted = self._format_pvalue(p_values[0])
        print(f"{'(Intercept)':<{var_width}} {all_coefs[0]:>16.4f} {std_errors[0]:>16.4f} {t_stats[0]:>10.4f} {p_formatted:>12} {sig_stars:>8}")
        
        # Feature rows
        for i, feature in enumerate(self.feature_names):
            readable_feature = feature.replace(':', ' × ')
            sig_stars = self._get_significance_stars(p_values[i + 1])
            p_formatted = self._format_pvalue(p_values[i + 1])
            print(f"{readable_feature:<{var_width}} {all_coefs[i+1]:>16.4f} {std_errors[i+1]:>16.4f} {t_stats[i+1]:>10.4f} {p_formatted:>12} {sig_stars:>8}")
        
        print("-"*table_width)
        print("Significance codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        
        # Regression equation
        print("\n" + "-"*table_width)
        print("REGRESSION EQUATION:")
        print("-"*table_width)
        equation = self.get_equation()
        print(f"\n{equation}\n")
        
        # Feature importance
        print("-"*table_width)
        print("FEATURE IMPORTANCE (by absolute coefficient):")
        print("-"*table_width)
        coef_df = self.get_coefficients()
        importance_df = coef_df.copy()
        importance_df['abs_coefficient'] = importance_df['coefficient'].abs()
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
        print(importance_df[['feature', 'coefficient']].to_string(index=False))
        
        print("\nNote: Feature importance assumes features are on similar scales.")
        print("      Consider standardizing features for fair comparison.")
        
        print("\n" + "="*table_width)
        
    def _format_pvalue(self, p_value):
        """
        Format p-value in a readable way (no scientific notation for most values).
        
        Parameters:
        -----------
        p_value : float
            P-value to format
        
        Returns:
        --------
        formatted : str
            Formatted p-value string
        """
        if p_value < 0.001:
            return '<0.001'
        else:
            return f'{p_value:.4f}'

    def _get_significance_stars(self, p_value):
        """
        Get significance stars based on p-value.
        
        Parameters:
        -----------
        p_value : float
            P-value from statistical test
        
        Returns:
        --------
        stars : str
            Significance indicators
        """
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        elif p_value < 0.1:
            return '.'
        else:
            return ''

def fit_lm(df, target=None, features=None, formula=None):
    """
    Fit a linear regression model.
    
    Supports three ways to specify the model:
    1. Auto-detect: Use all features (default)
    2. Feature list: Specify which features to use
    3. Formula: R-style formula for interactions, transformations, etc.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Training data with features and target column
    target : str, optional
        Name of the target column (required unless using formula)
    features : list of str, optional
        List of feature column names to use as predictors.
        If None and no formula, uses all columns except target.
    formula : str, optional
        R-style formula (e.g., 'y ~ x1 + x2' or 'y ~ x1 * x2' for interaction)
        If provided, target and features are ignored.
    
    Returns:
    --------
    model : LinearRegressionModel
        Fitted linear regression model
    
    Examples:
    ---------
    >>> # Use all features
    >>> model = fit_lm(df, target='price')
    >>> 
    >>> # Use specific features only
    >>> model = fit_lm(df, target='price', 
    ...                features=['sqft', 'bedrooms'])
    >>> 
    >>> # Use formula for interaction effects
    >>> model = fit_lm(df, formula='price ~ sqft + bedrooms + sqft:bedrooms')
    >>> 
    >>> # Shorthand for interaction (includes main effects + interaction)
    >>> model = fit_lm(df, formula='price ~ sqft * bedrooms')
    >>> # Equivalent to: price ~ sqft + bedrooms + sqft:bedrooms
    >>> 
    >>> # Use all features with formula
    >>> model = fit_lm(df, formula='price ~ .')
    
    Formula Syntax:
    ---------------
    - y ~ x1 + x2        : Main effects only
    - y ~ x1 * x2        : Main effects + interaction (x1:x2)
    - y ~ x1:x2          : Interaction only (no main effects)
    - y ~ .              : All features
    - y ~ . - x1         : All features except x1
    - y ~ I(x1**2)       : Polynomial term (x1 squared)
    - y ~ np.log(x1)     : Transformed feature
    """
    if formula is not None:
        # Formula-based approach
        return LinearRegressionModel(df, target=None, features=None, formula=formula)
    elif target is None:
        raise ValueError("Must provide either 'target' or 'formula'")
    else:
        # Standard approach
        return LinearRegressionModel(df, target=target, features=features, formula=None)