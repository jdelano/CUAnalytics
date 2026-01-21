# cuanalytics/regression/linear.py
"""
Linear regression implementation for ITM 4150.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from cuanalytics.formula import ModelFormula

class LinearRegressionModel:
    """
    Linear Regression model for predicting continuous outcomes.
    
    This class wraps scikit-learn's LinearRegression to provide a student-friendly
    interface for learning about linear regression.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and target
    formula : str
        R-style formula for specifying the model
    """
    
    def __init__(self, df, formula):
        self.original_df = df  # Keep original for reference
        self.formula = formula

        self.model_formula = ModelFormula(df, formula, drop_intercept=True)
        self.model_formula.validate_regression_target(df)

        self.model_formula.validate_numeric_features(df)

        # Create and fit model
        try:
            self._fit()
        except Exception as e:
            raise RuntimeError(f"Failed to fit linear regression model: {str(e)}")
        

    
    def _fit(self):
        """Fit the linear regression model."""
        self.model = LinearRegression()
        self.model.fit(self.model_formula.X, self.model_formula.y)
        
    def _check_fitted(self):
        """Check if the model has been fitted."""
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError(
                "Model has not been fitted yet. "
                "Create model with: model = fit_lm(df, formula='y ~ x1 + x2')"
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
        if getattr(self, 'model_formula', None) is None or self.model_formula.model_spec is None:
            raise RuntimeError("Formula metadata missing; cannot transform new data.")

        return self.model_formula.transform(df)

    @property
    def model_spec(self):
        return self.model_formula.model_spec

    @model_spec.setter
    def model_spec(self, value):
        self.model_formula.model_spec = value

    @property
    def target(self):
        return self.model_formula.target

    @property
    def feature_names(self):
        return list(self.model_formula.X.columns)

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
        
        X = self._transform_data_with_formula(df)
        predictions = self.model.predict(X)
        return pd.Series(predictions, index=df.index, name=self.target)
    
    def score(self, df):
        """
        Calculate regression metrics on a dataset.
        
        Returns:
        --------
        metrics : dict
            Dictionary with R², RMSE, and MAE
        """
        self._check_fitted()
        
        X = self._transform_data_with_formula(df)
        y_true = df[self.target]
        y_pred = self.model.predict(X)

        return {
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
        }
    
    def get_metrics(self):
        """
        Calculate regression metrics and model statistics from training data.

        Returns:
        --------
        metrics : dict
            Dictionary with model metrics and statistics
        """
        self._check_fitted()

        from scipy import stats

        y_pred = self.model.predict(self.model_formula.X)
        residuals = self.model_formula.y - y_pred
        n = len(self.model_formula.y)
        k = len(self.feature_names)

        r2 = r2_score(self.model_formula.y, y_pred)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        rmse = np.sqrt(mean_squared_error(self.model_formula.y, y_pred))
        mae = mean_absolute_error(self.model_formula.y, y_pred)

        rse = np.sqrt(np.sum(residuals**2) / (n - k - 1))
        X_with_intercept = np.column_stack([np.ones(n), self.model_formula.X])
        try:
            var_covar = rse**2 * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            std_errors = np.sqrt(np.diag(var_covar))
        except np.linalg.LinAlgError:
            std_errors = np.ones(k + 1) * rse / np.sqrt(n)

        all_coefs = np.concatenate([[self.model.intercept_], self.model.coef_])
        t_stats = all_coefs / std_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k - 1))

        t_critical = stats.t.ppf(0.975, df=n - k - 1)
        conf_lower = all_coefs - t_critical * std_errors
        conf_upper = all_coefs + t_critical * std_errors

        ss_total = np.sum((self.model_formula.y - self.model_formula.y.mean())**2)
        ss_regression = np.sum((y_pred - self.model_formula.y.mean())**2)
        ss_residual = np.sum(residuals**2)

        ms_regression = ss_regression / k
        ms_residual = ss_residual / (n - k - 1)

        f_statistic = ms_regression / ms_residual
        f_pvalue = 1 - stats.f.cdf(f_statistic, k, n - k - 1)

        readable_features = [f.replace(':', ' × ') for f in self.feature_names]
        coef_rows = [{
            'feature': '(Intercept)',
            'coefficient': all_coefs[0],
            'std_error': std_errors[0],
            't_stat': t_stats[0],
            'p_value': p_values[0],
            'conf_lower': conf_lower[0],
            'conf_upper': conf_upper[0]
        }]

        for i, feature in enumerate(readable_features, start=1):
            coef_rows.append({
                'feature': feature,
                'coefficient': all_coefs[i],
                'std_error': std_errors[i],
                't_stat': t_stats[i],
                'p_value': p_values[i],
                'conf_lower': conf_lower[i],
                'conf_upper': conf_upper[i]
            })

        metrics = {
            'model_type': 'linear_regression',
            'target': self.target,
            'n_samples': n,
            'n_features': k,
            'feature_names': self.feature_names,
            'metrics': {
                'r2': r2,
                'adj_r2': adj_r2,
                'rmse': rmse,
                'mae': mae,
                'f_statistic': f_statistic,
                'f_pvalue': f_pvalue
            },
            'coefficients': pd.DataFrame(coef_rows),
            'equation': self.get_equation()
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
        y_pred = self.model.predict(self.model_formula.X)
        residuals = self.model_formula.y - y_pred
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        
        # Plot 1: Predicted vs Actual
        ax1.scatter(self.model_formula.y, y_pred, alpha=0.6, edgecolors='black', s=50)
        
        # Add perfect prediction line
        min_val = min(self.model_formula.y.min(), y_pred.min())
        max_val = max(self.model_formula.y.max(), y_pred.max())
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
        feature_values = np.linspace(self.model_formula.X[feature].min(), self.model_formula.X[feature].max(), 100)
        
        # Create prediction DataFrame with other features at mean
        X_pred = pd.DataFrame(index=range(100))
        
        for feat in self.feature_names:
            if feat == feature:
                X_pred[feat] = feature_values
            else:
                mean_val = self.model_formula.X[feat].mean()
                # Check for NaN (shouldn't happen with valid data, but just in case)
                if pd.isna(mean_val):
                    mean_val = self.model_formula.X[feat].median()  # Fall back to median
                X_pred[feat] = mean_val
        
        # Predict using the DataFrame
        y_pred = self.model.predict(X_pred)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Actual data points
        ax.scatter(self.model_formula.X[feature], self.model_formula.y, alpha=0.6, s=50, 
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
        >>> model = fit_lm(train, formula='monthly_sales ~ .')
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
            ax.scatter(self.model_formula.X[feature], self.model_formula.y, alpha=0.6, s=30, 
                    edgecolors='black', linewidth=0.5)
            
            # Add trend line
            z = np.polyfit(self.model_formula.X[feature], self.model_formula.y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(self.model_formula.X[feature].min(), self.model_formula.X[feature].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
            
            # Calculate and display correlation
            corr = self.model_formula.X[feature].corr(self.model_formula.y)
            
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
            corr = self.model_formula.X[feature].corr(self.model_formula.y)
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
        None
        
        Examples:
        ---------
        >>> model = fit_lm(train, formula='price ~ .')
        >>> summary = model.summary()
        """
        self._check_fitted()
        
        from scipy import stats
        
        # Calculate metrics
        y_pred = self.model.predict(self.model_formula.X)
        residuals = self.model_formula.y - y_pred
        n = len(self.model_formula.y)
        k = len(self.feature_names)  # Number of predictors
        
        # Calculate R², adjusted R², RMSE, MAE
        r2 = r2_score(self.model_formula.y, y_pred)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        rmse = np.sqrt(mean_squared_error(self.model_formula.y, y_pred))
        mae = mean_absolute_error(self.model_formula.y, y_pred)
        
        # Calculate standard errors and t-statistics
        # Residual standard error
        rse = np.sqrt(np.sum(residuals**2) / (n - k - 1))
        
        # Variance-covariance matrix
        X_with_intercept = np.column_stack([np.ones(n), self.model_formula.X])
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
        ss_total = np.sum((self.model_formula.y - self.model_formula.y.mean())**2)
        ss_regression = np.sum((y_pred - self.model_formula.y.mean())**2)
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

def fit_lm(df, formula):
    """
    Fit a linear regression model.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Training data with features and target column
    formula : str
        R-style formula (e.g., 'y ~ x1 + x2' or 'y ~ x1 * x2' for interaction)
    
    Returns:
    --------
    model : LinearRegressionModel
        Fitted linear regression model
    
    Examples:
    ---------
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
    if formula is None:
        raise ValueError("Must provide 'formula' for model specification")
    return LinearRegressionModel(df, formula=formula)
