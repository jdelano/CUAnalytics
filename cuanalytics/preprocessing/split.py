from sklearn.model_selection import train_test_split
import pandas as pd
# cuanalytics/preprocessing/split.py

def split_data(df, test_size=0.2, val_size=None, random_state=42, stratify_on=None):
    """
    Split data into training and test sets, optionally with validation set.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to split
    test_size : float
        Proportion of data for test set (0.0 to 1.0)
    val_size : float, optional
        Proportion of data for validation set (0.0 to 1.0)
        If None, only returns train/test split
        If provided, returns train/test/val split
    random_state : int
        Random seed for reproducibility
    stratify_on : str, optional
        Column name to use for stratified splitting
        - None (default): Random split, no stratification
        - 'column_name': Stratify by this column (for classification)
    
    Returns:
    --------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Test data
    val_df : pd.DataFrame (only if val_size is provided)
        Validation data
    
    Examples:
    ---------
    >>> # Two-way random split (80/20)
    >>> train, test = split_data(df, test_size=0.2)
    >>> 
    >>> # Two-way stratified split (80/20, stratified by 'species')
    >>> train, test = split_data(df, test_size=0.2, stratify_on='species')
    >>> 
    >>> # Three-way random split (60/20/20)
    >>> train, test, val = split_data(df, test_size=0.2, val_size=0.2)
    >>> 
    >>> # Three-way stratified split (60/20/20)
    >>> train, test, val = split_data(df, test_size=0.2, val_size=0.2, 
    ...                                stratify_on='species')
    """
    from sklearn.model_selection import train_test_split
    
    # Validate inputs
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1 (exclusive)")
    
    if val_size is not None:
        if val_size <= 0 or val_size >= 1:
            raise ValueError("val_size must be between 0 and 1 (exclusive)")
        
        if test_size + val_size >= 1:
            raise ValueError(
                f"test_size + val_size must be less than 1. "
                f"Got test_size={test_size}, val_size={val_size}, sum={test_size + val_size}"
            )
    
    # Check if stratify_on column exists
    if stratify_on is not None and stratify_on not in df.columns:
        raise KeyError(f"Stratification column '{stratify_on}' not found in DataFrame")
    
    # Determine stratification
    stratify_col = df[stratify_on] if stratify_on is not None else None
    
    # Two-way split
    if val_size is None:
        try:
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_col
            )
        except ValueError as e:
            # If stratification fails (e.g., continuous target), warn and fall back
            if stratify_on is not None and "least populated classes" in str(e):
                import warnings
                warnings.warn(
                    f"Stratification on '{stratify_on}' failed (likely continuous values). "
                    f"Falling back to random split.",
                    UserWarning
                )
                train_df, test_df = train_test_split(
                    df,
                    test_size=test_size,
                    random_state=random_state
                )
            else:
                raise
        
        return train_df, test_df
    
    # Three-way split
    else:
        # Calculate the proportion of validation from the remaining data after test split
        val_size_adjusted = val_size / (1 - test_size)
        
        try:
            # First split: separate test set
            train_val_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=df[stratify_on] if stratify_on else None
            )
            
            # Second split: separate validation from training
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=train_val_df[stratify_on] if stratify_on else None
            )
        except ValueError as e:
            # If stratification fails, fall back to random
            if stratify_on is not None and "least populated classes" in str(e):
                import warnings
                warnings.warn(
                    f"Stratification on '{stratify_on}' failed (likely continuous values). "
                    f"Falling back to random split.",
                    UserWarning
                )
                train_val_df, test_df = train_test_split(
                    df,
                    test_size=test_size,
                    random_state=random_state
                )
                train_df, val_df = train_test_split(
                    train_val_df,
                    test_size=val_size_adjusted,
                    random_state=random_state
                )
            else:
                raise
        
        return train_df, test_df, val_df