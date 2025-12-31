from sklearn.model_selection import train_test_split

def split_data(df, target=None, test_size=0.2, val_size=None, random_state=42, stratify=True):
    """
    Split a DataFrame into train/test or train/test/validation sets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to split
    target : str, optional
        Target column name. If provided and stratify=True, maintains class proportions.
    test_size : float
        Proportion of data for test set (0.0 to 1.0)
    val_size : float, optional
        Proportion of data for validation set (0.0 to 1.0)
        If None, only creates train/test split
        If provided, creates train/test/val split
    random_state : int
        Random seed for reproducibility
    stratify : bool
        If True and target is provided, maintains class proportions in splits
    
    Returns:
    --------
    train_df : pd.DataFrame
        Training set (always returned)
    test_df : pd.DataFrame
        Test set (always returned)
    val_df : pd.DataFrame, optional
        Validation set (only if val_size is specified)
    
    Examples:
    ---------
    # Simple train/test split (80/20)
    >>> train, test = split_data(df, target='class')
    
    # Custom proportions
    >>> train, test = split_data(df, target='class', test_size=0.3)
    
    # Train/test/validation split (60/20/20)
    >>> train, test, val = split_data(df, target='class', test_size=0.2, val_size=0.2)
    
    # Without stratification
    >>> train, test = split_data(df, stratify=False)
    
    # No target column
    >>> train, test = split_data(df, test_size=0.25)
    """
    # Validation
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1")
    
    if val_size is not None:
        if val_size <= 0 or val_size >= 1:
            raise ValueError("val_size must be between 0 and 1")
        if test_size + val_size >= 1:
            raise ValueError("test_size + val_size must be less than 1")
    
    # Determine stratification
    stratify_col = None
    if stratify and target is not None:
        stratify_col = df[target]
    
    # Two-way split (train/test)
    if val_size is None:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col
        )
        
        print(f"Split: {len(train_df)} train ({len(train_df)/len(df)*100:.1f}%), "
              f"{len(test_df)} test ({len(test_df)/len(df)*100:.1f}%)")
        
        if target and stratify:
            print(f"\nTrain class distribution:")
            print(train_df[target].value_counts(normalize=True))
            print(f"\nTest class distribution:")
            print(test_df[target].value_counts(normalize=True))
        
        return train_df, test_df
    
    # Three-way split (train/test/val)
    else:
        # First split: separate out test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col
        )
        
        # Second split: split remaining into train and validation
        # Adjust validation size relative to remaining data
        val_size_adjusted = val_size / (1 - test_size)
        
        stratify_col_remaining = None
        if stratify and target is not None:
            stratify_col_remaining = train_val_df[target]
        
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=stratify_col_remaining
        )
        
        print(f"Split: {len(train_df)} train ({len(train_df)/len(df)*100:.1f}%), "
              f"{len(test_df)} test ({len(test_df)/len(df)*100:.1f}%), "
              f"{len(val_df)} validation ({len(val_df)/len(df)*100:.1f}%)")
        
        if target and stratify:
            print(f"\nTrain class distribution:")
            print(train_df[target].value_counts(normalize=True))
            print(f"\nTest class distribution:")
            print(test_df[target].value_counts(normalize=True))
            print(f"\nValidation class distribution:")
            print(val_df[target].value_counts(normalize=True))
        
        return train_df, test_df, val_df