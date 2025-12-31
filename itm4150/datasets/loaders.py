import pandas as pd
import requests
from io import StringIO

def load_mushroom_data():
    """
    Load the UCI Mushroom dataset.
    
    This dataset contains descriptions of hypothetical samples corresponding to 
    23 species of gilled mushrooms. Each sample is classified as either 
    poisonous (p) or edible (e).
    
    Returns:
    --------
    pd.DataFrame
        Mushroom dataset with 8124 samples and 23 features
        
    Examples:
    ---------
    >>> from itm4150.datasets import load_mushroom_data
    >>> df = load_mushroom_data()
    >>> print(df.shape)
    (8124, 23)
    >>> print(df['class'].value_counts())
    
    References:
    -----------
    https://archive.ics.uci.edu/dataset/73/mushroom
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    
    column_names = [
        'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
        'stalk-surface-below-ring', 'stalk-color-above-ring',
        'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
        'ring-type', 'spore-print-color', 'population', 'habitat'
    ]
    
    try:
        # Try using requests (handles SSL better)
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text), names=column_names)
    except Exception as e:
        print(f"Error loading mushroom data: {e}")
        print("Please ensure you have an internet connection.")
        raise
    
    return df

def load_iris_data():
    """
    Load the classic Iris dataset.
    
    This dataset contains 150 samples of iris flowers with measurements
    of sepal and petal dimensions, classified into 3 species.
    
    Returns:
    --------
    pd.DataFrame
        Iris dataset with 150 samples and 5 columns
        
    Examples:
    ---------
    >>> from itm4150.datasets import load_iris_data
    >>> df = load_iris_data()
    >>> print(df['species'].value_counts())
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    
    column_names = [
        'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'
    ]
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text), names=column_names)
        # Remove any empty rows
        df = df[df['species'].notna()]
    except Exception as e:
        print(f"Error loading iris data: {e}")
        raise
    
    return df


# Module-level convenience dictionary
AVAILABLE_DATASETS = {
    'mushroom': load_mushroom_data,
    'iris': load_iris_data,
    # 'titanic': load_titanic_data,
    # 'sales': get_sample_sales_data,
}


def list_datasets():
    """
    List all available datasets in the itm4150 package.
    
    Returns:
    --------
    list
        Names of available datasets
        
    Examples:
    ---------
    >>> from itm4150.datasets import list_datasets
    >>> print(list_datasets())
    """
    return list(AVAILABLE_DATASETS.keys())


def load_dataset(name):
    """
    Load a dataset by name.
    
    Parameters:
    -----------
    name : str
        Name of the dataset ('mushroom', etc.)
    
    Returns:
    --------
    pd.DataFrame
        Requested dataset
        
    Examples:
    ---------
    >>> from itm4150.datasets import load_dataset
    >>> df = load_dataset('mushroom')
    """
    if name not in AVAILABLE_DATASETS:
        available = ', '.join(AVAILABLE_DATASETS.keys())
        raise ValueError(f"Dataset '{name}' not found. Available datasets: {available}")
    
    return AVAILABLE_DATASETS[name]()