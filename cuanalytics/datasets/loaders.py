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
    >>> from cuanalytics.datasets import load_mushroom_data
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
    >>> from cuanalytics.datasets import load_iris_data
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

def load_breast_cancer_data():
    """
    Load the Breast Cancer Wisconsin (Diagnostic) dataset.
    
    This dataset contains features computed from digitized images of 
    fine needle aspirate (FNA) of breast masses. Features describe 
    characteristics of cell nuclei present in the images.
    
    The target variable is 'diagnosis':
    - 'M' = Malignant (cancerous)
    - 'B' = Benign (non-cancerous)
    
    Features include measurements like:
    - radius, texture, perimeter, area, smoothness
    - compactness, concavity, concave points, symmetry, fractal dimension
    
    Each feature has three versions: mean, standard error (se), and worst (largest).
    For example: radius_mean, radius_se, radius_worst
    
    Returns:
    --------
    df : pd.DataFrame
        DataFrame with 569 samples and 31 columns (30 features + diagnosis)
    
    Examples:
    ---------
    >>> from cuanalytics import load_breast_cancer_data, fit_lda, split_data
    >>> df = load_breast_cancer_data()
    >>> print(df.shape)
    (569, 31)
    >>> print(df['diagnosis'].value_counts())
    >>> train, test = split_data(df, test_size=0.2)
    >>> lda = fit_lda(train, formula='diagnosis ~ .')
    
    Dataset Information:
    --------------------
    Source: UCI Machine Learning Repository
    URL: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
    Samples: 569
    Features: 30 (all numeric, real-valued)
    Classes: 2 (Malignant, Benign)
    Missing values: None
    
    Feature Groups (each has mean, se, worst):
    - radius: distances from center to points on perimeter
    - texture: standard deviation of gray-scale values
    - perimeter: size of core tumor
    - area: area of tumor
    - smoothness: local variation in radius lengths
    - compactness: perimeter^2 / area - 1.0
    - concavity: severity of concave portions of contour
    - concave points: number of concave portions of contour
    - symmetry: symmetry of the tumor
    - fractal dimension: "coastline approximation" - 1
    
    Citation:
    ---------
    William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian.
    "Breast Cancer Wisconsin (Diagnostic) Data Set."
    UCI Machine Learning Repository, 1995.
    """
    from sklearn.datasets import load_breast_cancer
    import pandas as pd
    
    # Load from sklearn (which gets it from UCI)
    data = load_breast_cancer()
    
    # Create DataFrame with feature names
    df = pd.DataFrame(data.data, columns=data.feature_names)
    
    # Add target column
    # sklearn uses 0/1, convert to M/B for clarity
    df['diagnosis'] = ['M' if target == 0 else 'B' for target in data.target]
    
    return df



# Add to cuanalytics/datasets/loaders.py

def load_real_estate_data():
    """
    Load the Real Estate Valuation dataset from UCI.
    
    This dataset contains real estate valuation data from Sindian District,
    New Taipei City, Taiwan. The target is house price per unit area.
    
    Features:
    - transaction_date: Transaction date (e.g., 2013.250 = March 2013)
    - house_age: House age in years
    - distance_to_MRT: Distance to nearest MRT station in meters
    - num_convenience_stores: Number of convenience stores nearby
    - latitude: Latitude coordinate
    - longitude: Longitude coordinate
    
    Target:
    - price_per_unit: House price per unit area (10000 New Taiwan Dollar/Ping)
                      Note: 1 Ping = 3.3 square meters
    
    Returns:
    --------
    df : pd.DataFrame
        DataFrame with 414 samples and 7 columns (6 features + target)
    
    Examples:
    ---------
    >>> from cuanalytics import load_real_estate_data, fit_lm, split_data
    >>> df = load_real_estate_data()
    >>> print(df.shape)
    (414, 7)
    >>> print(df.head())
    >>> train, test = split_data(df, test_size=0.2)
    >>> model = fit_lm(train, formula='price_per_unit ~ .')
    >>> model.summary()
    
    Dataset Information:
    --------------------
    Source: UCI Machine Learning Repository
    URL: https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set
    Location: Sindian District, New Taipei City, Taiwan
    Time period: 2012.667 - 2013.583 (August 2012 - July 2013)
    Samples: 414
    Features: 6 (all numeric)
    Target: Continuous (price per unit area)
    Missing values: None
    
    Citation:
    ---------
    Yeh, I-Cheng. (2018). Real Estate Valuation.
    UCI Machine Learning Repository.
    https://doi.org/10.24432/C5J30W
    """
    import pandas as pd
    import io
    import urllib.request
    import ssl
    
    # URL for the dataset
    url = "https://archive.ics.uci.edu/static/public/477/real+estate+valuation+data+set.zip"
    
    try:
        # Create SSL context that doesn't verify certificates
        ssl_context = ssl._create_unverified_context()
        
        print("Downloading Real Estate Valuation dataset from UCI...")
        
        # Download the zip file
        with urllib.request.urlopen(url, context=ssl_context) as response:
            zip_data = response.read()
        
        # Extract the Excel file from the zip
        import zipfile
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zip_file:
            # The dataset is in an Excel file
            excel_filename = 'Real estate valuation data set.xlsx'
            with zip_file.open(excel_filename) as excel_file:
                # Read the Excel file (skip first row which is just a title)
                df = pd.read_excel(excel_file, header=1)
        
        print("Dataset loaded successfully!")
        
    except Exception as e:
        print(f"\nError downloading real estate data: {e}")
        print("\nFalling back to synthetic sales dataset...")
        print("If you need the real estate data, try downloading manually from:")
        print("https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set")
        return 
    
    # Clean up column names
    df.columns = [
        'No',  # Row number (we'll drop this)
        'transaction_date',
        'house_age',
        'distance_to_MRT',
        'num_convenience_stores',
        'latitude',
        'longitude',
        'price_per_unit'
    ]
    
    # Drop the row number column
    df = df.drop('No', axis=1)
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df

# Module-level convenience dictionary
AVAILABLE_DATASETS = {
    'mushroom': load_mushroom_data,
    'breast_cancer': load_breast_cancer_data,
    'iris': load_iris_data,
    'real_estate': load_real_estate_data,
    # 'titanic': load_titanic_data,
    # 'sales': get_sample_sales_data,
}


def list_datasets():
    """
    List all available datasets in the cuanalytics package.
    
    Returns:
    --------
    list
        Names of available datasets
        
    Examples:
    ---------
    >>> from cuanalytics.datasets import list_datasets
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
    >>> from cuanalytics.datasets import load_dataset
    >>> df = load_dataset('mushroom')
    """
    if name not in AVAILABLE_DATASETS:
        available = ', '.join(AVAILABLE_DATASETS.keys())
        raise ValueError(f"Dataset '{name}' not found. Available datasets: {available}")
    
    return AVAILABLE_DATASETS[name]()
