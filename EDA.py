import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

def convert_float_to_int(df, columns='all'):
    """
    Converts float columns (or specified columns) to integer type if possible (i.e., if all values are whole numbers or NaN).

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame to process
    columns : str or list of str, default 'all'
        Columns to process. If 'all', processes all float columns

    Returns:
    --------
    pd.DataFrame
        DataFrame with specified columns converted to integer type where possible
    """
    transformed_df = df.copy()
    if columns == 'all':
        cols_to_process = transformed_df.select_dtypes(include=['float']).columns
    else:
        cols_to_process = [columns] if isinstance(columns, str) else columns

    for col in cols_to_process:
        if col not in transformed_df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        # Only convert if all non-NaN values are whole numbers
        col_series = transformed_df[col]
        if col_series.dropna().apply(float.is_integer).all():
            transformed_df[col] = col_series.astype('Int64')
    return transformed_df

def handle_nans(df):
    """
    Process NaN values in a DataFrame:
    - For float columns: Replace NaN with column mean
    - For integer columns: Replace NaN with (max value + 1)
    - Other columns remain unchanged

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame to process

    Returns:
    --------
    pd.DataFrame
        Processed DataFrame with NaN handling
    """
    processed_df = df.copy()
    for col in processed_df.columns:
        if pd.api.types.is_float_dtype(processed_df[col]):
            mean_val = processed_df[col].mean()
            processed_df[col] = processed_df[col].fillna(mean_val)
        elif pd.api.types.is_integer_dtype(processed_df[col]):
            max_val = processed_df[col].max()
            fill_val = max_val + 1 if pd.notna(max_val) else 0
            processed_df[col] = processed_df[col].fillna(fill_val)
    return processed_df

def remove_classification_anomalies(df, y, contamination=0.05, random_state=42):
    """
    Remove classification anomalies from a DataFrame based on the target variable.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing features and target variable
    y : str
        Name of the target variable column
    contamination : float (default=0.05)
        Expected proportion of anomalies in the data
    random_state : int (default=42)
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame
        DataFrame with anomalies removed
    """

    # Make a copy of the original DataFrame to avoid modifying it
    df_clean = df.copy()

    # Separate features and target
    X = df_clean.drop(columns=[y])
    target = df_clean[y]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize and fit Isolation Forest for each class
    classes = target.unique()
    anomaly_mask = pd.Series(False, index=df_clean.index)

    for class_label in classes:
        # Get indices of current class
        class_indices = target[target == class_label].index

        # Fit Isolation Forest on this class's data
        clf = IsolationForest(contamination=contamination,
                            random_state=random_state)
        clf.fit(X_scaled[class_indices])

        # Predict anomalies for this class
        class_pred = clf.predict(X_scaled[class_indices])

        # Update anomaly mask (anomalies are marked as -1)
        anomaly_mask.loc[class_indices] = (class_pred == -1)

    # Remove rows marked as anomalies
    df_clean = df_clean[~anomaly_mask]

    return df_clean.reset_index(drop=True)


def intend(df, y):
    """
    Determines the type of problem based on the dtype of column y:
    - Returns 'classification' if the column is integer type
    - Returns 'regression' if the column is float type

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to check.
    y : str
        The column name to check.

    Returns:
    --------
    str
        'classification' if the column is integer type, 'regression' if float type.
    """
    if y not in df.columns:
        raise ValueError(f"Column '{y}' not found in DataFrame")
    if pd.api.types.is_integer_dtype(df[y]):
        return 'classification'
    if pd.api.types.is_float_dtype(df[y]):
        return 'regression'
    raise ValueError(f"Column '{y}' is neither integer nor float dtype")
    

