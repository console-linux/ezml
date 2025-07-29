import pandas as pd
from typing import Union, List
def map_unique_strings(df: pd.DataFrame,
                       columns: Union[str, List[str]] = 'all',
                       return_mappings: bool = False) -> Union[pd.DataFrame, tuple]:
    """
    Processes DataFrame columns containing string objects by creating and applying unique mappings.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame to process
    columns : str or list of str, default 'all'
        Columns to process. If 'all', processes all object/string columns
    return_mappings : bool, default False
        Whether to return the mapping dictionaries along with the transformed DataFrame

    Returns:
    --------
    pd.DataFrame or tuple (pd.DataFrame, dict)
        Transformed DataFrame, optionally with mapping dictionaries if return_mappings=True
    """

    # Initialize mapping storage
    mappings = {}

    # Determine which columns to process
    if columns == 'all':
        # Select all object/string columns
        cols_to_process = df.select_dtypes(include=['object', 'string']).columns
    else:
        # Process specified columns (ensure it's a list even if single column provided)
        cols_to_process = [columns] if isinstance(columns, str) else columns

    # Create a copy of the DataFrame to avoid modifying the original
    transformed_df = df.copy()

    for col in cols_to_process:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

        # Get unique values and create mapping
        unique_values = transformed_df[col].unique()
        # Create mapping from value to integer code, ignoring NaN values
        value_map = {val: idx for idx, val in enumerate(unique_values) if pd.notna(val)}

        # Apply mapping to the column
        transformed_df[col] = transformed_df[col].map(value_map)

        # Store the mapping if requested
        if return_mappings:
            mappings[col] = value_map

    if return_mappings:
        return transformed_df, mappings
    else:
        return transformed_df