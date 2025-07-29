import pandas as pd

# ETL methods
from ETL import map_unique_strings, encode, decode

# EDA methods
from EDA import convert_float_to_int, handle_nans

# Visualization methods
from visualization import corr_matrix, nans_look

# Model selection
from grid_search import grid_search

def intend(y: pd.Series) -> str:
    """
    Infers the ML task type ('classification' or 'regression') from the target variable.
    """
    if y.dtype.kind in 'biu':  # integer types
        n_unique = y.nunique(dropna=True)
        if n_unique <= 20:
            return 'classification'
        else:
            return 'regression'
    elif y.dtype.kind == 'f':  # float types
        n_unique = y.nunique(dropna=True)
        if n_unique <= 20 and (y.dropna() == y.dropna().astype(int)).all():
            return 'classification'
        else:
            return 'regression'
    else:
        return 'classification'

def ezml_pipeline(
    df: pd.DataFrame,
    target: str,
    model_dict: dict = None,
    param_grids: dict = None,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
    visualize: bool = True,
    string_columns: 'str|list' = 'all'
):
    """
    End-to-end ML pipeline using ETL, EDA, visualization, and model selection.

    Args:
        df: Input DataFrame
        target: Name of target column
        model_dict: Dictionary of models to use (optional)
        param_grids: Dictionary of parameter grids for each model (optional)
        test_size: Fraction for test split
        random_state: Random seed
        verbose: Print progress
        visualize: Show visualizations
        string_columns: Columns to encode as categorical (default 'all')

    Returns:
        best_model: Trained best model on all data
        mappings: Dict of mappings for categorical columns
        X: Final feature DataFrame
        y: Final target Series
        task: Inferred task type ('classification' or 'regression')
    """
    df = df.copy()

    if verbose:
        print("Step 1: Visualizing missing values and correlations...")
    if visualize:
        nans_look(df)
        try:
            from IPython.display import display
            display(corr_matrix(df))
        except ImportError:
            print("Install IPython to display styled correlation matrix.")

    if verbose:
        print("Step 2: Handling NaNs...")
    df = handle_nans(df)

    if verbose:
        print("Step 3: Converting float columns to int where possible...")
    df = convert_float_to_int(df)

    if verbose:
        print("Step 4: Encoding string columns...")
    df_encoded, mappings = map_unique_strings(df, columns=string_columns, return_mappings=True)

    # Separate features and target
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]

    # Infer task type
    task = intend(y)
    if verbose:
        print(f"Step 5: Inferred task type: {task}")

    if verbose:
        print("Step 6: Model selection and training...")
    best_model = grid_search(
        X, y,
        task=task,
        model_dict=model_dict,
        param_grids=param_grids,
        test_size=test_size,
        random_state=random_state,
        verbose=verbose
    )

    if verbose:
        print("Pipeline complete.")

    return best_model, mappings, X, y, task