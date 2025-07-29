import pandas as pd
from EDA import intend
from ETL import map_unique_strings
from EDA import convert_float_to_int, handle_nans

# GPU ML libraries
import xgboost as xgb
import lightgbm as lgb
import catboost

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

def gpu_grid_search(X, y, param_grids=None, test_size=0.2, random_state=42, scoring=None, verbose=1):
    """
    Performs grid search for the best GPU-accelerated ML method and its parameters.
    Determines classification or regression using intend() from EDA.py.

    Parameters:
    -----------
    X : pd.DataFrame
        Feature DataFrame (already preprocessed)
    y : pd.Series or np.ndarray
        Target vector
    param_grids : dict or None
        Dictionary with parameter grids for each model. If None, uses defaults.
    test_size : float
        Fraction of data to use as test set
    random_state : int
        Random seed for train/test split
    scoring : str or None
        Scoring metric for GridSearchCV. If None, uses sensible default.
    verbose : int
        Verbosity level

    Returns:
    --------
    model
        The best model, trained on all X and y
    """

    # Determine task type
    if isinstance(X, pd.DataFrame) and isinstance(y, (pd.Series, pd.DataFrame)):
        if isinstance(y, pd.Series):
            y_name = y.name if y.name is not None else "__target__"
            temp_df = X.copy()
            temp_df[y_name] = y
            task = intend(temp_df, y_name)
        else:
            y_name = y.columns[0]
            temp_df = X.copy()
            temp_df[y_name] = y[y_name]
            task = intend(temp_df, y_name)
    else:
        if hasattr(y, "dtype"):
            if pd.api.types.is_integer_dtype(y):
                task = "classification"
            elif pd.api.types.is_float_dtype(y):
                task = "regression"
            else:
                raise ValueError("Cannot infer task type from y's dtype.")
        else:
            raise ValueError("Cannot infer task type from y.")

    if verbose:
        print(f"Detected task: {task}")

    # Default parameter grids
    default_param_grids = {
        'classification': {
            'xgboost': {
                'n_estimators': [40, 60, 100, 200],
                'max_depth': [3, 5, 6, 8, 10],
                'learning_rate': [0.3, 0.1, 0.05, 0.01],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'gamma': [0, 1, 5],
                'min_child_weight': [1, 3, 5],
                'tree_method': ['gpu_hist']
            },
            'lightgbm': {
                'n_estimators': [40, 60, 100, 200],
                'max_depth': [3, 5, 6, 8, -1],
                'learning_rate': [0.3, 0.1, 0.05, 0.01],
                'num_leaves': [15, 31, 63, 127],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'min_child_samples': [10, 20, 50],
                'device': ['gpu']
            },
            'catboost': {
                'iterations': [40, 60, 100, 200],
                'depth': [3, 5, 6, 8, 10],
                'learning_rate': [0.3, 0.1, 0.05, 0.01],
                'l2_leaf_reg': [1, 3, 5, 7, 9],
                'border_count': [32, 64, 128],
                'task_type': ['GPU'],
                'verbose': [0]
            }
        },
        'regression': {
            'xgboost': {
                'n_estimators': [40, 60, 100, 200],
                'max_depth': [3, 5, 6, 8, 10],
                'learning_rate': [0.3, 0.1, 0.05, 0.01],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'gamma': [0, 1, 5],
                'min_child_weight': [1, 3, 5],
                'tree_method': ['gpu_hist']
            },
            'lightgbm': {
                'n_estimators': [40, 60, 100, 200],
                'max_depth': [3, 5, 6, 8, -1],
                'learning_rate': [0.3, 0.1, 0.05, 0.01],
                'num_leaves': [15, 31, 63, 127],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'min_child_samples': [10, 20, 50],
                'device': ['gpu']
            },
            'catboost': {
                'iterations': [40, 60, 100, 200],
                'depth': [3, 5, 6, 8, 10],
                'learning_rate': [0.3, 0.1, 0.05, 0.01],
                'l2_leaf_reg': [1, 3, 5, 7, 9],
                'border_count': [32, 64, 128],
                'task_type': ['GPU'],
                'verbose': [0]
            }
        }
    }

    # Use provided param_grids or defaults
    grids = param_grids if param_grids is not None else default_param_grids[task]

    # Model selection
    if task == 'classification':
        models = {
            'xgboost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', tree_method='gpu_hist', verbosity=0),
            'lightgbm': lgb.LGBMClassifier(device='gpu', verbosity=-1),
            'catboost': catboost.CatBoostClassifier(task_type='GPU', verbose=0)
        }
        default_scoring = 'f1_weighted'
    else:
        models = {
            'xgboost': xgb.XGBRegressor(tree_method='gpu_hist', verbosity=0),
            'lightgbm': lgb.LGBMRegressor(device='gpu', verbosity=-1),
            'catboost': catboost.CatBoostRegressor(task_type='GPU', verbose=0)
        }
        default_scoring = 'neg_root_mean_squared_error'

    # Use provided scoring or sensible default
    scoring_metric = scoring if scoring is not None else default_scoring

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    best_result = {
        'model_name': None,
        'best_estimator': None,
        'best_params': None,
        'best_score': None,
        'test_score': None
    }

    for model_name, model in models.items():
        if verbose:
            print(f"\nGrid searching {model_name}...")
        grid = GridSearchCV(
            estimator=model,
            param_grid=grids[model_name],
            scoring=scoring_metric,
            cv=3,
            n_jobs=-1,
            verbose=verbose
        )
        grid.fit(X_train, y_train)
        best_estimator = grid.best_estimator_
        best_params = grid.best_params_
        best_score = grid.best_score_

        # Evaluate on test set
        y_pred = best_estimator.predict(X_test)
        if task == 'classification':
            test_score = f1_score(y_test, y_pred, average='weighted')
        else:
            test_score = mean_squared_error(y_test, y_pred, squared=False)  # RMSE

        if verbose:
            print(f"Best params for {model_name}: {best_params}")
            print(f"CV best score: {best_score}")
            print(f"Test set score: {test_score}")

        # Update best if better
        if (best_result['best_score'] is None) or \
           ((task == 'classification' and test_score > best_result['test_score']) or
            (task == 'regression' and (best_result['test_score'] is None or test_score < best_result['test_score']))):
            best_result.update({
                'model_name': model_name,
                'best_estimator': best_estimator,
                'best_params': best_params,
                'best_score': best_score,
                'test_score': test_score
            })

    if verbose:
        print(f"\nBest model: {best_result['model_name']}")
        print(f"Best parameters: {best_result['best_params']}")
        print(f"Best CV score: {best_result['best_score']}")
        print(f"Best test set score: {best_result['test_score']}")

    # Retrain the best model on all data
    best_model_name = best_result['model_name']
    best_model_class = models[best_model_name].__class__
    best_params = best_result['best_params']

    # Remove parameters that are not accepted by the constructor (e.g., 'tree_method' for LGBM, etc.)
    # We'll try to only pass parameters that are in the model's __init__ signature
    import inspect
    model_init_params = inspect.signature(best_model_class.__init__).parameters
    filtered_params = {k: v for k, v in best_params.items() if k in model_init_params}

    # Add required fixed params for GPU if not in best_params
    if best_model_name == 'xgboost':
        filtered_params.setdefault('tree_method', 'gpu_hist')
        if task == 'classification':
            filtered_params.setdefault('use_label_encoder', False)
            filtered_params.setdefault('eval_metric', 'logloss')
            filtered_params.setdefault('verbosity', 0)
    elif best_model_name == 'lightgbm':
        filtered_params.setdefault('device', 'gpu')
        filtered_params.setdefault('verbosity', -1)
    elif best_model_name == 'catboost':
        filtered_params.setdefault('task_type', 'GPU')
        filtered_params.setdefault('verbose', 0)

    # Instantiate and fit the best model on all data
    best_model = best_model_class(**filtered_params)
    best_model.fit(X, y)

    return best_model
