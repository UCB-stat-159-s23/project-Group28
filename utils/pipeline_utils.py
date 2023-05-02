from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd


def create_pipelines():
    """
    Create a dictionary of pipelines with combinations of different imputers and models.
    
    Returns:
        dict: A dictionary of pipelines with keys as imputer+model names and values as pipeline objects.
    """

    imputers = {
        "simple_imputer": SimpleImputer(strategy="most_frequent"),
        "knn_imputer": KNNImputer()
    }
    models = {
        "rf": RandomForestRegressor(n_estimators=100, min_samples_leaf=5),
        "lasso": Lasso(),
        "ridge": Ridge()
    }
    pipes = {}
    for imputer_name, imputer in imputers.items():
        for model_name, model in models.items():
            pipe = Pipeline(steps=[(imputer_name, imputer), (model_name, model)])
            pipe_name = imputer_name + "+" + model_name
            pipes[pipe_name] = pipe
    
    return pipes


def create_summary(valid_errs, y_valid, ypred_valid):
    """
    Create a summary table of validation errors, mean squared error (MSE), and mean absolute error (MAE) for different models.
    
    Args:
        valid_errs (dict): A dictionary containing validation errors for different models.
        y_valid (pd.Series): The true target values for validation data.
        ypred_valid (dict): A dictionary containing predicted target values for validation data from different models.
        
    Returns:
        pd.DataFrame: A summary table with model names, validation errors, MSE, and MAE.
    """
    summary = pd.DataFrame.from_dict(valid_errs, columns=["Valid Errors"], orient='index')
    MSE = [mean_squared_error(y_valid, ypred_valid[col]) for col in ypred_valid]
    MAE = [mean_absolute_error(y_valid, ypred_valid[col]) for col in ypred_valid]
    summary.index.name = 'Model'
    summary["MSE"] = MSE
    summary["MAE"] = MAE
    return summary


def calculate_metrics(y_true, y_pred):
    """
    Calculate mean squared error (MSE), mean absolute error (MAE), and R-squared score for the given true and predicted target values.
    
    Args:
        y_true (pd.Series): The true target values.
        y_pred (pd.Series): The predicted target values.
        
    Returns:
        tuple: A tuple containing MSE, MAE, and R-squared score.
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return mse, mae, r2

def display_results(mse, mae, r2, model_name='knn_imputer+rf'):
    """
    Create a table to display evaluation metrics for a given model.
    
    Args:
        mse (float): Mean squared error.
        mae (float): Mean absolute error.
        r2 (float): R-squared score.
        model_name (str, optional): The name of the model. Defaults to 'knn_imputer+rf'.
        
    Returns:
        pd.DataFrame: A table displaying evaluation metrics for the given model.
    """
    results = {
        'R-squared': [r2],
        'MSE': [mse],
        'MAE': [mae]
    }

    results_df = pd.DataFrame(results, index=[model_name])
    results_df.index.name = 'Model'
    return results_df