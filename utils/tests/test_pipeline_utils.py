import numpy as np
import pandas as pd
from utils.pipeline_utils import create_pipelines, create_summary, calculate_metrics, display_results

def test_create_pipelines():
    pipes = create_pipelines()
    
    assert isinstance(pipes, dict), "Returned object is not a dictionary"

    expected_keys = [
        "simple_imputer+rf",
        "simple_imputer+lasso",
        "simple_imputer+ridge",
        "knn_imputer+rf",
        "knn_imputer+lasso",
        "knn_imputer+ridge",
    ]
    for key in expected_keys:
        assert key in pipes, f"{key} not found in the returned dictionary"

    # Check if all dictionary values are of type Pipeline
    from sklearn.pipeline import Pipeline
    for value in pipes.values():
        assert isinstance(value, Pipeline), f"{value} is not of type Pipeline"



def test_create_summary():
    valid_errs = {"model1": 0.9, "model2": 0.8}
    y_valid = [1, 2, 3]
    ypred_valid = {"model1": [0.8, 2.1, 3.2], "model2": [1.2, 2.3, 2.9]}

    summary = create_summary(valid_errs, y_valid, ypred_valid)

    assert isinstance(summary, pd.DataFrame), "Returned object is not a DataFrame"

    # Check if the expected columns are present in the summary
    expected_columns = ["Valid Errors", "MSE", "MAE"]
    for col in expected_columns:
        assert col in summary.columns, f"{col} not found in the returned DataFrame"

    # Check if the number of rows in the summary is correct
    assert summary.shape[0] == len(valid_errs), "Incorrect number of rows in the summary"


def test_calculate_metrics():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    mse, mae, r2 = calculate_metrics(y_true, y_pred)
    
    assert np.isclose(mse, 0.375, atol=1e-8)
    assert np.isclose(mae, 0.5, atol=1e-8)
    assert np.isclose(r2, 0.9486081370449679, atol=1e-8)

def test_display_results():
    mse, mae, r2 = 0.375, 0.5, 0.9486081370449679
    results_df = display_results(mse, mae, r2)
    
    assert results_df.loc['knn_imputer+rf', 'R-squared'] == r2
    assert results_df.loc['knn_imputer+rf', 'MSE'] == mse
    assert results_df.loc['knn_imputer+rf', 'MAE'] == mae
