"""
Custom functions to evaluate, print and log (using mlflow) training scores for different models.

Author: Tom Fleet
Created: 30/05/2020
"""


import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


def score_model(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray) -> str:
    """
    Runs a cross_val_score with cv = 5 on arrays X, y with a neg mean squared error score.
    Performs the RMSE conversion and prints out scores.

    Args:
        estimator (BaseEstimator): Trained sklearn estimator object (Regressor)
        X (np.ndarray): Feature array
        y (np.ndarray): Target array

    Returns:
        no_val_rmse: [np.float64] RMSE score based on the training data
        no_val_r2: [np.float64] R^2 score based on the training data
        val_rmse_scores: [np.ndarray] Series of RMSE scores from cross validation
        cv_mean: [np.float64] Mean of all cross-validated RMSE scores
        cv_std: [np.float64] StDev of all cross-validated RMSE scores
        cv_cov: [np.float64] CoV of all cross-validated RMSE scores (CoV = StDev / Mean)
    """

    val_scores = cross_val_score(estimator, X, y, scoring="neg_mean_squared_error")
    val_scores = val_scores * -1
    val_rmse_scores = np.sqrt(val_scores)

    no_val_mse = mean_squared_error(y, estimator.predict(X))
    no_val_rmse = np.sqrt(no_val_mse)
    no_val_r2 = r2_score(y, estimator.predict(X))

    cv_mean = np.mean(val_rmse_scores)
    cv_std = np.std(val_rmse_scores)
    cv_cov = cv_std / cv_mean

    print("Non-validation Scores")
    print("-----------")
    print(f"RMSE (No Val): {np.round(no_val_rmse, 3)}")
    print(f"R^2 (No Val): {np.round(no_val_r2, 3)}")
    print()
    print("Validation Scores")
    print("-----------")
    print(f"RMSE's: {np.round(val_rmse_scores, 3)}")
    print(f"Mean: {np.round(cv_mean, 3)}")
    print(f"StDev: {np.round(cv_std, 3)}")
    print(f"CoV: {np.round(cv_cov, 3)}")

    return no_val_rmse, no_val_r2, val_rmse_scores, cv_mean, cv_std, cv_cov


def auto_mlflow(
    run_name: str,
    model_name: BaseEstimator,
    data_params: dict = None,
    X: np.ndarray = "X_train",
    y: np.ndarray = "y_train",
) -> str:
    """
    Wrapper function that automates the application of mlflow to a model training event.

    Args:
        run_name (str): Desired name of the run, this will appear in the database
        model_name (BaseEstimator): Variable name of the sklearn estimator object
                                    (must refer to an already instantiated model)
        data_params (dict, optional): Dictionary containing params on the data
                                    e.g. {'standard_scaled': False}. Defaults to None.
        X (np.ndarray, optional): Feature array. Defaults to "X_train".
        y (np.ndarray, optional): Target array. Defaults to "y_train".

    Returns:
        str: Logs data to mlflow, also prints representation of evaluation scores to console
    """

    with mlflow.start_run(run_name=run_name):

        model_name.fit(X, y)

        no_val_rmse, no_val_r2, val_rmse_scores, cv_mean, cv_std, cv_cov = score_model(
            model_name, X, y
        )

        data_params = data_params
        model_params = model_name.get_params()

        mlflow.log_params(data_params)
        mlflow.log_params(model_params)

        mlflow.log_metrics(
            {
                "no_val_rmse": no_val_rmse,
                "no_val_r2": no_val_r2,
                "cv_score_1": val_rmse_scores[0],
                "cv_score_2": val_rmse_scores[1],
                "cv_score_3": val_rmse_scores[2],
                "cv_score_4": val_rmse_scores[3],
                "cv_score_5": val_rmse_scores[4],
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "cv_cov": cv_cov,
            }
        )

        mlflow.sklearn.log_model(model_name, "model")

    return None
