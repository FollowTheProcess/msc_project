"""
Refactored final model implementation.

Author: Tom Fleet
Created: 23/06/2020
"""

from pathlib import Path
from typing import BinaryIO, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODEL_PARAMS, PROJECT_ROOT


class NotTrainedError(BaseException):
    # Custom exception to warn of innapropriate actions on an untrained model.

    def __init__(
        self, message="The model is not yet trained, run Model.train and try again"
    ):
        self.message = message
        super().__init__(self.message)


class CrackDepthPredictor:
    """
    Class implementation of the final chosen model.

    Trains on the entire original data set.

    Also provides API for testing the model against new data in the future as well as cross validation.
    """

    # Collect model params from config file to avoid hardcoding
    # Also avoids relying on defaults for futureproofing
    params = MODEL_PARAMS

    def __init__(self):

        self.model = None
        self.preprocessor = None
        self.is_trained = False

    def __repr__(self):
        return f"Crack Depth Prediction Model: Trained = {self.is_trained}, Params = {self.params}"

    def preprocess_training_data(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies custom transformer pipelines required by the model to the training data.

        Use this method only when training the model for the first time.

        When generating predictions using a pre-trained (e.g. joblib) model, use preprocess_unseen_data

        Args:
            df (pd.DataFrame): Dataframe containing the experimental data in the form:

            | x | temp | nf_hz | amp_mm | tc_act |
            ---------------------------------------

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of [feature_array, target_array]

        Usage:
            model = Model()

            X_train, y_train = model.preprocess_training_data(df)
        """

        # Shuffle the data to prevent model weirdness
        df_shuf = df.sample(frac=1)

        self.preprocessor = Pipeline([("scaler", StandardScaler())])

        X_train = self.preprocessor.fit_transform(df_shuf.drop("tc_act", axis=1))
        y_train = df_shuf["tc_act"]

        return X_train, y_train

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the instantiated model on X and Y arrays

        Args:
            X (np.ndarray): Feature array
            y (np.ndarray): Target array

        Returns:
            [None]: Returns none
        """

        self.model = Ridge(**self.params)
        self.model.fit(X, y)

        # Used for the following stages and for the custom exception
        self.is_trained = True

        return None

    def preprocess_unseen_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies custom transformer pipelines required by the model to the training data.

        Use this method only when testing the pre-trained (e.g. joblib) model.

        When training the model for the first time, use preprocess_unseen_data

        Args:
            df (pd.DataFrame): Dataframe containing experimental data in the form

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of [feature_array, target_array]

        Usage:

            X_test, y_test = model.preprocess_unseen_data(df)
        """

        # Again, shuffle the data to prevent weirdness
        df_shuf = df.sample(frac=1)

        X_test = self.preprocessor.transform(df_shuf.drop("tc_act", axis=1))
        y_test = df_shuf["tc_act"]

        return X_test, y_test

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates an array of predictions from the feature array X

        Args:
            X (np.ndarray): Array of features

        Raises:
            NotTrainedError: If attempting to predict using an untrained model, will raise NotTrainedError.

        Returns:
            np.ndarray: Array of predictions
        """

        if self.is_trained:
            return self.model.predict(X)
        else:
            raise NotTrainedError()

    def test(
        self, y_test: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[np.float64, np.float64]:
        """
        Tests the models performance using root mean squared error and r2 score

        Args:
            y_test (np.ndarray): True target labels
            y_pred (np.ndarray): Model generated predictions

        Returns:
            Tuple[np.float64, np.float64]: Tuple of [rmse, r2_score]
        """

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"RMSE: {np.round(rmse, 3)}")
        print(f"R^2: {np.round(r2, 3)}")

        return rmse, r2

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, cv: int = 5
    ) -> List[np.float64]:
        """
        Performs k-fold cross validation to evaluate the rmse of the trained model instance.

        Args:
            X (np.ndarray): Feature array, e.g. X_train
            y (np.ndarray): Target array, e.g. y_train
            cv (int, optional): Number of folds for the k-fold validation. Defaults to 5.

        Returns:
            List[np.float64]: List of RMSE scores of length = cv
        """

        val_rmses = np.sqrt(
            cross_val_score(self.model, X, y, scoring="neg_mean_squared_error") * -1
        )

        return val_rmses

    def save(self, file_name: str) -> BinaryIO:
        """
        Saves a trained model to a pkl file in Models/

        Args:
            file_name (str): File name for saved model. Must be valid pkl e.g. my_model.pkl

        Raises:
            NotTrainedError: If attempting to save an untrained model, will raise NotTrainedError.

        Returns:
            BinaryIO: Saves model to Models/
        """

        path = PROJECT_ROOT / "Models"

        if self.is_trained:
            if not Path.exists(path / file_name):
                joblib.dump(self, path / file_name)
            else:
                print(f"Model {file_name} already exists")
        else:
            raise NotTrainedError()

        return None
