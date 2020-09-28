"""
Script to train the final chosen model implementation on the entire
provided dataset.

Author: Tom Fleet
Created: 24/06/2020
"""

import pandas as pd

from src.config import FINAL_DATA
from src.models.model import CrackDepthPredictor


def main():

    # Initialise the model instance
    final_model = CrackDepthPredictor()

    # Set up the data
    data = pd.read_csv(FINAL_DATA / "al_data_final.csv")

    # Process the data and train the model
    X, y = final_model.preprocess_training_data(data)

    # Train the model
    final_model.train(X, y)

    # Cross validate as a sanity check
    cv_scores = final_model.cross_validate(X, y)

    # Save the model as a serialised pkl file
    final_model.save("final_model.pkl")

    print(f"Cross Validation Scores: {cv_scores}")


if __name__ == "__main__":
    main()
