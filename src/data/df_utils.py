"""
Module to hold custom DataFrame manipulation functions to be imported in other files.

Author: Tom Fleet
Created: 21/05/2020
"""

from typing import List

import pandas as pd

from src.config import PROCESSED_DATA


# Convert processed data into data for input into model pipeline
def process_data_final(cols: List[str]) -> pd.DataFrame:
    """
    Converts process interim data into format required for input into model pipeline.
    Expects file in Data/Processed called "full_with_poly_preds.csv.
    This file is generated automatically when make_data.py is run.

    Args:
        cols (List[str]): List of DataFrame column names to be dropped, happens inplace.

    Returns:
        pd.DataFrame: Dataframe of formatted values
    """
    data_in = PROCESSED_DATA / "full_with_poly_preds.csv"
    df = pd.read_csv(data_in)
    df.drop(cols, axis=1, inplace=True)

    return df
