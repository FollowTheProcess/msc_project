"""
Script to generate a randomly sampled subset of experiments from the final modelling data
to be run as a further validation sample by other experimenters.

Author: Tom Fleet
Created: 21/06/2020
"""

from pathlib import Path
from typing import BinaryIO

import pandas as pd

from src.config import FINAL_DATA, PROCESSED_DATA

data = FINAL_DATA / "abs_data_final.csv"


def make_validation_worksheet(df: pd.DataFrame, n_rows: int = 30) -> BinaryIO:
    """
    Generates random sample of x and temp for use as a validation worksheet for future experiments

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the final modelling data in the correct format
        i.e. columns = [x, temp, nf_hz, amp_mm, tc_act], by default data
    n_rows : int, optional
        Number of validation instances to create, by default 30

    Returns
    -------
    BinaryIO
        Saves 'random_validation_worksheet.csv' in Data/Interim containing validation sample.
    """
    out_file = PROCESSED_DATA / "abs_random_validation_worksheet.csv"

    final_data = pd.read_csv(data)
    sample = final_data.sample(n_rows)

    # Drop the observed measurements but maintain the blank columns
    sample["nf_hz"] = ""
    sample["amp_mm"] = ""
    sample["tc_act"] = ""

    if not Path.exists(out_file):
        sample.to_csv(out_file, index=False)

    else:
        print(f"File: {out_file} already exists")

    return None


if __name__ == "__main__":
    make_validation_worksheet(df=data)
