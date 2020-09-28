"""
Single script that when run, will:

1) Extract the raw data from the two excel files in Data/Raw: Al_Nat_Freq.xlsx and Al_Amplitude.xlsx
2) Convert that data to tidy format required for modelling
3) Replicate the polynomial model from the paper and append the predictions to the data
4) Do the required formatting for final modelling

Required Inputs
---------
file: Data/Raw/Al_Nat_Freq.xlsx   <- Excel file containing raw frequency data (provided at start of project)
file: Data/Raw/Al_Amplitude.xlsx  <- Excel file containing raw amplitude data (provided at start of project)

Outputs
---------
file: Data/Processed/processed_combined.csv  <- File with all observations from raw data in "tidy" format
file: Data/Processed/full_with_poly_preds.csv  <- File with all observations plus predictions from
                                                    the paper's polynomial model
file: Data/Final/al_data_final.csv   <- File with final format for input to modelling

Author: Tom Fleet
Created: 10/06/2020
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from src.config import FINAL_DATA, PROCESSED_DATA, RAW_DATA
from src.data.df_utils import process_data_final

amp_path = RAW_DATA / "Al_Amplitude.xlsx"
freq_path = RAW_DATA / "Al_Nat_Freq.xlsx"

# Because the data is on sheets with different names
amp_sheet = "Summary drop"
freq_sheet = "Drop summary"

# Sets the rows to skip at the start for each set of data in order to grab the correct chunk
skip_rows5 = [0, 1]
skip_rows15 = list(range(18))
skip_rows25 = list(range(35))
# Number of rows per chunk is the same for all the excel files
n_rows = 12

# Sets the columns to either ignore or load depending on what features we want
# Since the sheets are formatted the same, we can either select:
# Values (frequency or amplitude) or 'drops' (frequency or amplitude drop)
val_cols = [0, 3, 4, 6, 8, 10, 12]
drop_cols = [0, 4, 5, 7, 9, 11, 13]

# Chunks of amplitude data
# 5mm crack position
amp5_vals = pd.read_excel(
    amp_path, sheet_name=amp_sheet, skiprows=skip_rows5, nrows=n_rows, usecols=val_cols
)

assert amp5_vals.shape == (12, 7)

# 15mm crack position
amp15_vals = pd.read_excel(
    amp_path, sheet_name=amp_sheet, skiprows=skip_rows15, nrows=n_rows, usecols=val_cols
)

assert amp15_vals.shape == (12, 7)

# 25mm crack position
amp25_vals = pd.read_excel(
    amp_path, sheet_name=amp_sheet, skiprows=skip_rows25, nrows=n_rows, usecols=val_cols
)

assert amp25_vals.shape == (12, 7)

# Chunks of frequency data
# 5mm crack position
freq5_vals = pd.read_excel(
    freq_path,
    sheet_name=freq_sheet,
    skiprows=skip_rows5,
    nrows=n_rows,
    usecols=val_cols,
)

assert freq5_vals.shape == (12, 7)

# 15mm crack position
freq15_vals = pd.read_excel(
    freq_path,
    sheet_name=freq_sheet,
    skiprows=skip_rows15,
    nrows=n_rows,
    usecols=val_cols,
)

assert freq15_vals.shape == (12, 7)

# 25mm crack position
freq25_vals = pd.read_excel(
    freq_path,
    sheet_name=freq_sheet,
    skiprows=skip_rows25,
    nrows=n_rows,
    usecols=val_cols,
)

assert freq25_vals.shape == (12, 7)

# Chunks of amplitude drop data
# 5mm crack position
amp5_drops = pd.read_excel(
    amp_path, sheet_name=amp_sheet, skiprows=skip_rows5, nrows=n_rows, usecols=drop_cols
)

assert amp5_drops.shape == (12, 7)

# 15mm crack position
amp15_drops = pd.read_excel(
    amp_path,
    sheet_name=amp_sheet,
    skiprows=skip_rows15,
    nrows=n_rows,
    usecols=drop_cols,
)

assert amp15_drops.shape == (12, 7)

# 25mm crack position
amp25_drops = pd.read_excel(
    amp_path,
    sheet_name=amp_sheet,
    skiprows=skip_rows25,
    nrows=n_rows,
    usecols=drop_cols,
)

assert amp25_drops.shape == (12, 7)


# Chunks of frequency drop data
# 5mm crack position
freq5_drops = pd.read_excel(
    freq_path,
    sheet_name=freq_sheet,
    skiprows=skip_rows5,
    nrows=n_rows,
    usecols=drop_cols,
)

assert freq5_drops.shape == (12, 7)

# 15mm crack position
freq15_drops = pd.read_excel(
    freq_path,
    sheet_name=freq_sheet,
    skiprows=skip_rows15,
    nrows=n_rows,
    usecols=drop_cols,
)

assert freq15_drops.shape == (12, 7)

# 25mm crack position
freq25_drops = pd.read_excel(
    freq_path,
    sheet_name=freq_sheet,
    skiprows=skip_rows25,
    nrows=n_rows,
    usecols=drop_cols,
)

assert freq25_drops.shape == (12, 7)

# Group all the amps/freqs vals/drops together in combined dataframes
val_amp = [amp5_vals, amp15_vals, amp25_vals]
drop_amp = [amp5_drops, amp15_drops, amp25_drops]

val_freq = [freq5_vals, freq15_vals, freq25_vals]
drop_freq = [freq5_drops, freq15_drops, freq25_drops]

amp = pd.concat(val_amp, ignore_index=True)
amp_drop = pd.concat(drop_amp, ignore_index=True)

freq = pd.concat(val_freq, ignore_index=True)
freq_drop = pd.concat(drop_freq, ignore_index=True)

# Set what the columns refer to in each case
val_col_list = ["x", 22, "CD/t", 50, 100, 150, 200]
drop_col_list = ["x", "CD/t", 22, 50, 100, 150, 200]

freq.columns = val_col_list
amp.columns = val_col_list

freq_drop.columns = drop_col_list
amp_drop.columns = drop_col_list

# Now for a bit of reording to make the value ones more readable, the drop ones are already in this order
freq = freq[["x", "CD/t", 22, 50, 100, 150, 200]]
amp = amp[["x", "CD/t", 22, 50, 100, 150, 200]]


nf = pd.melt(freq, id_vars=["x", "CD/t"], var_name="temp", value_name="nf_hz")
nfdrop = pd.melt(
    freq_drop, id_vars=["x", "CD/t"], var_name="temp", value_name="nf_drop"
)

amp = pd.melt(amp, id_vars=["x", "CD/t"], var_name="temp", value_name="amp_mm")
ampdrop = pd.melt(
    amp_drop, id_vars=["x", "CD/t"], var_name="temp", value_name="amp_drop"
)

# Finally, combine them all into one master dataframe
master = nf
master["nf_drop"] = nfdrop["nf_drop"]
master["amp_mm"] = amp["amp_mm"]
master["amp_drop"] = ampdrop["amp_drop"]

# If this passes, the data was manipulated and processed correctly
assert master.shape == (180, 7)

# Save the processed version as a csv file
file_name = PROCESSED_DATA / "processed_combined.csv"

if not Path.exists(file_name):
    master.to_csv(file_name, index=False)
    print(f"Saved file: {file_name}")

# Now onto generating the polynomial features to recreate the paper's model
data = pd.read_csv(file_name)

# Only need frequency drop and temperature as these are the base features
freq = data.drop(["x", "CD/t", "nf_hz", "amp_mm", "amp_drop"], axis=1)
amp = data.drop(["x", "CD/t", "nf_hz", "nf_drop", "amp_mm"], axis=1)

# Reorder columns to make the matrix math work
freq = freq[["nf_drop", "temp"]]
amp = amp[["amp_drop", "temp"]]

# Generate the polynomial features on both the freq and amp dataframes
poly_features = PolynomialFeatures(degree=3, include_bias=False)

freq_poly = poly_features.fit_transform(freq)
amp_poly = poly_features.fit_transform(amp)

freq_poly = pd.DataFrame(freq_poly)
amp_poly = pd.DataFrame(amp_poly)

# Drop the 8 column as it refers to T^3 which is not used in the matrix dot product stage
freq_poly.drop(8, axis=1, inplace=True)
amp_poly.drop(8, axis=1, inplace=True)

# Add the x column back in so we can index off it
freq_poly["x"] = data["x"]
amp_poly["x"] = data["x"]

# Break it up into chunks corresponding to 5mm, 15mm and 25mm to be later matched in matrix math to the relevant coeff
# Then drop the 'x' as it's not used in the actual dot product
freq_5 = freq_poly.loc[freq_poly["x"] == 5].drop("x", axis=1).values
freq_15 = freq_poly.loc[freq_poly["x"] == 15].drop("x", axis=1).values
freq_25 = freq_poly.loc[freq_poly["x"] == 25].drop("x", axis=1).values

amp_5 = amp_poly.loc[amp_poly["x"] == 5].drop("x", axis=1).values
amp_15 = amp_poly.loc[amp_poly["x"] == 15].drop("x", axis=1).values
amp_25 = amp_poly.loc[amp_poly["x"] == 25].drop("x", axis=1).values

# Assertions to confirm the right bits of data were allocated
assert len(freq_5) == 60
assert len(freq_15) == 60
assert len(freq_25) == 60

assert len(amp_5) == 60
assert len(amp_15) == 60
assert len(amp_25) == 60


# Same for coeffs
# Save the constant A's first
# Frequency A's
A_freq_5, A_freq_15, A_freq_25 = (0.9367, 0.8726, 0.8574)
# Amplitude A's
A_amp_5, A_amp_15, A_amp_25 = (0.5348, -0.1923, -0.5979)


# Frequency coefficients, taken directly from the paper
coefs_freq_5 = np.array(
    [
        0.359,
        -0.01433,
        -0.01637,
        -0.001936,
        0.0001338,
        -0.0001470,
        0.0002067,
        -0.0000129,
    ]
)
coefs_freq_15 = np.array(
    [
        0.3701000,
        -0.0110600,
        -0.0182800,
        -0.0021360,
        0.0001200,
        -0.0000519,
        0.0002018,
        -0.0000117,
    ]
)
coefs_freq_25 = np.array(
    [
        0.3973000,
        -0.0089170,
        -0.0222300,
        -0.0023310,
        0.0001143,
        0.0001004,
        0.0002062,
        -0.0000115,
    ]
)


# Amplitude coefficients, taken directly from the paper
coefs_amp_5 = np.array(
    [
        0.6342000,
        0.0036250,
        0.0874400,
        -0.0119300,
        0.0000120,
        -0.0101300,
        -0.0015330,
        0.0000241,
    ]
)
coefs_amp_15 = np.array(
    [
        0.3510000,
        0.0205600,
        0.0629000,
        -0.0078400,
        -0.0000419,
        -0.0083890,
        -0.0010900,
        -0.0000223,
    ]
)
coefs_amp_25 = np.array(
    [
        0.1834000,
        0.0276500,
        0.0908900,
        -0.0004958,
        -0.0000509,
        -0.0029020,
        -0.0005170,
        0.0000102,
    ]
)


# Now do the dot products
tc_pred_freq_5 = A_freq_5 + np.dot(freq_5, coefs_freq_5)
tc_pred_freq_15 = A_freq_15 + np.dot(freq_15, coefs_freq_15)
tc_pred_freq_25 = A_freq_25 + np.dot(freq_25, coefs_freq_25)
tc_pred_amp_5 = A_amp_5 + np.dot(amp_5, coefs_amp_5)
tc_pred_amp_15 = A_amp_15 + np.dot(amp_15, coefs_amp_15)
tc_pred_amp_25 = A_amp_25 + np.dot(amp_25, coefs_amp_25)


# Add all the predictions into two 180 long 1D arrays, ready for merging back into the dataframe
tc_preds_freq = np.concatenate((tc_pred_freq_5, tc_pred_freq_15, tc_pred_freq_25))

assert tc_preds_freq.shape == (180,)

tc_preds_amp = np.concatenate((tc_pred_amp_5, tc_pred_amp_15, tc_pred_amp_25))

assert tc_preds_amp.shape == (180,)


data["tc_pred_freq"] = tc_preds_freq
data["tc_pred_amp"] = tc_preds_amp

# Add a column to hold the actual crack depth (from the CD/t column)
# We know the specimens were all 3mm thick
data["tc_act"] = data["CD/t"].apply(lambda x: x * 3)

# Save the output data as a csv
file_name = PROCESSED_DATA / "full_with_poly_preds.csv"

if not Path.exists(file_name):
    data.to_csv(file_name, index=False)
    print(f"Saved file {file_name}")

# Generate final data to input into modelling
final_df = process_data_final(
    cols=["CD/t", "tc_pred_freq", "tc_pred_amp", "amp_drop", "nf_drop"]
)

file_name = FINAL_DATA / "al_data_final.csv"

if not Path.exists(file_name):
    final_df.to_csv(file_name, index=False)
    print(f"Saved file {file_name}")
