"""
Top level config file for the project.

Purpose
---------
Here you can put any global constants that the rest of the project can refer to.

Examples include: project directories and filepaths, data dtypes, model parameters etc.

Author
------
Tom Fleet

License
-------
BSD-3-Clause

"""


from pathlib import Path

# Key project directories, using pathlib for os-agnostic relative paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "Data" / "Raw"
PROCESSED_DATA = PROJECT_ROOT / "Data" / "Processed"
FINAL_DATA = PROJECT_ROOT / "Data" / "Final"
FIGURES = PROJECT_ROOT / "Reports" / "Figures"
MODELS = PROJECT_ROOT / "Models"

# Final model params
MODEL_PARAMS = {
    "alpha": 0.01,
    "fit_intercept": True,
    "normalize": False,
    "copy_X": True,
    "max_iter": None,
    "tol": 0.001,
    "solver": "auto",
    "random_state": None,
}
