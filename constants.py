import shutil
from pathlib import Path
from sklearn.model_selection import GridSearchCV
import json
import pandas as pd

DATA_PATH = "raw_dataset/weatherHistory.csv"  # Replace with your actual dataset path
PROCESSED_DIR= "processed_dataset"
PLOT_DIR = "plots"
target_col='Precip Type'  # Replace with your actual target column name
drop_columns=["Summary", "Daily Summary"]


def delete_and_recreate_dir(path):
    try:
        shutil.rmtree(path)
    except:
        pass
    finally:
        Path(path).mkdir(parents=True, exist_ok=True)


def load_hyperparameters(hyperparameter_file):
    with open(hyperparameter_file, "r") as json_file:
        hyperparameters = json.load(json_file)
    return hyperparameters


def get_hp_tuning_results(grid_search: GridSearchCV) -> str:
    """Get the results of hyperparameter tuning in a Markdown table"""
    cv_results = pd.DataFrame(grid_search.cv_results_)

    # Extract and split the 'params' column into subcolumns
    params_df = pd.json_normalize(cv_results["params"])

    # Concatenate the params_df with the original DataFrame
    cv_results = pd.concat([cv_results, params_df], axis=1)

    # Get the columns to display in the Markdown table
    cv_results = cv_results[
        ["rank_test_score", "mean_test_score", "std_test_score"]
        + list(params_df.columns)
    ]

    cv_results.sort_values(by="mean_test_score", ascending=False, inplace=True)
    return cv_results.to_markdown(index=False)
