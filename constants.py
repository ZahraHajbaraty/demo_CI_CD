import shutil
from pathlib import Path

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