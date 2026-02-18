import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import List
from constants import DATA_PATH, delete_and_recreate_dir, target_col, drop_columns, PROCESSED_DIR
from pathlib import Path




def read_dataset(
    filename: str, drop_columns: List[str]=None, target_column: str=None
) -> pd.DataFrame:
    """
    Reads the raw data file and returns pandas dataframe
    Target column values are expected in binary format with Yes/No values

    Parameters:
    filename (str): raw data filename
    drop_columns (List[str]): column names that will be dropped
    target_column (str): name of target column

    Returns:
    pd.Dataframe: Target encoded dataframe
    """
    df = pd.read_csv(filename, sep=",").drop(columns=drop_columns)
    df["Formatted Date"] = pd.to_datetime(df["Formatted Date"], utc=True)
    df = df.rename(columns={"Formatted Date": "Date"})
    if target_column:
        df[target_column] = df[target_column].map({"rain": 1, "snow": 0})
    return df


def categorical_column(df:pd.DataFrame) -> List[str]:
    """
    Identifies categorical columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    List[str]: A list of categorical column names.
    """
    return df.select_dtypes(include=[object]).columns.to_list()




def target_encode_categorical_features(df:pd.DataFrame, target_col:str, categorical_cols:list) -> pd.DataFrame:
    """
    Target encode categorical features in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_col (str): The name of the target column.
    categorical_cols (list): A list of categorical column names to be target encoded.

    Returns:
    pd.DataFrame: A DataFrame with target encoded features.
    """
    categorical_cols = [c for c in categorical_cols if c != target_col and c != "Date"]
    for col in categorical_cols:
        # Calculate the mean of the target variable for each category
        target_mean = df.groupby(col)[target_col].mean()
        # Map the mean values back to the original DataFrame
        df[col] = df[col].map(target_mean)
    
    return df


def impute_and_scale_data(df:pd.DataFrame, numeric_cols:list=[]) -> pd.DataFrame:
    """
    Impute missing values and scale numeric features in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numeric_cols (list): A list of numeric column names to be imputed and scaled.

    Returns:
    pd.DataFrame: A DataFrame with imputed and scaled numeric features.
    """
    if not numeric_cols:
        numeric_cols = [c for c in df.select_dtypes(include=[float, int]).columns if c != target_col]
        # print(f"No numeric columns specified. Scaling all numeric columns: {numeric_cols}")
    
    # Impute missing values with the mean
    imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # Scale the numeric features
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        
        

    return df

def nan_duplicate(df:pd.DataFrame) -> pd.DataFrame:
    """
    Duplicate rows with NaN values in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: A DataFrame with duplicated rows for NaN values.
    """
    df = df.dropna()
    df = df.drop_duplicates(subset=["Date"], keep="first")

    return df

def main():
    df = read_dataset(filename=DATA_PATH, drop_columns=drop_columns, target_column=target_col)
    categorical_cols = categorical_column(df)
    df = target_encode_categorical_features(df, target_col=target_col, categorical_cols=categorical_cols)
    df = impute_and_scale_data(df)
    df = nan_duplicate(df)
    delete_and_recreate_dir(PROCESSED_DIR)
    df.to_csv(os.path.join(PROCESSED_DIR, "weatherHistory.csv"), index=False)


if __name__ == "__main__":
    main()
