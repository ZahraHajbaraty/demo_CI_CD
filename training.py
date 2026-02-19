import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay)
import json
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from constants import target_col

def scale_data(X_train, X_test) -> pd.DataFrame:
    """
    Impute missing values and scale numeric features in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numeric_cols (list): A list of numeric column names to be imputed and scaled.

    Returns:
    pd.DataFrame: A DataFrame with imputed and scaled numeric features.
    """
    # Impute missing values with the mean
    # imputer = SimpleImputer(strategy='mean')
    # df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    num_cols = X_train.select_dtypes(include=["float64", "int64"]).columns

    # exclude binary columns
    binary_cols = [c for c in num_cols if X_train[c].nunique() == 2]
    cont_cols = [c for c in num_cols if c not in binary_cols]

    scaler = StandardScaler()

    X_train[cont_cols] = scaler.fit_transform(X_train[cont_cols])
    X_test[cont_cols] = scaler.transform(X_test[cont_cols])
    return X_train, X_test



def split_data(df, target_col, test_size=0.20, random_state=42):
    """
    Split dataset by time:
    - first (1 - test_size) for training
    - last test_size for testing

    Parameters:
    df (pd.DataFrame): Input DataFrame
    target_col (str): Target column name
    test_size (float): Fraction of data used for test (e.g. 0.2)

    Returns:
    X_train, X_test, y_train, y_test
    """
    df = df.sort_values(by="Date").reset_index(drop=True)

    # 2. Compute split index
    split_idx = int(len(df) * (1 - test_size))

    # 3. Split
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # 4. Separate X and y
    X_train = train_df.drop(columns=[target_col, "Date"])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col, "Date"])
    y_test = test_df[target_col]

    X_train, X_test = scale_data(X_train, X_test)
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, n_estimators=50, random_state=42, max_depth=10):
    """
    Train a Random Forest Classifier on the training data.

    Parameters:
    X_train (pd.DataFrame): The training features.
    y_train (pd.Series): The training target variable.

    Returns:
    model: The trained Random Forest model.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth)
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test data.

    Parameters:
    model: The trained model to be evaluated.
    X_test (pd.DataFrame): The test features.
    y_test (pd.Series): The true labels for the test set.

    Returns:
    None: Prints the classification report and confusion matrix.
    """
    y_pred = model.predict(X_test)

    print("Classification Report:", "\n",
           classification_report(y_test, y_pred))
    
    print("Confusion Matrix:", "\n",
           confusion_matrix(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    return json.loads(
        json.dumps(metrics), parse_float=lambda x: round(float(x), 2)
    )
