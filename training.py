from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay)
import json

def split_data(df, target_col, test_size=0.2, random_state=42):
    """
    Split the DataFrame into training and testing sets.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_col (str): The name of the target column.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): Controls the randomness of the split.

    Returns:
    X_train, X_test, y_train, y_test: The split data.
    """
    df = df.sort_values(by="Date")
    X = df.drop(columns=[target_col, "Date"])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, n_estimators=50, random_state=42, max_depth=5):
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
