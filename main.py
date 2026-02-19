import os
from preprocessing import load_data
from training import split_data, train_model, evaluate_model
import pandas as pd
from metrics_and_plots import plot_confusion_matrix, save_metrics, save_predictions, save_roc_curve
import json
from constants import PROCESSED_DIR, load_hyperparameters, target_col




def main():
    df = load_data(PROCESSED_DIR)
    X_train, X_test, y_train, y_test = split_data(df, target_col=target_col)
    hyperparameters = load_hyperparameters("rfc_best_params.json")
    model = train_model(X_train, y_train, hyperparameters)
    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)

    # Save metrics into json file
    save_metrics(metrics)
    save_predictions(y_test, y_pred)
    save_roc_curve(y_test, y_pred_proba)
    plot_confusion_matrix(model, X_test, y_test)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    print("\ny_train distribution:")
    print(y_train.value_counts(dropna=False))

    print("\ny_test distribution:")
    print(y_test.value_counts(dropna=False))

    print("\nIndex overlap:",
        len(set(X_train.index).intersection(set(X_test.index))))

    print("\nFirst rows of X_train:")
    print(X_train.head(10))

    print("\nFirst rows of y_train:")
    print(y_train.head(10))




if __name__ == "__main__":
    main()