import os
from training import split_data, train_model, evaluate_model
import pandas as pd
from metrics_and_plots import plot_confusion_matrix, save_metrics
import json
from constants import PROCESSED_DIR, target_col




def main():
    csv_files = [
        f for f in os.listdir(PROCESSED_DIR)
        if f.endswith(".csv")
    ]
    df = pd.read_csv(os.path.join(PROCESSED_DIR, csv_files[0]), parse_dates=["Date"])
    X_train, X_test, y_train, y_test = split_data(df, target_col=target_col)
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    # Save metrics into json file
    save_metrics(metrics)
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