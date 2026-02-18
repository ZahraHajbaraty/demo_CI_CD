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
    print("====================Test Set Metrics==================")
    print(json.dumps(metrics, indent=2))
    print("======================================================")

    # Save metrics into json file
    save_metrics(metrics)
    plot_confusion_matrix(model, X_test, y_test)



if __name__ == "__main__":
    main()