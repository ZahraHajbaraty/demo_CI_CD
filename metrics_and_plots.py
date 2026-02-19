import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from constants import PLOT_DIR
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def plot_confusion_matrix(model, X_test, y_test):
    _ = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap=plt.cm.Blues)
    Path(PLOT_DIR).mkdir(exist_ok=True)
    plt.savefig(f"{PLOT_DIR}/confusion_matrix.png")


def save_metrics(metrics):
    Path(PLOT_DIR).mkdir(exist_ok=True)
    with open(f"{PLOT_DIR}/metrics.json", "w") as fp:
        json.dump(metrics, fp)



def save_predictions(y_test, y_pred):
    # Store predictions data for confusion matrix
    cdf = pd.DataFrame(
        np.column_stack([y_test, y_pred]), columns=["true_label", "predicted_label"]
    ).astype(int)
    cdf.to_csv("predictions.csv", index=None)


def save_roc_curve(y_test, y_pred_proba):
    # Calcualte ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    # Store roc curve data
    cdf = pd.DataFrame(np.column_stack([fpr, tpr]), columns=["fpr", "tpr"]).astype(
        float
    )
    cdf.to_csv("roc_curve.csv", index=None)

