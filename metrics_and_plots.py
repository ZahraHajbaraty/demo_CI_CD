import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from constants import PLOT_DIR


def plot_confusion_matrix(model, X_test, y_test):
    _ = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap=plt.cm.Blues)
    Path(PLOT_DIR).mkdir(exist_ok=True)
    plt.savefig(f"{PLOT_DIR}/confusion_matrix.png")


def save_metrics(metrics):
    Path(PLOT_DIR).mkdir(exist_ok=True)
    with open(f"{PLOT_DIR}/metrics.json", "w") as fp:
        json.dump(metrics, fp)
