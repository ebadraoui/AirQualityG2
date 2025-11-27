import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error
)

# ------------------------------------
# Paths
# ------------------------------------
def get_project_paths():
    """
    Returns ROOT, FIGS_DIR, TABLES_DIR for consistent saving.
    """
    ROOT = Path(__file__).resolve().parents[1]
    FIGS_DIR = ROOT / "reports" / "figs"
    TABLES_DIR = ROOT / "reports" / "tables"
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    return ROOT, FIGS_DIR, TABLES_DIR


# ------------------------------------
# Plotting helpers
# ------------------------------------
def save_plot(fig, name):
    """
    Saves a generated plot into reports/figs.
    """
    _, FIGS_DIR, _ = get_project_paths()
    path = FIGS_DIR / name
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def plot_confusion_matrix(cm, title, filename):
    fig = plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    save_plot(fig, filename)


def plot_residuals(pred, residuals, title, filename):
    fig = plt.figure(figsize=(6,4))
    plt.scatter(pred, residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(title)
    save_plot(fig, filename)


def plot_learning_curve(history, metric, val_metric, title, filename, ylabel):
    fig = plt.figure(figsize=(6,4))
    plt.plot(history.history[metric], label=f"Train {ylabel}")
    plt.plot(history.history[val_metric], label=f"Val {ylabel}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    save_plot(fig, filename)


# ------------------------------------
# Metric Helpers
# ------------------------------------
def eval_classification(model, X_test, y_test):
    """
    Returns accuracy, F1, cm
    """
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return acc, f1, cm, y_pred, y_prob


def eval_regression(model, X_test, y_test):
    """
    Returns MAE, RMSE, residuals, preds
    """
    pred = model.predict(X_test).ravel()
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    residuals = y_test - pred
    return mae, rmse, pred, residuals


# ------------------------------------
# Table helpers
# ------------------------------------
def save_table(df, filename):
    _, _, TABLES_DIR = get_project_paths()
    df.to_csv(TABLES_DIR / filename, index=False)
