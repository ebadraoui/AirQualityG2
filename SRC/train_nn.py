import numpy as np
import pandas as pd
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import f1_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Try relative import (when run as module) then fallback (when run as script)
try:
    from .utils import (
        plot_learning_curve,
        plot_confusion_matrix,
        plot_residuals,
        eval_classification,
        eval_regression,
        save_table,
        get_project_paths,
        save_plot,
    )
except ImportError:  # running as plain script
    from utils import (
        plot_learning_curve,
        plot_confusion_matrix,
        plot_residuals,
        eval_classification,
        eval_regression,
        save_table,
        get_project_paths,
        save_plot,
    )

# -------------------------------------------------------------------
# Global paths / seeds
# -------------------------------------------------------------------
ROOT, FIGS_DIR, TABLES_DIR = get_project_paths()
DATA_PATH = ROOT / "data" / "PRSA_data_2010.1.1-2014.12.31.csv"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# -------------------------------------------------------------------
# 1. Load data + basic feature engineering
# -------------------------------------------------------------------
def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    # keep only valid hours
    df = df[df["hour"].between(0, 23)].copy()
    # drop rows with missing target
    df = df.dropna(subset=["pm2.5"]).copy()

    # classification target
    df["healthy"] = (df["pm2.5"] < 75).astype(int)

    # cyclic time features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # feature set
    feature_cols = [
        "DEWP",
        "TEMP",
        "PRES",
        "Iws",
        "Is",
        "Ir",
        "hour",
        "month",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "cbwd",  # categorical wind direction
    ]

    X = df[feature_cols].copy()
    y_class = df["healthy"].copy()
    y_reg = df["pm2.5"].copy()

    return X, y_class, y_reg


# -------------------------------------------------------------------
# 2. Split + preprocessing (scaling + one-hot)
# -------------------------------------------------------------------
def split_and_preprocess(X, y_class, y_reg):
    # 70 / 15 / 15 split
    X_train, X_temp, y_class_train, y_class_temp, y_reg_train, y_reg_temp = train_test_split(
        X, y_class, y_reg, test_size=0.30, random_state=RANDOM_STATE
    )
    X_valid, X_test, y_class_valid, y_class_test, y_reg_valid, y_reg_test = train_test_split(
        X_temp, y_class_temp, y_reg_temp, test_size=0.50, random_state=RANDOM_STATE
    )

    num_features = [
        "DEWP",
        "TEMP",
        "PRES",
        "Iws",
        "Is",
        "Ir",
        "hour",
        "month",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
    ]
    cat_features = ["cbwd"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )

    # Fit on TRAIN only
    X_train_proc = preprocessor.fit_transform(X_train)
    X_valid_proc = preprocessor.transform(X_valid)
    X_test_proc = preprocessor.transform(X_test)

    # convert to dense if sparse
    if hasattr(X_train_proc, "toarray"):
        X_train_proc = X_train_proc.toarray()
        X_valid_proc = X_valid_proc.toarray()
        X_test_proc = X_test_proc.toarray()

    feature_names = preprocessor.get_feature_names_out()

    return (
        X_train_proc,
        X_valid_proc,
        X_test_proc,
        y_class_train.values,
        y_class_valid.values,
        y_class_test.values,
        y_reg_train.values,
        y_reg_valid.values,
        y_reg_test.values,
        feature_names,
    )


# -------------------------------------------------------------------
# 3. Build Keras models
# -------------------------------------------------------------------
def build_classifier(input_dim: int):
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_regressor(input_dim: int):
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mae",
        metrics=["mae"],
    )
    return model


# -------------------------------------------------------------------
# 4. Training functions (with learning curves)
# -------------------------------------------------------------------
def train_classifier_nn(X_train, X_valid, y_train, y_valid):
    model = build_classifier(X_train.shape[1])
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        epochs=100,
        batch_size=256,
        verbose=1,
        callbacks=callbacks,
    )

    # Plot 1: learning curve (classification)
    plot_learning_curve(
        history,
        metric="accuracy",
        val_metric="val_accuracy",
        title="Plot 1: Learning Curve (Classification NN)",
        filename="plot1_nn_classification_learning_curve.png",
        ylabel="Accuracy",
    )
    return model


def train_regressor_nn(X_train, X_valid, y_train, y_valid):
    model = build_regressor(X_train.shape[1])
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        epochs=100,
        batch_size=256,
        verbose=1,
        callbacks=callbacks,
    )

    # Plot 2: learning curve (regression)
    plot_learning_curve(
        history,
        metric="mae",
        val_metric="val_mae",
        title="Plot 2: Learning Curve (Regression NN)",
        filename="plot2_nn_regression_learning_curve.png",
        ylabel="MAE",
    )
    return model


# -------------------------------------------------------------------
# 5. Evaluation + required plots
# -------------------------------------------------------------------
def evaluate_classifier(model, X_test, y_test):
    acc, f1, cm, y_pred, y_prob = eval_classification(model, X_test, y_test)
    print(f"[NN Classification] Acc={acc:.3f}, F1={f1:.3f}")

    # Plot 3: confusion matrix
    plot_confusion_matrix(
        cm,
        title="Plot 3: Confusion Matrix (Best Final Classification Model)",
        filename="plot3_final_confusion_matrix.png",
    )

    return acc, f1, y_pred, y_prob


def evaluate_regressor(model, X_test, y_test):
    mae, rmse, y_pred, residuals = eval_regression(model, X_test, y_test)
    print(f"[NN Regression] MAE={mae:.2f}, RMSE={rmse:.2f}")

    # Plot 4: residuals vs predicted
    plot_residuals(
        y_pred,
        residuals,
        title="Plot 4: Residuals vs Predicted (Best Final Regression Model)",
        filename="plot4_final_residuals_vs_predicted.png",
    )

    return mae, rmse, y_pred, residuals


# -------------------------------------------------------------------
# 6. Permutation feature importance (Plot 5)
# -------------------------------------------------------------------
def feature_importance_permutation(model, X_test, y_test, feature_names, n_repeats=5):
    """
    Simple permutation importance using F1 score for the classifier.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    # baseline score
    base_probs = model.predict(X_test).ravel()
    base_pred = (base_probs >= 0.5).astype(int)
    baseline_f1 = f1_score(y_test, base_pred)

    importances = []
    X_copy = X_test.copy()

    for j in range(X_test.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_perm = X_copy.copy()
            rng.shuffle(X_perm[:, j])
            p = model.predict(X_perm).ravel()
            y_perm = (p >= 0.5).astype(int)
            scores.append(f1_score(y_test, y_perm))
        importances.append(baseline_f1 - np.mean(scores))

    importances = np.array(importances)
    idx = np.argsort(importances)[::-1][:15]  # top 15
    imp_vals = importances[idx]
    imp_names = feature_names[idx]

    fig = plt.figure(figsize=(8, 5))
    sns.barplot(x=imp_vals, y=imp_names, orient="h")
    plt.xlabel("Decrease in F1 after permutation")
    plt.ylabel("Feature")
    plt.title("Plot 5: Feature Importance via Permutation (Classification NN)")

    save_plot(fig, "plot5_feature_importance_permutation.png")


# -------------------------------------------------------------------
# 7. Build comparison tables (classical vs NN)
# -------------------------------------------------------------------
def build_comparison_tables(nn_clf_metrics, nn_reg_metrics):
    # nn_clf_metrics: (acc, f1)
    # nn_reg_metrics: (mae, rmse)

    baseline_clf = pd.read_csv(TABLES_DIR / "table1_classification_metrics.csv")
    baseline_reg = pd.read_csv(TABLES_DIR / "table2_regression_metrics.csv")

    nn_row_clf = pd.DataFrame(
        [
            {
                "Model": "MLP Classifier",
                "Accuracy": nn_clf_metrics[0],
                "F1": nn_clf_metrics[1],
            }
        ]
    )
    nn_row_reg = pd.DataFrame(
        [
            {
                "Model": "MLP Regressor",
                "MAE": nn_reg_metrics[0],
                "RMSE": nn_reg_metrics[1],
            }
        ]
    )

    clf_final = pd.concat([baseline_clf, nn_row_clf], ignore_index=True)
    reg_final = pd.concat([baseline_reg, nn_row_reg], ignore_index=True)

    save_table(clf_final, "final_table1_classical_vs_nn.csv")
    save_table(reg_final, "final_table2_classical_vs_nn.csv")

    print("Saved comparison tables:")
    print("  - final_table1_classical_vs_nn.csv")
    print("  - final_table2_classical_vs_nn.csv")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    print("Loading and preprocessing data...")
    X, y_class, y_reg = load_and_prepare()
    (
        X_tr,
        X_va,
        X_te,
        yc_tr,
        yc_va,
        yc_te,
        yr_tr,
        yr_va,
        yr_te,
        feature_names,
    ) = split_and_preprocess(X, y_class, y_reg)

    print("Training classification NN...")
    clf_nn = train_classifier_nn(X_tr, X_va, yc_tr, yc_va)

    print("Training regression NN...")
    reg_nn = train_regressor_nn(X_tr, X_va, yr_tr, yr_va)

    print("Evaluating classification NN on test set...")
    nn_acc, nn_f1, _, _ = evaluate_classifier(clf_nn, X_te, yc_te)

    print("Evaluating regression NN on test set...")
    nn_mae, nn_rmse, _, _ = evaluate_regressor(reg_nn, X_te, yr_te)

    print("Computing permutation feature importance...")
    feature_importance_permutation(clf_nn, X_te, yc_te, feature_names)

    build_comparison_tables(
        nn_clf_metrics=(nn_acc, nn_f1), nn_reg_metrics=(nn_mae, nn_rmse)
    )

    print("Done. Plots saved in reports/figs, tables in reports/tables.")


if __name__ == "__main__":
    main()
