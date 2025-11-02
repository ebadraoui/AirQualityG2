import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- Load cleaned dataset ----------
CSV = "C:/Users/elyes/Downloads/beijing+pm2+5+data (1)/PRSA_data_2010.1.1-2014.12.31.csv"
df = pd.read_csv(CSV)
df = df[df['hour'].between(0,23)].copy()

# label + features
df['healthy'] = np.where(df['pm2.5'] < 75, 1, 0)
df = df.dropna(subset=['pm2.5']).copy()

df['hour_sin']  = np.sin(2*np.pi*df['hour']/24.0)
df['hour_cos']  = np.cos(2*np.pi*df['hour']/24.0)
df['month_sin'] = np.sin(2*np.pi*df['month']/12.0)
df['month_cos'] = np.cos(2*np.pi*df['month']/12.0)

# --------- Train/Val/Test Split ---------
X = df[['DEWP','TEMP','PRES','Iws','Is','Ir','hour_sin','hour_cos','month_sin','month_cos']]
y_class = df['healthy']
y_reg = df['pm2.5']

X_train, X_temp, y_class_train, y_class_temp, y_reg_train, y_reg_temp = train_test_split(
    X, y_class, y_reg, test_size=0.3, random_state=42
)
X_valid, X_test, y_class_valid, y_class_test, y_reg_valid, y_reg_test = train_test_split(
    X_temp, y_class_temp, y_reg_temp, test_size=0.5, random_state=42
)

# --------- Classification Models ---------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled  = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=1000)
tree_clf = DecisionTreeClassifier(max_depth=10, random_state=42)

models_clf = {'Logistic Regression': log_reg, 'Decision Tree': tree_clf}
clf_metrics = []

for name, model in models_clf.items():
    model.fit(X_train_scaled, y_class_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_class_test, y_pred)
    f1 = f1_score(y_class_test, y_pred)
    clf_metrics.append([name, acc, f1])
    print(f"{name}: Accuracy={acc:.3f}, F1={f1:.3f}")

# Confusion matrix for best classifier
best_clf_name = max(clf_metrics, key=lambda x: x[1])[0]
best_clf = models_clf[best_clf_name]
y_pred_best = best_clf.predict(X_test_scaled)
cm = confusion_matrix(y_class_test, y_pred_best)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Plot 3: Confusion Matrix ({best_clf_name})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("plot3_confusion_matrix.png")
plt.show()

# --------- Regression Models ---------
reg_models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(max_depth=10, random_state=42)
}
reg_metrics = []

for name, model in reg_models.items():
    model.fit(X_train_scaled, y_reg_train)
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_reg_test, y_pred)
    rmse = mean_squared_error(y_reg_test, y_pred, squared=False)
    reg_metrics.append([name, mae, rmse])
    print(f"{name}: MAE={mae:.2f}, RMSE={rmse:.2f}")

# Residuals plot for best regressor
best_reg_name = min(reg_metrics, key=lambda x: x[1])[0]
best_reg = reg_models[best_reg_name]
y_pred_reg = best_reg.predict(X_test_scaled)
residuals = y_reg_test - y_pred_reg
plt.figure(figsize=(6,4))
plt.scatter(y_pred_reg, residuals, alpha=0.5)
plt.axhline(0, color='r', linestyle='--')
plt.title(f"Plot 4: Residuals vs Predicted ({best_reg_name})")
plt.xlabel("Predicted PM2.5")
plt.ylabel("Residuals")
plt.tight_layout()
plt.savefig("plot4_residuals_vs_predicted.png")
plt.show()

# --------- Save Metrics for Tables ---------
pd.DataFrame(clf_metrics, columns=['Model','Accuracy','F1']).to_csv("table1_classification_metrics.csv", index=False)
pd.DataFrame(reg_metrics, columns=['Model','MAE','RMSE']).to_csv("table2_regression_metrics.csv", index=False)
