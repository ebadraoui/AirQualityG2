import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ... (same data prep as above) ...

candidates = {
    "LogisticRegression": Pipeline([("scaler", StandardScaler(with_mean=False)),
                                    ("clf", LogisticRegression(max_iter=1000))]),
    "DecisionTreeClassifier": Pipeline([("scaler", StandardScaler(with_mean=False)),
                                        ("clf", DecisionTreeClassifier(max_depth=10, random_state=42))]),
}

scores = {}
for name, pipe in candidates.items():
    pipe.fit(X_tr, y_tr)
    pred_va = pipe.predict(X_va)
    scores[name] = dict(acc=accuracy_score(y_va, pred_va), f1=f1_score(y_va, pred_va))
    print(f"{name}  valid: acc={scores[name]['acc']:.3f}, f1={scores[name]['f1']:.3f}")

best_name = max(scores, key=lambda k: scores[k]['f1'])
best_model = candidates[best_name]
pred_te = best_model.predict(X_te)

print(f"\nBEST: {best_name}  test acc={accuracy_score(y_te, pred_te):.3f}, test f1={f1_score(y_te, pred_te):.3f}")

cm = confusion_matrix(y_te, pred_te)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Plot 3: Confusion Matrix ({best_name})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("plot3_confusion_matrix.png")
plt.show()
